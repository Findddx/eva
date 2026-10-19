"""Independent opt-in graph candidate for existing canonical FP32 M1 linears.

Nothing imports/calls this module from production. Capture preserves the M1
F.linear list and cat, rather than substituting a batched GEMM. CUDA numerical
and full-model gates are still required. Inference-tensor weights must remain
immutable: PyTorch provides no version counter with which to detect mutation.
"""
import hashlib
import os
from pathlib import Path
import threading
import time

import torch
import torch.nn.functional as F

from ds41.verification import verification_max_rows


ALLOWED_ROWS = (32, 64, 128)
MAX_WEIGHTS = 48
INPUT_WIDTH = 5120
OUTPUT_WIDTHS = (32, 384)


def canonical_linear_reference(x, weight):
    """The bias-free canonical branch of model.stable_linear, without dispatch."""
    flat = x.reshape(-1, x.shape[-1])
    rows = [F.linear(flat[i:i + 1], weight, None) for i in range(flat.shape[0])]
    y = rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)
    return y.reshape(*x.shape[:-1], weight.shape[0])


def _validate_rows(rows, canonical_cap):
    rows = tuple(rows)
    if (not rows or len(rows) > 3 or len(set(rows)) != len(rows)
            or any(type(row) is not int or row not in ALLOWED_ROWS
                   or row > canonical_cap for row in rows)):
        raise ValueError("Require distinct rows from 32/64/128 within the startup canonical cap")
    return rows


def _config():
    result = {name: os.environ.get(name) for name in (
        "CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE", "TORCH_CUBLASLT_UNIFIED_WORKSPACE",
        "NVIDIA_TF32_OVERRIDE", "DS41_STRICT_DENSE")}
    result["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    result["allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    for name in ("fp32_precision", "allow_fp16_reduced_precision_reduction",
                 "allow_bf16_reduced_precision_reduction", "allow_fp16_accumulation"):
        value = getattr(torch.backends.cuda.matmul, name, None)
        result[name] = value
    for name in ("cublas_workspace_size", "cublaslt_workspace_size"):
        getter = getattr(torch.backends.cuda, name, None)
        if getter is None and torch.version.cuda is not None:
            raise RuntimeError("CUDA workspace getter unavailable: " + name)
        try:
            result[name] = None if getter is None else int(getter())
        except (AttributeError, RuntimeError):
            if torch.version.cuda is not None:
                raise  # A CUDA build must expose its actual workspace state.
            result[name] = None
    result["preferred_blas_library"] = str(torch.backends.cuda.preferred_blas_library())
    result["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    result["deterministic_warn_only"] = torch.is_deterministic_algorithms_warn_only_enabled()
    return result


def _weight_stamp(weight):
    try:
        version = int(weight._version)
    except RuntimeError:
        if not weight.is_inference():
            raise
        version = None
    return (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()),
            weight.dtype, weight.device, version)


def _pool_idle(snapshot, handle):
    if not isinstance(snapshot, list) or any(
            not isinstance(segment, dict) or "segment_pool_id" not in segment
            or not isinstance(segment["segment_pool_id"], (tuple, list))
            or len(segment["segment_pool_id"]) != 2 for segment in snapshot):
        raise RuntimeError("Cannot verify graph allocator snapshot schema")
    own = [segment for segment in snapshot if tuple(segment["segment_pool_id"]) == tuple(handle)]
    if not own:
        raise RuntimeError("Shared graph pool missing from allocator snapshot")
    try:
        idle = all(segment["allocated_size"] == 0 and segment["active_size"] == 0
                   and all(block["state"] == "inactive" for block in segment["blocks"])
                   for segment in own)
        reserved = sum(segment["total_size"] for segment in own)
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Cannot verify graph allocator snapshot schema") from exc
    if not idle:
        raise RuntimeError("Shared graph pool retains live captured allocations")
    return dict(reserved_bytes=reserved, active_bytes=0, segments=len(own))


def source_metadata():
    return dict(schema=1, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                production_integrated=False, cuda_numerically_verified=False,
                input_width=INPUT_WIDTH, output_widths=list(OUTPUT_WIDTHS),
                allowed_rows=list(ALLOWED_ROWS), max_weights=MAX_WEIGHTS,
                arithmetic="M1 F.linear per row with bias=None, original cat, external output copy",
                inference_weight_mutation_detectable=False, owned_inference_output=True)


class CanonicalStableLinearGraphRunner:
    """One shared scratch pool and external buffers, owned results per replay.

    Construct explicitly on the inference thread/stream with resident, ready,
    immutable weights. No source tensors are copied/pinned by construction.
    Callers must establish producer happens-before for input/weight tensors.
    Capture and replay require the native allocator, a fixed precision/workspace
    configuration, and serialized use on that original thread and CUDA stream.
    """
    @torch.inference_mode()
    def __init__(self, weights, *, rows=(32, 64, 128), strict_dense=True, warmup=2):
        if strict_dense is not True or os.environ.get("DS41_STRICT_DENSE", "1") == "0":
            raise ValueError("Canonical strict-dense arithmetic must already be enabled")
        self.canonical_cap = verification_max_rows()
        self.rows = _validate_rows(rows, self.canonical_cap)
        if type(warmup) is not int or not 1 <= warmup <= 4:
            raise ValueError("warmup must be an integer from1 through4")
        if not isinstance(weights, dict) or not 1 <= len(weights) <= MAX_WEIGHTS:
            raise ValueError("Require1..48 explicitly named existing weights")
        self.weights = dict(weights)  # Strong references, no weight duplication.
        for key, weight in self.weights.items():
            if (not isinstance(key, str) or not key or not isinstance(weight, torch.Tensor)
                    or weight.ndim != 2 or weight.shape[1] != INPUT_WIDTH
                    or weight.shape[0] not in OUTPUT_WIDTHS or weight.dtype != torch.float32
                    or not weight.is_cuda or not weight.is_contiguous() or weight.requires_grad):
                raise ValueError("Weights must be named, contiguous CUDA FP32 [32|384,5120], no grad")
        self.device = next(iter(self.weights.values())).device
        if any(weight.device != self.device for weight in self.weights.values()):
            raise ValueError("All weights must share one CUDA device")
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError("Construction device must be the current CUDA device")
        if torch.cuda.memory.get_allocator_backend() != "native":
            raise RuntimeError("Shared graph scratch requires the native CUDA allocator")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot construct inside CUDA graph capture")
        stream = torch.cuda.current_stream(self.device)
        self._owner = (threading.current_thread(), stream.device, stream.cuda_stream)
        self._lock = threading.Lock()
        self._busy = False
        self._closed = False
        self._poisoned = False
        self._config = _config()
        self._stamps = {key: _weight_stamp(weight) for key, weight in self.weights.items()}
        self._graphs = {}
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.pool = torch.cuda.graph_pool_handle()
        self.pool_memory = None
        start = time.perf_counter()
        torch.cuda.synchronize(self.device)
        before = self._memory()
        # Both allocations precede capture and are never allocated in its pool.
        self._input = torch.zeros(max(self.rows), INPUT_WIDTH, dtype=torch.float32, device=self.device)
        self._output = torch.empty(max(self.rows) * max(OUTPUT_WIDTHS), dtype=torch.float32, device=self.device)
        try:
            self.capture_stream.wait_stream(stream)
            for key, weight in self.weights.items():
                weight.record_stream(self.capture_stream)
                for count in self.rows:
                    with torch.cuda.stream(self.capture_stream):
                        for _ in range(warmup):
                            self._body(key, count)
                    self.capture_stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=self.pool, stream=self.capture_stream):
                        self._body(key, count)
                    # _body returns None: no pool-allocated Tensor may escape.
                    torch.cuda.synchronize(self.device)
                    self.pool_memory = _pool_idle(torch.cuda.memory_snapshot(), self.pool)
                    self._graphs[key, count] = graph
            self._check_configuration()
        except BaseException:
            self._poisoned = True
            torch.cuda.synchronize(self.device)
            self._graphs.clear()
            raise
        after = self._memory()
        self.startup = dict(seconds=time.perf_counter()-start, before=before, after=after,
                            allocated_delta=after["allocated_bytes"]-before["allocated_bytes"],
                            reserved_delta=after["reserved_bytes"]-before["reserved_bytes"],
                            graph_count=len(self._graphs), pool=dict(self.pool_memory))

    def _memory(self):
        free, total = torch.cuda.mem_get_info(self.device)
        return dict(allocated_bytes=torch.cuda.memory_allocated(self.device),
                    reserved_bytes=torch.cuda.memory_reserved(self.device),
                    device_free_bytes=free, device_total_bytes=total)

    def _body(self, key, count):
        weight = self.weights[key]
        value = canonical_linear_reference(self._input[:count], weight)
        self._output[:count * weight.shape[0]].view(count, weight.shape[0]).copy_(value)

    def _check_owner(self):
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError("Current CUDA device changed")
        stream = torch.cuda.current_stream(self.device)
        if (threading.current_thread(), stream.device, stream.cuda_stream) != self._owner:
            raise RuntimeError("Require original CPU thread and CUDA stream")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Runner cannot replay inside CUDA graph capture")
        return stream

    def _check_configuration(self, key=None):
        if verification_max_rows() != self.canonical_cap or _config() != self._config:
            self._poisoned = True
            raise RuntimeError("Canonical cap or precision/workspace configuration changed")
        items = self.weights.items() if key is None else ((key, self.weights[key]),)
        if any(_weight_stamp(weight) != self._stamps[name] for name, weight in items):
            self._poisoned = True
            raise RuntimeError("Bound weight storage, geometry or tracked version changed")

    @property
    def buffer_metadata(self):
        return dict(input_bytes=self._input.numel()*self._input.element_size(),
                    output_bytes=self._output.numel()*self._output.element_size(),
                    borrowed_weight_bytes=sum(weight.numel()*weight.element_size() for weight in self.weights.values()),
                    weight_copies=0, owned_output=True, inference_weights_without_version=sum(
                        stamp[-1] is None for stamp in self._stamps.values()), graph_pool=self.pool_memory)

    @torch.inference_mode()
    def run(self, key, x):
        if self._closed or self._poisoned:
            raise RuntimeError("Runner is closed or poisoned")
        stream = self._check_owner()
        if key not in self.weights:
            raise ValueError("Unknown weight key")
        self._check_configuration(key)
        if (not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or x.device != self.device
                or not x.is_contiguous() or x.requires_grad or x.ndim not in (2, 3)
                or x.shape[-1] != INPUT_WIDTH or (x.ndim == 3 and x.shape[0] != 1)):
            raise ValueError("Input must be contiguous CUDA FP32 [M,5120] or[1,M,5120], no grad")
        count = x.numel() // INPUT_WIDTH
        if (key, count) not in self._graphs:
            raise ValueError("Exact row count was not captured")
        with self._lock:
            if self._busy:
                raise RuntimeError("Runner cannot reenter")
            self._busy = True
        try:
            x.record_stream(stream)
            self._input[:count].copy_(x.reshape(count, INPUT_WIDTH))
            self._graphs[key, count].replay()
            out = self._output[:count * self.weights[key].shape[0]]
            # The clone is enqueued before any later replay overwrites scratch.
            return out.reshape(*x.shape[:-1], self.weights[key].shape[0]).clone()
        except BaseException:
            self._poisoned = True
            torch.cuda.synchronize(self.device)
            raise
        finally:
            with self._lock:
                self._busy = False

    def close(self):
        """Drain before releasing graphs, borrowed weights or shared buffers."""
        if self._closed:
            return
        if threading.current_thread() is not self._owner[0] or self._busy:
            raise RuntimeError("Close requires the original CPU thread and no active call")
        torch.cuda.synchronize(self.device)
        self._graphs.clear()
        self.weights.clear()
        self._input = self._output = None
        self._closed = True
