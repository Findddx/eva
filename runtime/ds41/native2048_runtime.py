"""Default-off production ownership for the unchanged bounded coop2048 runner.

No CUDA import/allocation occurs on import or when the flag is disabled. The
standalone primitive's metadata remains unmodified; production certification
must come from a new external full-model gate.
"""
import copy
import hashlib
import os
from pathlib import Path
import threading


FLAG = "DS41_MOE_COOP2048"
COUNTERS = ("calls", "rows", "routed_slots", "failures")
PROTOCOL = "checked_M65_256_actual_union_coop2048_v1"


def requested(environment=None):
    value = (os.environ if environment is None else environment).get(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(FLAG+" must be 0 or 1")
    return value == "1"


def validate_configuration(environment=None, *, cap, speculation=None, prefill_chunk_size=None,
                           volatile_experts=None, cpu_experts=False, adaptive=False, next_chunk=False):
    env = os.environ if environment is None else environment
    if not requested(env):
        return dict(requested=False)
    if cap not in (128, 256):
        raise ValueError("coop2048 production requires explicit target cap128 or256")
    required = {"DS41_MOE_COOP512":"1", "DS41_FAST_POINTER_BIND":"1", "DS41_ROUTE_SNAPSHOT":"0",
        "DS41_ROUTE_AWARE_CHUNKS":"1", "DS41_STRICT_BATCH":"1", "DS41_STRICT_DENSE":"1",
        "DS41_FUSED_HC":"0", "DS41_DENSE_CANONICAL_TILE":"32", "DS41_NEXT_CHUNK_PREFETCH":"0",
        "DS41_DEMAND_H2D_PIPELINE":"1", "EXL3_MOE_COOP_WIDE":"0", "EXL3_MOE_COOP_KSPLIT":"1"}
    for name,value in required.items():
        if env.get(name) != value:
            raise ValueError("coop2048 requires "+name+"="+value)
    if speculation is not None and speculation != "lookup":
        raise ValueError("coop2048 production first supports lookup only")
    if prefill_chunk_size is not None and prefill_chunk_size != 128:
        raise ValueError("coop2048 production retains prefill chunk128")
    if volatile_experts is not None and (type(volatile_experts) is not int or not 6 <= volatile_experts <= 256):
        raise ValueError("coop2048 requires6..256 actual volatile slots")
    if cpu_experts or adaptive or next_chunk:
        raise ValueError("coop2048 excludes CPU experts, continuous adaptive and next-chunk prefetch")
    return dict(requested=True, protocol=PROTOCOL, verification_cap=cap, minimum_native_rows=65,
        maximum_native_rows=256, maximum_routed_slots=1536, maximum_volatile_slots=256,
        observer_max384_keeps_original64_row_calls=True, GPU_bounds_checks=True,
        original512_remains_enabled=True, required_environment=required,
        production_GPU_certificate=False, full_model_GPU_certificate=False)


def _sources():
    root = Path(__file__).parent
    names = ("native2048_runtime.py", "cache.py", "route_chunks.py", "verification.py", "generate.py")
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def _binary(runner):
    path = Path(runner._backend.__file__).resolve(strict=True)
    private = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not private or not path.is_relative_to(Path(private).resolve(strict=True)):
        raise RuntimeError("coop2048 consumer must reside in explicit private TORCH_EXTENSIONS_DIR")
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def disabled_status(cache=None):
    return dict(requested=False, effective=False, production_integrated=False, closed=False,
        failed=False, identity=dict(requested=False), counters=dict.fromkeys(COUNTERS, 0))


class Native2048Runtime:
    def __init__(self, cache, device, *, volatile_experts):
        # An attached, partially initialized owner must remain reportable and
        # closable even if the first CUDA API/import raises.
        self._cache_id = id(cache)
        self._thread = threading.get_ident()
        self._stream = None
        self._runner = None
        self._closed = self._failed = False
        self.counters = dict.fromkeys(COUNTERS, 0)
        self._identity = None
        self.configuration = None
        import torch
        from ds41.verification import verification_max_rows
        from ds41.moe_coop2048 import CoopRunner2048, WorkspaceLayout, source_metadata
        self.configuration = validate_configuration(cap=verification_max_rows(), volatile_experts=volatile_experts)
        self._stream = torch.cuda.current_stream(device)
        self._validate_cache(cache, placement=False)
        before = self._memory(torch, device)
        self._runner = CoopRunner2048(device, topk=6, hidden=5120, inter=2304, max_rows=256)
        self._stream.synchronize()
        after = self._memory(torch, device)
        self._identity = dict(self.configuration, source_sha256=_sources(), native_build=source_metadata(),
            extension=_binary(self._runner), nominal_workspace_bytes=WorkspaceLayout().scratch_bytes,
            device=str(self._runner.device), stream_contract="construction_stream_only",
            allocated_before_expert_budget=True, constructor_memory=dict(before=before,after=after,
                allocated_delta_bytes=after["allocated"]-before["allocated"],
                reserved_delta_bytes=after["reserved"]-before["reserved"],
                device_free_delta_bytes=before["free"]-after["free"]),
            original512_workspace_is_additional=True, new_copy_streams=0, new_weight_slots=0,
            runtime_GPU_gate_passed=False, full_model_gate_passed=False)

    @staticmethod
    def _memory(torch, device):
        return dict(allocated=torch.cuda.memory_allocated(device), reserved=torch.cuda.memory_reserved(device),
                    free=torch.cuda.mem_get_info(device)[0])

    def _validate_cache(self, cache, *, placement=True):
        if id(cache) != self._cache_id or threading.get_ident() != self._thread:
            raise RuntimeError("coop2048 requires its original cache/CPU owner")
        if self._closed or self._failed:
            raise RuntimeError("coop2048 runtime is closed or failed")
        from ds41.pointer_table import FastPointerTable
        table = getattr(cache, "_fast_pointer_table", None)
        if (not isinstance(table, FastPointerTable) or table.host is not cache.host_ptrs
                or getattr(cache, "_route_snapshot_runtime", None) is not None
                or getattr(cache, "cpu_backend", None) is not None or not cache.retain_host
                or getattr(cache, "adaptive_gpu_enabled", False)
                or getattr(cache, "_demand_h2d_pipeline", None) is None
                or getattr(cache, "_chunk_prefetch", None) is not None):
            raise RuntimeError("coop2048 cache dependencies changed or are incompatible")
        if placement and (type(cache.stage_limit) is not int or not 6 <= cache.stage_limit <= 256):
            raise RuntimeError("coop2048 actual volatile capacity must remain6..256")

    def run(self, cache, x, indices, weights, packed, k, limit):
        import torch
        self._validate_cache(cache)
        if not 65 <= x.shape[0] <= min(256, self.configuration["verification_cap"]):
            raise ValueError("production coop2048 is restricted to65..target-cap rows")
        if torch.cuda.current_stream(x.device).cuda_stream != self._stream.cuda_stream:
            raise RuntimeError("coop2048 requires its actual construction stream")
        try:
            result = self._runner.run(x, indices, weights, packed, k, limit, check_bounds=True)
        except BaseException:
            self.poison()
            cache._coop2048_failed = True
            raise
        self.counters["calls"] += 1
        self.counters["rows"] += x.shape[0]
        self.counters["routed_slots"] += indices.numel()
        return result

    def verify_metadata(self):
        if self._closed or self._failed or threading.get_ident() != self._thread:
            raise RuntimeError("coop2048 metadata requested from closed/failed/foreign runtime")
        from ds41.moe_coop2048 import source_metadata
        from ds41.verification import verification_max_rows
        if (validate_configuration(cap=verification_max_rows()) != self.configuration
                or _sources() != self._identity["source_sha256"]
                or source_metadata() != self._identity["native_build"]
                or _binary(self._runner) != self._identity["extension"]):
            self.poison()
            raise RuntimeError("coop2048 source/build/consumer changed")

    def poison(self):
        if not self._failed:
            self.counters["failures"] += 1
        self._failed = True

    def report(self):
        return dict(requested=True, effective=self._identity is not None, production_integrated=True, closed=self._closed,
            failed=self._failed, identity=copy.deepcopy(self._identity), counters=dict(self.counters),
            counter_scope="Successful checked65..256 actual target calls; prefill and rejected verifier rows included")

    def close(self):
        if self._closed:
            return
        if threading.get_ident() != self._thread:
            self.poison()
            raise RuntimeError("coop2048 close requires original CPU owner")
        # Drain the recorded actual stream even if the caller changed streams.
        # Keep runner/scratch strongly owned on failure and after close for the
        # external lifecycle receipt; no clearing/freeing a live DMA target.
        try:
            if self._stream is not None:
                self._stream.synchronize()
        except BaseException:
            self.poison()
            raise
        self._closed = True


def initialize(cache, device, *, volatile_experts):
    if not getattr(cache, "_coop2048_requested", False):
        return
    if getattr(cache, "_native2048_runtime", None) is not None:
        raise RuntimeError("coop2048 cannot be initialized twice")
    runtime = Native2048Runtime.__new__(Native2048Runtime)
    cache._native2048_runtime = runtime  # Strong owner before any allocation.
    try:
        runtime.__init__(cache, device, volatile_experts=volatile_experts)
    except BaseException:
        cache._coop2048_failed = True
        runtime.poison()
        try:
            runtime.close()
        except BaseException:
            pass  # Retain poisoned runtime and original error.
        raise


def status(cache):
    runtime = getattr(cache, "_native2048_runtime", None)
    if runtime is not None:
        report = runtime.report()
        stats = getattr(cache, "stats", {})
        report["cache_fallback_counts"] = {name:stats.get("moe_coop2048_fallback_"+name,0)
                                           for name in ("calls", "rows")}
        return report
    if getattr(cache, "_coop2048_requested", False):
        return dict(disabled_status(), requested=True, failed=getattr(cache,"_coop2048_failed",False),
                    initialization_pending=True)
    return disabled_status()
