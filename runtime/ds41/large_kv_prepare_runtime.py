"""Opt-in large canonical KV packing using the unchanged legacy CUDA entry.

Disabled imports use only stdlib. This owner changes allocation/layout, not
FP4 decoding or attention arithmetic.
"""
import copy
import hashlib
import os
from pathlib import Path
import threading


FLAG = "DS41_LARGE_KV_PREPARE"
PROTOCOL = "large_canonical_sparse_kv_prepare_v1"
MAX_COMPRESSED_CAPACITY = 262144
LEGACY_SOURCE_SHA256 = "d73f4ad572cc7511d304710edc1117063e316955320b16e2c300a84cdf171fd7"
COUNTERS = ("calls", "queries", "output_requested_bytes", "indices_requested_bytes", "failures", "close_drains")
_FAILED_OWNERS = []


def _require(value, message):
    if not value:
        raise ValueError(message)


def requested(environ=None):
    value = (os.environ if environ is None else environ).get(FLAG, "0")
    _require(value in ("0", "1"), FLAG + " must be 0 or 1")
    return value == "1"


def validate_options(*, strict=True, cap=None, environ=None):
    from ds41.full_compressed_kv import requested as full_requested
    full=full_requested(environ)
    enabled=requested(environ)
    _require(not full or enabled, 'DS41_FULL_COMPRESSED_KV requires DS41_LARGE_KV_PREPARE=1')
    if not enabled:
        return False
    env = os.environ if environ is None else environ
    if cap is None:
        from ds41.verification import verification_max_rows
        cap = verification_max_rows()
    _require(strict is True and env.get("DS41_STRICT_DENSE", "1") != "0",
             "Large KV preparation requires strict canonical dense arithmetic")
    _require(type(cap) is int and (9 <= cap <= 128 or cap in (256, 512, 1024)),
             "Large KV preparation requires a canonical row cap from9..128 or256/512")
    _require(cap != 1024 or full,'Cap1024 requires full-source KV preparation')
    return True


def layout(queries, nw, supplied, *, capacity, length, ws=128, topk=512, dim=512):
    """Metadata-only integer/allocation bounds; no device scalar readbacks."""
    _require(all(type(v) is int for v in (queries, nw, supplied, capacity, length, ws, topk, dim)),
             "Large KV dimensions must be plain integers")
    _require(9 <= queries <= 512 and 1 <= nw <= queries + 127 and 0 <= supplied <= 512
             and 0 <= length <= capacity <= MAX_COMPRESSED_CAPACITY and (ws, topk, dim) == (128, 512, 512),
             f"Unsupported large KV geometry: Q={queries},Nw={nw},I={supplied},capacity={capacity},length={length},ws={ws},topk={topk},dim={dim}")
    rows = nw + queries * topk
    _require(max(rows, queries * (ws + topk)) <= 327680 and rows * dim < 2**31,
             "Large KV native integer bound exceeded")
    return dict(output_shape=(1, rows, dim), indices_shape=(1, queries, ws + topk),
                output_bytes=rows * dim * 2, indices_bytes=queries * (ws + topk) * 4)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _sources(*, full=False):
    base = Path(__file__).parent
    result = {name: _sha(base / name) for name in ("large_kv_prepare_runtime.py", "kv_prepare_cuda.py")}
    _require(result["kv_prepare_cuda.py"] == LEGACY_SOURCE_SHA256,
             "Large KV legacy CUDA source differs from the standalone primitive")
    if full:
        from ds41 import full_compressed_kv
        result.update({name:_sha(base/name) for name in ('full_compressed_kv.py','quant_cuda.py')})
        _require(result['quant_cuda.py']==full_compressed_kv.QUANT_SOURCE_SHA256,
                 'Full-source KV quant CUDA differs from the standalone prototype')
    return result


def execution_configuration(cap, *, full=None):
    """Stable source identity only; actual binary/stream addresses stay in status."""
    _require(type(cap) is int and (9 <= cap <= 128 or cap in (256, 512, 1024)), "Unsupported canonical cap")
    from ds41 import full_compressed_kv
    if full is None:full=full_compressed_kv.requested()
    _require(type(full) is bool,'Full-source mode must be explicit boolean')
    _require(cap != 1024 or full,'Cap1024 requires full-source KV preparation')
    value=dict(protocol=PROTOCOL, requested=True, verification_cap=cap,
        source_sha256=_sources(full=full), support=dict(batch=1, min_queries=9, max_queries=min(cap, 1024 if cap==1024 else 512),
            head_dim=512, window_size=128, index_topk=512, max_compressed_capacity=MAX_COMPRESSED_CAPACITY,
            dtypes=["float16", "bfloat16"], explicit_window_indices=True, contiguous=True),
        legacy_small_queries_unchanged=True, graph_capture_supported=False,
        owned_outputs=True, persistent_tensor_scratch_bytes=0, extra_JIT_modules=0,
        production_GPU_verified=False, full_model_gate_passed=False)
    if full:value['full_compressed_source']=full_compressed_kv.configuration(cap)
    if cap==1024:
        value.update(protocol='large_canonical_sparse_kv_prepare_1024_v1',mandatory_full_source_above512=True)
    return value


def _binary(extension):
    path = Path(extension.__file__).resolve(strict=True)
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    _require(root and path.is_file() and path.is_relative_to(Path(root).resolve(strict=True)),
             "Large KV extension must be inside explicit private TORCH_EXTENSIONS_DIR")
    return dict(path=str(path), bytes=path.stat().st_size, sha256=_sha(path))


class LargeKVPrepareRuntime:
    """One serial model owner; producers establish HB before calling prepare.

    record_stream protects allocator lifetime, not producer ordering. Inputs
    must remain immutable until the construction stream has consumed them.
    Returned tensors are fresh owned outputs, never reusable arena aliases.
    """
    def __init__(self, model, *, strict=True):
        from ds41.verification import verification_max_rows
        self.cap = verification_max_rows()
        validate_options(strict=strict, cap=self.cap)
        _require(requested(), "Large KV preparation requires explicit opt-in")
        from ds41 import full_compressed_kv
        self._full_requested=full_compressed_kv.requested()
        import torch
        self.device = torch.device(model.device)
        _require(self.device.type == "cuda", "Large KV preparation requires CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._thread = threading.current_thread()
        self._stream = torch.cuda.current_stream(self.device)
        self._lock = threading.Lock()
        self._closed = self._failed = self._active = False
        self._lease = None
        self._failed_leases = []
        self.counters = dict.fromkeys(COUNTERS, 0)
        self._identity = execution_configuration(self.cap,full=self._full_requested)
        from ds41 import kv_prepare_cuda
        self._extension = kv_prepare_cuda._extension()
        self._binary_identity = _binary(self._extension)
        if self._full_requested:
            from ds41 import quant_cuda
            self._full_counters=dict.fromkeys(full_compressed_kv.COUNTERS,0)
            self._full_failed_exceptions=[]
            self._quant_extension=quant_cuda._ext()
            self._quant_dequant=quant_cuda.fp4_dequant
            self._quant_binary_identity=_binary(self._quant_extension)
        self._torch_identity = dict(version=str(torch.__version__), cuda=torch.version.cuda,
                                    git_version=getattr(torch.version, "git_version", None))
        self.verify_metadata()

    def _owner(self, torch):
        _require(threading.current_thread() is self._thread, "Wrong large KV CPU owner")
        _require(not self._closed and not self._failed, "Large KV owner closed or failed")
        _require(torch.cuda.current_stream(self.device).cuda_stream == self._stream.cuda_stream,
                 "Wrong large KV CUDA stream")
        with torch.cuda.device(self.device):
            _require(not torch.cuda.is_current_stream_capturing(), "Large KV graph capture is unsupported")

    def _validate_inputs(self, window, window_idx, codes, scales, selected, length, ws, topk):
        import torch
        operands = (window, window_idx, codes, scales, selected)
        _require(all(isinstance(t, torch.Tensor) for t in operands), "Large KV tensor inputs required")
        _require(window.ndim == codes.ndim == scales.ndim == selected.ndim == 3,
                 "Large KV input rank differs")
        _require(window.shape[0] == codes.shape[0] == scales.shape[0] == selected.shape[0] == 1,
                 "Large KV supports batch1 only")
        if selected.shape[1]>512:
            from ds41 import full_compressed_kv
            _require(self.cap==1024 and getattr(self,'_full_requested',False)
                     and selected.shape[2]==512 and (ws,topk,window.shape[2])==(128,512,512),
                     'Q513..1024 requires exact1024 full-source geometry')
            meta=full_compressed_kv.layout(selected.shape[1],window.shape[1],selected.shape[2],
                                           capacity=codes.shape[1],length=length)
            _require(meta['eligible'],'Q513..1024 full-source cost rejected; eager allocation is forbidden')
        else:
            meta = layout(selected.shape[1], window.shape[1], selected.shape[2], capacity=codes.shape[1],
                          length=length, ws=ws, topk=topk, dim=window.shape[2])
        _require(selected.shape[1] <= self.cap, "Large KV queries exceed canonical startup cap")
        _require(tuple(codes.shape) == (1, codes.shape[1], 256)
                 and tuple(scales.shape) == (1, codes.shape[1], 32), "Large KV FP4 group16 shape differs")
        _require(window.dtype in (torch.float16, torch.bfloat16) and codes.dtype == torch.uint8
                 and scales.dtype == torch.float8_e4m3fn
                 and selected.dtype in (torch.int32, torch.int64)
                 and window_idx.dtype in (torch.int32, torch.int64)
                 and tuple(window_idx.shape) == (1, selected.shape[1], 128),
                 "Large KV window indices shape/dtype differs")
        _require(all(t.device == self.device and t.is_contiguous() and not t.requires_grad
                     and not t.is_conj() and not t.is_neg() for t in operands),
                 "Large KV needs contiguous inference tensors on its owner device")
        return operands, meta

    def prepare(self, window, window_idx, codes, scales, selected, length, ws=128, topk=512):
        import torch
        self._owner(torch)
        operands, meta = self._validate_inputs(window, window_idx, codes, scales, selected, length, ws, topk)
        _require(self._lock.acquire(blocking=False), "Large KV owner is already active")
        output = indices = None
        self._active = True
        try:
            # Earlier consumers have already been enqueued on this same stream.
            # Release the old strong lease BEFORE allocating the next large KV;
            # record_stream + FIFO preserve any still-running GPU consumers.
            self._lease = None
            full=getattr(self,'_full_requested',False)
            use_full=False
            if full:
                from ds41 import full_compressed_kv
                use_full=full_compressed_kv.layout(selected.shape[1],window.shape[1],selected.shape[2],
                    capacity=codes.shape[1],length=length)['eligible']
            if use_full:
                for tensor in operands:tensor.record_stream(self._stream)
                def charge(kind,nbytes):
                    if kind=='dequant':self._full_counters['full_dequant_requested_bytes']+=nbytes
                    elif kind=='output':
                        self.counters['output_requested_bytes']+=nbytes
                        self._full_counters['full_output_requested_bytes']+=nbytes
                    elif kind=='indices':self.counters['indices_requested_bytes']+=nbytes
                    else:raise ValueError('Unknown full-source allocation ledger')
                output,indices=full_compressed_kv.prepare(window,window_idx,codes,scales,selected,length,
                    dequant=self._quant_dequant,charge=charge)
                for tensor in (output,indices):tensor.record_stream(self._stream)
                self._lease=(*operands,output,indices)
                self._full_counters['full_calls']+=1
                self._full_counters['full_queries']+=selected.shape[1]
            else:
                _require(selected.shape[1]<=512,'Q513..1024 cannot fall back to the duplicate large-KV layout')
                output = torch.empty(meta["output_shape"], dtype=window.dtype, device=self.device)
                self.counters["output_requested_bytes"] += meta["output_bytes"]
                if full:self._full_counters['fallback_output_requested_bytes']+=meta['output_bytes']
                indices = torch.empty(meta["indices_shape"], dtype=torch.int32, device=self.device)
                self.counters["indices_requested_bytes"] += meta["indices_bytes"]
                for tensor in (*operands, output, indices):
                    tensor.record_stream(self._stream)
                self._lease = (*operands, output, indices)
                self._extension.prepare_kv_cuda(window, window_idx, codes, scales, selected, output, indices,
                                                length, ws, topk, 0)
                if full:
                    self._full_counters['fallback_calls']+=1
                    self._full_counters['fallback_queries']+=selected.shape[1]
            self.counters["calls"] += 1
            self.counters["queries"] += selected.shape[1]
            return output, indices
        except BaseException as error:
            self._failed = True
            self.counters["failures"] += 1
            self._failed_leases.append((*operands, output, indices))
            if getattr(self,'_full_requested',False):self._full_failed_exceptions.append(error)
            if not any(owner is self for owner in _FAILED_OWNERS):
                _FAILED_OWNERS.append(self)
            raise
        finally:
            self._active = False
            self._lock.release()

    def verify_metadata(self):
        try:
            full=getattr(self,'_full_requested',False)
            _require(_sources(full=full) == self._identity["source_sha256"], "Large KV source changed")
            _require(_binary(self._extension) == self._binary_identity, "Large KV binary changed")
            if full:
                _require(_binary(self._quant_extension)==self._quant_binary_identity,'Full-source quant binary changed')
        except BaseException:
            self._failed = True
            if not any(owner is self for owner in _FAILED_OWNERS):
                _FAILED_OWNERS.append(self)
            raise

    def close(self):
        _require(threading.current_thread() is self._thread, "Wrong large KV close CPU owner")
        _require(not self._active, "Cannot close active large KV owner")
        if not self._closed:
            try:
                self._stream.synchronize()
            except BaseException:
                self._failed = True
                if not any(owner is self for owner in _FAILED_OWNERS):
                    _FAILED_OWNERS.append(self)
                raise
            self.counters["close_drains"] += 1
            self._lease = None
            self._failed_leases.clear()
            if getattr(self,'_full_requested',False):self._full_failed_exceptions.clear()
            self._closed = True
            _FAILED_OWNERS[:] = [owner for owner in _FAILED_OWNERS if owner is not self]
        return self.report()

    def configuration(self):
        return copy.deepcopy(self._identity)

    def report(self):
        value=dict(enabled=True, closed=self._closed, failed=self._failed, active=self._active,
            identity=self.configuration(), counters=dict(self.counters),
            successful_lease_retained=self._lease is not None, failed_lease_count=len(self._failed_leases),
            actual_extension=dict(self._binary_identity), installed_torch=dict(self._torch_identity),
            allocation_counter_scope="cumulative requested output/index bytes; not peak or reserved VRAM",
            previous_successful_lease_released_before_next_allocation=True,
            inputs_immutable_until_stream_consumed=True, producer_HB_is_callers_responsibility=True,
            whole_owner_drain_on_close=True, runtime_GPU_gate_passed=False, full_model_gate_passed=False)
        if getattr(self,'_full_requested',False):
            value['full_compressed_source']=dict(enabled=True,identity=copy.deepcopy(self._identity['full_compressed_source']),
                counters=dict(self._full_counters),actual_quant_extension=dict(self._quant_binary_identity),
                failed_temporary_owners=len(self._full_failed_exceptions),single_original_owner_lease=True,
                old_output_lease_released_before_either_branch=True,cost_rejected_uses_original_CUDA_pack=True,
                allocation_counter_scope='Cumulative requested outputs/indices/dequant temporaries; not peak or reserved VRAM',
                runtime_GPU_gate_passed=False,full_model_gate_passed=False)
            if self.cap==1024:
                value['full_compressed_source']['cost_rejected_fallback_scope']='OnlyQ9..512;Q513..1024 fails before allocation'
        return value


def initialize(model, *, strict=True):
    if not validate_options(strict=strict):
        return None
    _require(getattr(model, "_large_kv_prepare_runtime", None) is None, "Large KV owner already initialized")
    args = model.args
    _require((args.max_batch_size, args.head_dim, args.window_size, args.index_topk) == (1, 512, 128, 512),
             "Large KV model geometry differs from the standalone primitive")
    runtime = LargeKVPrepareRuntime(model, strict=strict)
    model._large_kv_prepare_runtime = runtime
    for layer in model.layers:
        if layer.attn.compress_ratio:
            layer.attn._large_kv_prepare_owner = runtime
    return runtime


def configuration(model):
    owner = getattr(model, "_large_kv_prepare_runtime", None)
    return None if owner is None else owner.configuration()


def status(model):
    owner = getattr(model, "_large_kv_prepare_runtime", None)
    return None if owner is None else owner.report()


def close(model):
    owner = getattr(model, "_large_kv_prepare_runtime", None)
    return None if owner is None else owner.close()
