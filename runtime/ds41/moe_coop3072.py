"""Bounded SM120 M512/coop3072 expert operator.

No Torch/CUDA import or build occurs on module import.
Callers must keep the whole actual cold union resident (at
most their existing stage_limit), pointer tables and weight owners leased until
completion. Outputs are owned clones; the raw expert targets remain borrowed.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import threading


MAX_ROWS = 512
TOPK = 6
MAX_ROUTED_SLOTS = MAX_ROWS * TOPK
KERNEL_MAX_SLOTS = 3072
SORT_SLOT_BITS = 12
UPSTREAM_REVISION = "a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a"
PARENT_SOURCE_SHA256 = "d389e386d5ec120892df7a691a4e4ceab157b830f813437c678eca75b8c5fe7b"
FLOATING_EXCERPT_SHA256 = "0ecf91b2993a9a048f86e22b6f91e7d02bd84ba36e7faf3c3e58bd6e809d7ba9"
SOURCE_DIR = Path(__file__).parent / "csrc" / "moe_coop3072"
CPP_FLAGS = ("-O3", "-std=c++20")
CUDA_FLAGS = ("-O3", "--use_fast_math", "-std=c++20", "-gencode=arch=compute_120,code=sm_120")


def restore_parent_header(data):
    """Undo ONLY namespace/capacity; parent float and integer guards unchanged."""
    text=data.decode().replace('ds41_moe_coop3072_ns','ds41_moe_coop2048_ns')
    if text.count('constexpr int MAX_SLOTS = 3072;')!=1:
        raise RuntimeError('Unexpected3072 capacity declaration')
    return text.replace('constexpr int MAX_SLOTS = 3072;','constexpr int MAX_SLOTS = 2048;').encode()


def _integer(value, lower, upper, name):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
    return value


def sort_key(expert, slot):
    """CPU specification of the native ranking key (actual scratch bound3072)."""
    return (_integer(expert, 0, 383, "expert") << SORT_SLOT_BITS) | _integer(slot, 0, 3071, "slot")


def _routes(indices):
    if not isinstance(indices, (tuple, list)) or not 1 <= len(indices) <= MAX_ROWS:
        raise ValueError("routes must have 1..512 rows")
    result = []
    for row in indices:
        if not isinstance(row, (tuple, list)) or len(row) != TOPK:
            raise ValueError("routes require six experts per row")
        row = [_integer(e, 0, 383, "expert") for e in row]
        if len(set(row)) != TOPK:
            raise ValueError("official routed top6 must be distinct within a row")
        result.append(row)
    return result


def reference_runs(indices, weights=None):
    """CPU control-flow fixture, not a numerical MoE implementation.

    weights, if given, represent ALREADY Half-rounded routing weights. Only zero
    means inactive. Each run keeps ascending original slot order, at most8.
    """
    rows = _routes(indices)
    if weights is None:
        weights = [[1.] * TOPK for _ in rows]
    if len(weights) != len(rows) or any(len(row) != TOPK for row in weights):
        raise ValueError("weights must have the routed geometry")
    active = [(sort_key(e, r*TOPK+k), e, r*TOPK+k)
              for r, row in enumerate(rows) for k, e in enumerate(row) if weights[r][k] != 0]
    active.sort()
    runs = []
    for _, expert, slot in active:
        if not runs or runs[-1]["expert"] != expert or len(runs[-1]["slots"]) == 8:
            runs.append({"expert": expert, "slots": []})
        runs[-1]["slots"].append(slot)
    return runs


def reference_table(indices, weights=None, *, capacity=512):
    """Exact int32 p.runs storage oracle, including zeroed unused entries.

    weights must already be rounded to Half. This only specifies integer routing
    metadata; it does not calculate any expert or claim floating parity.
    """
    _integer(capacity, 1, 512, 'capacity')
    rows=_routes(indices)
    if len(rows)>capacity:
        raise ValueError('route rows exceed table capacity')
    runs=reference_runs(rows,weights)
    table=[0]*(2+capacity*12+1)
    table[0]=len(runs)
    position=0
    order_offset=2+capacity*6+1
    for number,run in enumerate(runs):
        table[2+number]=position
        for slot in run['slots']:
            table[order_offset+position]=slot;position+=1
    table[2+len(runs)]=position
    return table


def residency_eligibility(indices, frozen, stage_limit=128):
    """Future caller precondition only; does not install, evict or allocate slots."""
    rows = _routes(indices)
    _integer(stage_limit, 0, 384, "stage_limit")
    frozen = {_integer(e, 0, 383, "frozen expert") for e in frozen}
    cold = list(dict.fromkeys(e for row in rows for e in row if e not in frozen))
    return {"rows": len(rows), "routed_slots": len(rows)*TOPK,
            "cold_union": cold, "cold_union_size": len(cold),
            "stage_limit": stage_limit, "eligible": len(cold) <= stage_limit}


@dataclass(frozen=True)
class WorkspaceLayout:
    max_rows: int = 512
    hidden: int = 5120
    inter: int = 2304

    def __post_init__(self):
        _integer(self.max_rows, 1, 512, "max_rows")
        if (type(self.hidden) is not int or type(self.inter) is not int
                or self.hidden != 5120 or self.inter != 2304):
            raise ValueError("prototype supports the checkpoint's H5120/I2304 only")

    @property
    def slots(self):
        return self.max_rows * TOPK

    @property
    def counter_layout(self):
        a = self.slots * (self.inter//128)
        b = self.max_rows * (self.hidden//128)
        return {"ctr_a": [0, a], "ctr_b": [a, a+b], "n_runs_reserved": [a+b, a+b+2],
                "run_start": [a+b+2, a+b+2+self.slots+1],
                "order": [a+b+2+self.slots+1, a+b+2+self.slots+1+self.slots]}

    @property
    def counter_ints(self):
        return self.counter_layout["order"][1]

    @property
    def scratch_bytes(self):
        # had_g/u Half, gu_g/u/act Half, d_out Float + out/rw/ctr.
        return (self.slots * (4*self.hidden+6*self.inter+4*self.hidden)
                + self.max_rows*self.hidden*4 + self.max_rows*TOPK*2 + self.counter_ints*4)


def source_metadata():
    """Hash project sources, not weights; also validates every vendored header."""
    provenance = json.loads((SOURCE_DIR/"provenance.json").read_text("utf-8"))
    if (provenance["revision"] != UPSTREAM_REVISION
            or provenance.get("parent_source_sha256") != PARENT_SOURCE_SHA256):
        raise RuntimeError("unexpected coop3072 source revision")
    if hashlib.sha256((SOURCE_DIR/"LICENSE.exllamav3").read_bytes()).hexdigest()!=provenance["license_sha256"]:
        raise RuntimeError("coop3072 license changed")
    for item in provenance["headers"]:
        path = SOURCE_DIR/item["vendored_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["vendored_sha256"]:
            raise RuntimeError(f"coop3072 vendored header changed: {item['vendored_path']}")
        restored=restore_parent_header(path.read_bytes()) if item['vendored_path'].endswith('exl3_moe_coop_kernel.cuh') else path.read_bytes()
        if hashlib.sha256(restored).hexdigest()!=item['parent_candidate_sha256']:
            raise RuntimeError('Undeclared parent header/arithmetic change: '+item['vendored_path'])
    source_hash = hashlib.sha256()
    # Explicit POSIX-name ordering is identical on Windows and Linux.
    files = sorted((path for path in SOURCE_DIR.rglob("*") if path.is_file()),
                   key=lambda path: path.relative_to(SOURCE_DIR).as_posix())
    for path in files:
        source_hash.update(path.relative_to(SOURCE_DIR).as_posix().encode())
        source_hash.update(path.read_bytes())
    source_sha = source_hash.hexdigest()
    recipe = {"source_sha256": source_sha, "cpp_flags": list(CPP_FLAGS), "cuda_flags": list(CUDA_FLAGS)}
    build_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    header=(SOURCE_DIR/'vendor/quant/exl3_moe_coop_kernel.cuh').read_bytes()
    marker=b'template <int bits, int cb, bool WIDE, bool HALF = false>\n__global__ __launch_bounds__(THREADS)'
    floating=header[header.index(marker):header.rindex(b'}  // namespace')]
    if hashlib.sha256(floating).hexdigest()!=FLOATING_EXCERPT_SHA256:
        raise RuntimeError("coop3072 original floating A/B source changed")
    return {"source_sha256": source_sha, "build_sha256": build_sha, "upstream_revision": UPSTREAM_REVISION,
            "python_wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parent_candidate_source_sha256": provenance['parent_source_sha256'],
            "A_B_floating_source_excerpt_sha256": hashlib.sha256(floating).hexdigest(),
            "all_vendor_bytes_restore_exactly_to_parent": True,
            "source_files": len(files), "cpp_flags": list(CPP_FLAGS), "cuda_flags": list(CUDA_FLAGS), "arch": "sm_120",
            "K": [2, 3], "codebook": "MUL1", "wide": 0, "ksplit": 1,
            "activation": "gated_silu", "topk": 6, "max_rows": 512,
            "kernel_max_slots": 3072, "scratch_max_slots": 3072,
            "math_parity_GPU_validated": False, "cache_integration": False,
            "full_model_GPU_validated": False, "production_enabled": False,
            "graph_capture_supported": False, "output_ownership": "owned clone on construction stream",
            "bounds_checks_required": True,"prepare_threads":512,"integer_prepare_shared_bytes":12296,
            "integer_prepare": "stable_O_N_squared_rank_and_prefix_external_kernel",
            "floating_geometry": {"WK":16,"WNT":2,"COLS":32,"ROWS":8,"WIDE":0,"KSPLIT":1}}


@lru_cache(maxsize=2)
def _build(build_sha256, verbose):
    import torch
    if sys.platform != "linux" or torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("coop3072 build requires an explicit Linux CUDA13+ SM120 environment")
    if int(torch.version.cuda.split('.')[0]) < 13:
        raise RuntimeError("coop3072 SM120 candidate requires CUDA13+")
    from torch.utils.cpp_extension import load
    return load(name="ds41_moe_coop3072_"+build_sha256[:16],
                sources=[str(SOURCE_DIR/"moe_coop3072.cpp"), str(SOURCE_DIR/"moe_coop3072.cu")],
                extra_cflags=list(CPP_FLAGS),
                extra_cuda_cflags=list(CUDA_FLAGS), verbose=verbose)


def build_backend(*, verbose=False):
    """Explicit JIT build request. Never called on import or by the main engine."""
    metadata = source_metadata()
    return _build(metadata["build_sha256"], bool(verbose))


class CoopRunner3072:
    def __init__(self, device="cuda", topk=6, hidden=5120, inter=2304, max_rows=512, *, verbose=False):
        import torch
        if type(topk) is not int or topk != TOPK:
            raise ValueError("coop3072 prototype requires top6")
        layout = WorkspaceLayout(max_rows, hidden, inter)
        if not torch.cuda.is_available():
            raise RuntimeError("coop3072 runner requires CUDA; no CPU numerical fallback")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("coop3072 requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device) != (12, 0):
            raise ValueError("coop3072 prototype supports SM120 only")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("coop3072 construction during capture is unsupported")
        self.layout, self.metadata = layout, source_metadata()
        self._backend = build_backend(verbose=verbose)
        self._stream = torch.cuda.current_stream(self.device)
        self._owner_stream = self._stream.cuda_stream
        self._owner_thread = threading.get_ident()
        self._lock, self._failed = threading.Lock(), False
        self._closed, self._close_drains = False, 0
        self._failed_inputs = []
        self._table_valid = False
        opt = {"device": self.device, "dtype": torch.float16}
        self.had_g = torch.empty(layout.slots, hidden, **opt)
        self.had_u = torch.empty_like(self.had_g)
        self.gu_g = torch.empty(layout.slots, inter, **opt)
        self.gu_u = torch.empty_like(self.gu_g)
        self.act_out = torch.empty_like(self.gu_g)
        self.d_out = torch.empty(layout.slots, hidden, device=self.device, dtype=torch.float32)
        self.ctr = torch.empty(layout.counter_ints, device=self.device, dtype=torch.int32)
        self.out = torch.empty(max_rows, hidden, device=self.device, dtype=torch.float32)
        self.rw = torch.empty(max_rows, TOPK, **opt)
        self._scratch = [self.had_g, self.had_u, self.gu_g, self.gu_u, self.act_out, self.d_out, self.ctr, self.out]

    def run(self, x, sel, weights, packed, k, limit, *, check_bounds=True):
        """Run actual M rows (no padding), return an OWNED Float[M,5120].

        The caller must lease all pointer targets until GPU completion. Bounds
        checks are mandatory in this isolated prototype. No cache integration,
        graph capture, extra weight staging or unchecked route adapter exists.
        """
        import torch
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("coop3072 requires its owner thread")
        if self._failed or self._closed:
            raise RuntimeError("coop3072 launch failed; discard this runner")
        if (type(k) not in (int, float) or k not in (2, 3) or type(limit) not in (int, float)
                or not math.isfinite(limit) or not 0 <= limit <= 3.4028234663852886e38):
            raise ValueError("coop3072 supports uniform K2/K3 and a finite nonnegative SiLU limit")
        if check_bounds is not True:
            raise ValueError("coop3072 requires bounds checks; unchecked entry is not provided")
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != 5120
                or not 1 <= x.shape[0] <= self.layout.max_rows or x.dtype != torch.float16):
            raise ValueError("coop3072 x must be Half[1..capacity,5120]")
        rows = x.shape[0]
        if (not isinstance(sel, torch.Tensor) or tuple(sel.shape) != (rows, 6) or sel.dtype != torch.int64
                or not isinstance(weights, torch.Tensor) or weights.shape != sel.shape
                or weights.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            raise ValueError("coop3072 routes/weights require [rows,6] Long and Half/BF16/Float")
        if not isinstance(packed, (tuple, list)) or len(packed) != 9:
            raise ValueError("coop3072 requires nine borrowed pointer tables")
        if any(not isinstance(t, torch.Tensor) or t.device != self.device for t in [x, sel, weights, *packed]):
            raise ValueError("coop3072 inputs and pointer tables must be on the runner device")
        stream = torch.cuda.current_stream(self.device)
        if stream.cuda_stream != self._owner_stream:
            raise RuntimeError("coop3072 runner requires its construction stream; no implicit cross-stream reuse")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("coop3072 graph capture is not supported")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("coop3072 runner is already in use")
        try:
            # Match the existing Half route-weight conversion, without padding M.
            x, sel = x.contiguous(), sel.contiguous()
            rw = self.rw[:rows]
            # Register every allocator lease BEFORE any backend launch may
            # partially enqueue. Raw weight-pointer targets still require the
            # caller's strong expert lease until close/completion.
            inputs = [x, sel, weights, *packed, *self._scratch, self.rw]
            self._failed_inputs = inputs
            for tensor in inputs:
                tensor.record_stream(stream)
            rw.copy_(weights.to(dtype=torch.float16))
            self._backend.run(x, sel, rw, list(packed), self._scratch, int(k), float(limit), check_bounds)
            self._table_valid = True
            output = self.out[:rows].clone()
            self._failed_inputs = []
            return output
        except BaseException:
            self._failed = True
            raise
        finally:
            self._lock.release()

    def prepare_routes(self, sel, weights, n_local):
        """Explicit diagnostic of the same integer kernel, before floating work.

        Returns an owned int32 full-capacity table. run() always rebuilds its own
        metadata and never trusts this returned table as an authorization token.
        """
        import torch
        _integer(n_local,1,384,'n_local')
        if threading.get_ident()!=self._owner_thread or self._failed or self._closed:
            raise RuntimeError('coop3072 wrong owner or failed runner')
        if (not isinstance(sel,torch.Tensor) or sel.dtype!=torch.int64 or sel.ndim!=2
                or not 1<=sel.shape[0]<=self.layout.max_rows or sel.shape[1]!=6
                or not isinstance(weights,torch.Tensor) or weights.shape!=sel.shape
                or weights.dtype not in (torch.float16,torch.bfloat16,torch.float32)
                or sel.device!=self.device or weights.device!=self.device):
            raise ValueError('prepare requires same-device bounded Long routes and floating weights')
        stream=torch.cuda.current_stream(self.device)
        with torch.cuda.device(self.device):
            capturing=torch.cuda.is_current_stream_capturing()
        if stream.cuda_stream!=self._owner_stream or capturing:
            raise RuntimeError('prepare requires construction stream without graph capture')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('coop3072 runner is already in use')
        try:
            sel=sel.contiguous();rw=self.rw[:len(sel)]
            inputs=[sel,weights,self.rw,self.ctr]
            self._failed_inputs=inputs
            for tensor in inputs:tensor.record_stream(stream)
            rw.copy_(weights.to(torch.float16))
            table=self.ctr[self.layout.counter_layout['n_runs_reserved'][0]:]
            self._backend.prepare_routes(sel,rw,table,self.layout.max_rows,n_local)
            self._table_valid = True
            output=table.clone()
            self._failed_inputs=[]
            return output
        except BaseException:
            self._failed=True;raise
        finally:self._lock.release()

    def run_table(self):
        """Owned diagnostic snapshot, on the construction thread/stream only."""
        import torch
        if threading.get_ident()!=self._owner_thread or self._failed or self._closed or not self._table_valid:
            raise RuntimeError('coop3072 wrong owner or failed runner')
        with torch.cuda.device(self.device):
            capturing=torch.cuda.is_current_stream_capturing()
        if torch.cuda.current_stream(self.device).cuda_stream!=self._owner_stream or capturing:
            raise RuntimeError('run table requires construction stream without capture')
        if not self._lock.acquire(blocking=False):raise RuntimeError('coop3072 runner is already in use')
        try:return self.ctr[self.layout.counter_layout['n_runs_reserved'][0]:].clone()
        except BaseException:
            self._failed=True;raise
        finally:self._lock.release()

    def lifecycle(self):
        """Python ownership state, no CUDA query/synchronization or certification."""
        return dict(failed=self._failed,closed=self._closed,close_drains=self._close_drains,
            owner_thread_matches=threading.get_ident()==self._owner_thread,
            failed_input_leases=len(self._failed_inputs),workspace_owners_retained=bool(self._scratch),
            caller_weight_pointer_lease_required=True,raw_addresses_disclosed=False)

    def close(self):
        """Drain the construction stream; retain inputs on any drain failure.

        Failed runners can only drain/close, never retry. This never frees
        scratch still referenced by the runner. No destructor guesses DMA
        completion. The caller must retain the runner and raw expert owners
        after a failed close, drain successfully, or terminate the context.
        """
        if threading.get_ident()!=self._owner_thread:
            raise RuntimeError('coop3072 close requires its owner thread')
        if self._closed:return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('coop3072 close while runner is active')
        try:
            try:self._stream.synchronize()
            except BaseException:
                self._failed=True;raise
            self._close_drains+=1
            self._failed_inputs=[]
            self._closed=True
        finally:self._lock.release()
