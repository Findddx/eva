"""Independent default-unimported canonical sparse-attention query-grid candidate.

Imports only stdlib. Construction/build explicitly requires CUDA13+/SM120.
The original M1 per-query/head arithmetic is retained; no GPU parity or speed
claim follows from source preservation. No cache/model production hook exists.
"""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
from ds41.verification import canonical_query_capacity


SOURCE_DIR = Path(__file__).parent / "csrc" / "canonical_sparse_batch"
MAX_QUERIES, HEADS, DIM, MAX_KEYS = canonical_query_capacity(), 64, 512, 640
MAX_KV_ROWS = 127 + MAX_QUERIES * (1 + 512)
DEFAULT_QUERY_TILE = 32
CPP_FLAGS = ("-O3", "-std=c++20")
CUDA_FLAGS = ("-O3", "--use_fast_math", "-std=c++20", "-gencode=arch=compute_120,code=sm_120")
if MAX_QUERIES == 256:
    CUDA_FLAGS += ("-DDS41_CANONICAL_MAX_QUERIES=256",)


def _integer(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return value


def query_spans(queries, query_tile=DEFAULT_QUERY_TILE):
    _integer(queries,1,MAX_QUERIES,"queries")
    _integer(query_tile,1,MAX_QUERIES,"query_tile")
    return [(start,min(start+query_tile,queries)) for start in range(0,queries,query_tile)]


def workspace_metadata(query_tile=DEFAULT_QUERY_TILE, max_queries=MAX_QUERIES):
    _integer(query_tile,1,MAX_QUERIES,"query_tile")
    _integer(max_queries,1,MAX_QUERIES,"max_queries")
    if query_tile > max_queries:
        raise ValueError("query_tile exceeds max_queries")
    parts = (MAX_KEYS+63)//64
    partial = query_tile*HEADS*parts*DIM*4
    stats = query_tile*HEADS*parts*2*4
    return {"query_tile":query_tile,"max_queries":max_queries,"max_keys":MAX_KEYS,"max_parts":parts,
            "partial_bytes":partial,"stats_bytes":stats,"scratch_bytes":partial+stats,
            "maximum_owned_output_bytes":max_queries*HEADS*DIM*2,
            "scratch_plus_max_output_bytes":partial+stats+max_queries*HEADS*DIM*2,
            "output_ownership":"fresh owned tensor per run","same_stream_only":True,
            "graph_capture_supported":False,"weight_storage_bytes":0}


def source_metadata():
    provenance = json.loads((SOURCE_DIR/"provenance.json").read_text("utf-8"))
    original = Path(__file__).with_name("attn_cuda.py")
    if hashlib.sha256(original.read_bytes()).hexdigest() != provenance["origin_file_sha256"]:
        raise RuntimeError("canonical sparse baseline source changed; recertify this candidate")
    if hashlib.sha256((SOURCE_DIR/"kernels.cuh").read_bytes()).hexdigest() != provenance["kernels_header_sha256"]:
        raise RuntimeError("canonical sparse kernel differs from recorded provenance")
    source = hashlib.sha256()
    files = sorted((p for p in SOURCE_DIR.rglob('*') if p.is_file()),key=lambda p:p.relative_to(SOURCE_DIR).as_posix())
    for path in files:
        source.update(path.relative_to(SOURCE_DIR).as_posix().encode())
        source.update(path.read_bytes())
    recipe = {"source_sha256":source.hexdigest(),"origin_file_sha256":provenance["origin_file_sha256"],
              "origin_device_excerpt_sha256":provenance["origin_device_excerpt_sha256"],
              "cpp_flags":list(CPP_FLAGS),"cuda_flags":list(CUDA_FLAGS)}
    build = hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {**recipe,"build_sha256":build,"source_files":len(files),"arch":"sm_120",
            "max_queries":MAX_QUERIES,"heads":HEADS,"dim":DIM,"max_keys":MAX_KEYS,"max_kv_rows":MAX_KV_ROWS,
            "default_query_tile":DEFAULT_QUERY_TILE,"keys_per_part":64,"threads":256,
            "original_K_preserved":True,"query_padding":False,"FP8_or_FP4_requantization":False,
            "sink_policy":"unchanged original per-part maximum; denominator includes sink only in part0",
            "production_integration":False,"GPU_math_validated":False,"performance_measured":False,
            "output_ownership":"fresh owned","graph_capture_supported":False}


@lru_cache(maxsize=2)
def _build(build_sha256, verbose):
    import torch
    if (sys.platform != 'linux' or torch.version.cuda is None or not torch.cuda.is_available()
            or int(torch.version.cuda.split('.')[0]) < 13):
        raise RuntimeError("canonical sparse build requires explicit Linux CUDA13+/SM120 environment")
    from torch.utils.cpp_extension import load
    return load(name="ds41_canonical_sparse_"+build_sha256[:16],sources=[str(SOURCE_DIR/"binding.cu")],
                extra_cflags=list(CPP_FLAGS),extra_cuda_cflags=list(CUDA_FLAGS),verbose=bool(verbose))


def build_backend(*, verbose=False):
    return _build(source_metadata()["build_sha256"],bool(verbose))


class CanonicalSparseBatchRunner:
    def __init__(self, device="cuda", query_tile=DEFAULT_QUERY_TILE, max_queries=MAX_QUERIES, *, verbose=False):
        import torch
        self.workspace_metadata = workspace_metadata(query_tile,max_queries)
        self.query_tile, self.max_queries = query_tile,max_queries
        if not torch.cuda.is_available():
            raise RuntimeError("canonical sparse runner requires real CUDA; no CPU fallback")
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError("canonical sparse runner requires CUDA device")
        if self.device.index is None:
            self.device = torch.device('cuda',torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device) != (12,0):
            raise ValueError("canonical sparse candidate requires SM120")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("canonical sparse construction during capture is unsupported")
        self.metadata = source_metadata()
        self._backend = build_backend(verbose=verbose)
        self._owner_stream = torch.cuda.current_stream(self.device).cuda_stream
        self._lock, self._failed = threading.Lock(),False
        self.partial = torch.empty(self.workspace_metadata['partial_bytes']//4,device=self.device,dtype=torch.float32)
        self.stats = torch.empty(self.workspace_metadata['stats_bytes']//4,device=self.device,dtype=torch.float32)

    def _validate(self, q, kv, sink, idx, scale):
        import torch
        if (type(scale) not in (float,int) or not math.isfinite(scale) or abs(scale)>3.4028234663852886e38):
            raise ValueError("canonical sparse scale must be finite Float-range")
        if (not isinstance(q,torch.Tensor) or q.ndim != 4 or q.shape[0] != 1 or q.shape[2:] != (HEADS,DIM)
                or not 1 <= q.shape[1] <= self.max_queries or q.dtype != torch.float16):
            raise ValueError("canonical sparse q requires Half[1,M1..capacity,64,512]")
        rows = q.shape[1]
        if (not isinstance(kv,torch.Tensor) or kv.ndim != 3 or kv.shape[0] != 1 or kv.shape[2] != DIM
                or not 0 <= kv.shape[1] <= MAX_KV_ROWS or kv.dtype != torch.float16):
            raise ValueError(f"canonical sparse kv requires Half[1,N0..{MAX_KV_ROWS},512]")
        if (not isinstance(sink,torch.Tensor) or tuple(sink.shape) != (HEADS,) or sink.dtype != torch.float32):
            raise ValueError("canonical sparse sink requires Float[64]")
        if (not isinstance(idx,torch.Tensor) or idx.ndim != 3 or tuple(idx.shape[:2]) != (1,rows)
                or not 1 <= idx.shape[2] <= MAX_KEYS or idx.dtype != torch.int32):
            raise ValueError("canonical sparse indices require Int32[1,M,K1..640]")
        if any(t.device != self.device or not t.is_contiguous() for t in (q,kv,sink,idx)):
            raise ValueError("canonical sparse inputs must be contiguous on the runner device")
        return rows

    def run(self, q, kv, sink, idx, scale):
        """Owned Half output; no new indices, visibility, quantization or math.

        Invalid/high/negative/duplicate indices keep the original M1 handling.
        Inputs must already be visible on this stream: producer event/wait
        ordering is the caller's duty; record_stream protects lifetime only.
        Inputs/KV must remain immutable until the current stream completes.
        Scratch is reused only on its construction stream; capture is refused.
        """
        import torch
        if self._failed:
            raise RuntimeError("canonical sparse execution failed; discard runner")
        rows = self._validate(q,kv,sink,idx,scale)
        stream = torch.cuda.current_stream(self.device)
        if stream.cuda_stream != self._owner_stream:
            raise RuntimeError("canonical sparse runner requires its construction stream")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("canonical sparse graph capture is unsupported")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("canonical sparse runner is already in use")
        try:
            # Separate output storage survives later invocations of this runner.
            out = torch.empty_like(q)
            # Register allocator stream uses BEFORE any launch, so a partial
            # launch failure still retires even inputs allocated on other streams.
            for tensor in (q,kv,sink,idx,out,self.partial,self.stats):
                tensor.record_stream(stream)
            for start,end in query_spans(rows,self.query_tile):
                self._backend.run(q[:,start:end],kv,sink,idx[:,start:end],out[:,start:end],
                                  self.partial,self.stats,float(scale))
            return out
        except BaseException:
            self._failed = True
            raise
        finally:
            self._lock.release()
