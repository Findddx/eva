"""Independent, default-unimported query-grid version of original M1 tail RoPE.

Only stdlib imports and metadata on import. No production hooks or GPU parity
claims. Explicit construction/build requires Linux CUDA13+/SM120. All element
math remains in the original source excerpt, with integer query offsets only.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import threading
from ds41.verification import canonical_query_capacity


SOURCE_DIR = Path(__file__).parent / 'csrc' / 'canonical_rope_batch'
CPP_FLAGS = ('-O3','-std=c++20')
CUDA_FLAGS = ('-O3','--use_fast_math','-std=c++20','-gencode=arch=compute_120,code=sm_120')
MAX_QUERIES, DIM, ROPE_DIM = canonical_query_capacity(),512,64
if MAX_QUERIES == 256:
    CUDA_FLAGS += ('-DDS41_CANONICAL_MAX_QUERIES=256',)


def workspace_metadata(max_queries=MAX_QUERIES):
    if type(max_queries) is not int or not 1 <= max_queries <= MAX_QUERIES:
        raise ValueError(f'max_queries must be integer1..{MAX_QUERIES}')
    return {'max_queries':max_queries,'heads':[1,64],'dim':DIM,'rope_channels':ROPE_DIM,
            'scratch_bytes':0,'new_data_allocation_bytes_per_call':0,'inplace':True,
            'frequency_view_aliases_input':True,'graph_capture_supported':False,
            'construction_stream_only':True}


def source_metadata():
    p = json.loads((SOURCE_DIR/'provenance.json').read_text('utf-8'))
    if hashlib.sha256(Path(__file__).with_name('attn_cuda.py').read_bytes()).hexdigest() != p['origin_file_sha256']:
        raise RuntimeError('canonical RoPE baseline changed; recertify candidate')
    if hashlib.sha256((SOURCE_DIR/'kernels.cuh').read_bytes()).hexdigest() != p['kernels_header_sha256']:
        raise RuntimeError('canonical RoPE kernel differs from provenance')
    source = hashlib.sha256()
    files = sorted((path for path in SOURCE_DIR.rglob('*') if path.is_file()),
                   key=lambda path:path.relative_to(SOURCE_DIR).as_posix())
    for path in files:
        source.update(path.relative_to(SOURCE_DIR).as_posix().encode())
        source.update(path.read_bytes())
    recipe = {'source_sha256':source.hexdigest(),'origin_file_sha256':p['origin_file_sha256'],
              'origin_device_excerpt_sha256':p['origin_device_excerpt_sha256'],
              'cpp_flags':list(CPP_FLAGS),'cuda_flags':list(CUDA_FLAGS)}
    build = hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {**recipe,'build_sha256':build,'source_files':len(files),'arch':'sm_120','max_queries':MAX_QUERIES,
            'heads':[1,64],'dim':DIM,'rope_channels':ROPE_DIM,'threads':32,'inplace':True,
            'scratch_bytes':0,'frequency_dtype':'complex64[M,32] viewed as Float[M,32,2]',
            'inverse':'original fi=-fi branch','FP8_or_fake_quant_changes':False,
            'production_integration':False,'GPU_math_validated':False,'performance_measured':False,
            'graph_capture_supported':False}


@lru_cache(maxsize=2)
def _build(build_sha256,verbose):
    import torch
    if (sys.platform != 'linux' or torch.version.cuda is None or not torch.cuda.is_available()
            or int(torch.version.cuda.split('.')[0]) < 13):
        raise RuntimeError('canonical RoPE build requires explicit Linux CUDA13+/SM120 environment')
    from torch.utils.cpp_extension import load
    return load(name='ds41_canonical_rope_'+build_sha256[:16],sources=[str(SOURCE_DIR/'binding.cu')],
                extra_cflags=list(CPP_FLAGS),extra_cuda_cflags=list(CUDA_FLAGS),verbose=bool(verbose))


def build_backend(*,verbose=False):
    return _build(source_metadata()['build_sha256'],bool(verbose))


class CanonicalRopeBatchRunner:
    def __init__(self,device='cuda',max_queries=MAX_QUERIES,*,verbose=False):
        import torch
        self.workspace_metadata = workspace_metadata(max_queries)
        self.max_queries = max_queries
        if not torch.cuda.is_available():
            raise RuntimeError('canonical RoPE runner requires real CUDA; no CPU fallback')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('canonical RoPE runner requires CUDA device')
        if self.device.index is None:
            self.device = torch.device('cuda',torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device) != (12,0):
            raise ValueError('canonical RoPE candidate requires SM120')
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('canonical RoPE construction during capture is unsupported')
        self.metadata = source_metadata()
        self._backend = build_backend(verbose=verbose)
        self._owner_stream = torch.cuda.current_stream(self.device).cuda_stream
        self._lock,self._failed = threading.Lock(),False

    def _validate(self,x,freqs,inverse):
        import torch
        if type(inverse) is not bool:
            raise ValueError('canonical RoPE inverse must be bool')
        if (not isinstance(x,torch.Tensor) or x.dtype != torch.float16
                or x.ndim not in (3,4) or x.shape[0] != 1 or not 1 <= x.shape[1] <= self.max_queries
                or x.shape[-1] != DIM or x.ndim == 4 and x.shape[2] != 64):
            raise ValueError('canonical RoPE x requires Half[1,M1..capacity,512] or Half[1,M1..capacity,64,512]')
        if (not isinstance(freqs,torch.Tensor) or freqs.dtype != torch.complex64
                or tuple(freqs.shape) != (x.shape[1],32)):
            raise ValueError('canonical RoPE frequencies require Complex64[M,32]')
        if any(t.device != self.device or not t.is_contiguous() or t.is_conj() or t.is_neg() for t in (x,freqs)):
            raise ValueError('canonical RoPE requires contiguous tensors without conjugate/neg-view flags on runner device')

    def run_(self,x,freqs,inverse=False):
        """Mutate only x's last64channels; return the same tensor object.

        Supply the original absolute-position frequency slice. No frequencies
        are generated/rebased or rounded. Producer event/wait ordering is the
        caller's duty; record_stream protects lifetime only. x/frequencies must
        stay immutable to other streams until this stream completes. Capture
        and cross-stream runner reuse are unsupported. A failed launch can leave
        x partially rotated: discard both its request state and this runner.
        """
        import torch
        if self._failed:
            raise RuntimeError('canonical RoPE execution failed; discard runner and affected state')
        self._validate(x,freqs,inverse)
        stream = torch.cuda.current_stream(self.device)
        if stream.cuda_stream != self._owner_stream:
            raise RuntimeError('canonical RoPE runner requires its construction stream')
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('canonical RoPE graph capture is unsupported')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('canonical RoPE runner is already in use')
        try:
            real = torch.view_as_real(freqs)  # Exact original interleavedFloat bytes, no data copy.
            for tensor in (x,freqs,real):
                tensor.record_stream(stream)
            self._backend.run_(x,real,int(inverse))
            return x
        except BaseException:
            self._failed = True
            raise
        finally:
            self._lock.release()
