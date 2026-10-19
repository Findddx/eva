"""Independent, default-unimported query-grid version of original M1 tail RoPE.

Only stdlib imports and metadata on import. No production hooks or GPU parity
claims. Explicit construction/build requires Linux CUDA13+/SM120. All element
math remains in the original source excerpt, with integer query offsets only.
"""
from . import common
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import threading


SOURCE_DIR = Path(__file__).parent / "csrc" / "rope"
CPP_FLAGS = ('-O3','-std=c++20')
CUDA_FLAGS = ('-O3','--use_fast_math','-std=c++20','-gencode=arch=compute_120,code=sm_120')
MAX_QUERIES, DIM, ROPE_DIM = 1024,512,64
CUDA_FLAGS += ('-DDS41_CANONICAL_MAX_QUERIES=1024',)


def workspace_metadata(max_queries=MAX_QUERIES):
    if type(max_queries) is not int or not 1 <= max_queries <= MAX_QUERIES:
        raise ValueError(f'max_queries must be integer1..{MAX_QUERIES}')
    return {'max_queries':max_queries,'heads':[1,64],'dim':DIM,'rope_channels':ROPE_DIM,
            'scratch_bytes':0,'new_data_allocation_bytes_per_call':0,'inplace':True,
            'frequency_view_aliases_input':True,'graph_capture_supported':False,
            'construction_stream_only':True}


def _checked_provenance(source):
    return common.checked(source,"rope")[0]


def source_metadata(source):
    return common.metadata(source,"rope",__file__,CPP_FLAGS,CUDA_FLAGS)


@lru_cache(maxsize=2)
def _build(build_sha256,verbose):
    import torch
    if (sys.platform != 'linux' or torch.version.cuda is None or not torch.cuda.is_available()
            or int(torch.version.cuda.split('.')[0]) < 13):
        raise RuntimeError('canonical RoPE build requires explicit Linux CUDA13+/SM120 environment')
    from torch.utils.cpp_extension import load
    return load(name='ds41_canonical_rope1024_'+build_sha256[:16],sources=[str(SOURCE_DIR/'binding.cu')],
                extra_cflags=list(CPP_FLAGS),extra_cuda_cflags=list(CUDA_FLAGS),verbose=bool(verbose))


def build_backend(source, *, verbose=False):
    return _build(source_metadata(source)["build_sha256"],bool(verbose))


class CanonicalRope1024Runner:
    def __init__(self,device='cuda',max_queries=MAX_QUERIES,*,source,verbose=False):
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
        self.metadata = source_metadata(source)
        self._backend = build_backend(source,verbose=verbose)
        self._stream = torch.cuda.current_stream(self.device)
        self._owner_stream = self._stream.cuda_stream
        self._thread = threading.current_thread()
        self._closed = False
        self._failed_inputs = None
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
        if any(t.device != self.device or not t.is_contiguous() or t.is_conj() or t.is_neg()
               or t.requires_grad for t in (x,freqs)):
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
        if threading.current_thread() is not self._thread:
            raise RuntimeError("canonical RoPE1024 requires its construction CPU thread")
        if self._closed or self._failed:
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
            self._failed_inputs = (x,freqs)
            raise
        finally:
            self._lock.release()

    def close(self):
        """Drain the original stream before releasing async failure owners."""
        if self._closed:
            return
        if threading.current_thread() is not self._thread or not self._lock.acquire(blocking=False):
            raise RuntimeError("close requires idle construction CPU owner")
        try:
            self._stream.synchronize()
            self._closed = True
            self._failed_inputs = None
        except BaseException:
            self._failed = True
            raise
        finally:
            self._lock.release()
