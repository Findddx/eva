"""Isolated square+mean candidate using the installed Torch M1 row tree.

No production imports/integration, CPU fallback or numerical certification.
Only stdlib is imported until explicit construction in a Linux CUDA environment.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import threading


SOURCE_DIR = Path(__file__).parent / "csrc" / "canonical_mean_square"
WIDTHS = (128, 512, 1280, 5120, 20480)
MAX_ROWS = 256
TORCH_VERSION = "2.14.1+cu130"
TORCH_GIT = "5c4886908584029761b579af026dcfb627c84070"
CPP_FLAGS = ("-O3", "-std=c++20")
CUDA_FLAGS = ("-O3", "-std=c++20", "--fmad=false", "--ftz=false", "--prec-div=true",
              "--prec-sqrt=true", "-gencode=arch=compute_120,code=sm_120")


def m1_geometry(width):
    """Derive this allowlist from the pinned ReduceConfig rules, not batch M."""
    if type(width) is not int or width not in WIDTHS:
        raise ValueError("mean_square width must be one of " + str(WIDTHS))
    dim0 = width // 4  # Contiguous M1 float input; vectorize_input enabled.
    dim0_pow2 = min(512, 1 << (dim0.bit_length() - 1))
    block_x = min(dim0_pow2, 512)  # dim1=1 -> block_height=1.
    values_per_thread = (width + block_x - 1)//block_x
    split_y = values_per_thread >= 16
    if values_per_thread >= 256:
        raise ValueError("allowlist unexpectedly requires multi-CTA reduction")
    return {"block_x": block_x, "block_y": 1, "ctas_per_output": 1,
            "vectorize_input": True, "output_vec_size": 1, "step_input": block_x,
            "step_output": 1, "input_mult_x": 1, "input_mult_y": block_x if split_y else 0,
            "input_mult_cta": 0, "output_mult_x": 0, "output_mult_y": 0 if split_y else 1,
            "num_inputs": width, "num_outputs": 1,
            "factor_bits": struct.unpack("<I", struct.pack("<f", 1.0/width))[0]}


def source_metadata():
    digest = hashlib.sha256()
    files = sorted(p for p in SOURCE_DIR.iterdir() if p.is_file())
    for path in files:
        digest.update(path.name.encode());digest.update(path.read_bytes())
    recipe = {"source_sha256": digest.hexdigest(), "cpp_flags": list(CPP_FLAGS), "cuda_flags": list(CUDA_FLAGS),
              "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch_version": TORCH_VERSION, "torch_git": TORCH_GIT}
    build = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**recipe, "build_sha256": build, "source_files": len(files), "arch": "sm_120",
            "widths": list(WIDTHS), "max_rows": MAX_ROWS, "output_dtype": "float32",
            "float_conversion": "original x.reshape(-1,D).float() before the fused primitive",
            "production_integration": False, "GPU_math_validated": False, "performance_measured": False,
            "graph_capture_supported": False, "output_ownership": "fresh owned",
            "persistent_scratch_bytes": 0, "max_float_conversion_bytes": MAX_ROWS*max(WIDTHS)*4}


def validate_installed_torch(torch):
    if str(torch.__version__) != TORCH_VERSION or torch.version.git_version != TORCH_GIT or torch.version.cuda != "13.0":
        raise RuntimeError("Canonical mean requires the compatible Torch2.14.1/cu130 reduction backend")
    core = Path(torch._C.__file__).resolve(strict=True)
    core_sha = hashlib.sha256(core.read_bytes()).hexdigest()
    root = Path(torch.__file__).resolve().parent/"include"
    provenance = json.loads((SOURCE_DIR/"provenance.json").read_bytes())
    records = {}
    for relative, expected in provenance["installed_header_sha256"].items():
        path = root/relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Installed ATen reduction header differs: " + relative)
        records[relative] = actual
    for name in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS", "CUDAFLAGS", "CXXFLAGS", "TORCH_NVCC_FLAGS"):
        if os.environ.get(name):
            raise RuntimeError("Unexpected compiler flag environment: " + name)
    return {"torch_version": str(torch.__version__), "torch_git": torch.version.git_version,
            "cuda": torch.version.cuda, "installed_header_sha256": records,
            "torch_core_binary": {"path": str(core), "sha256": core_sha}}


@lru_cache(maxsize=1)
def _build(build_sha, verbose):
    import torch
    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError("Explicit Linux SM120 CUDA environment required; no CPU fallback")
    validate_installed_torch(torch)
    from torch.utils.cpp_extension import load
    return load(name="ds41_mean_square_"+build_sha[:16], sources=[str(SOURCE_DIR/"binding.cu")],
                extra_cflags=list(CPP_FLAGS), extra_cuda_cflags=list(CUDA_FLAGS), verbose=verbose)


class CanonicalMeanSquareRunner:
    def __init__(self, device="cuda", *, verbose=False):
        import torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("mean_square candidate requires a real CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device) != (12, 0):
            raise ValueError("mean_square candidate requires SM120")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("mean_square construction during capture unsupported")
        self.installed = validate_installed_torch(torch)
        self.metadata = source_metadata()
        self._backend = _build(self.metadata["build_sha256"], bool(verbose))
        self._stream = torch.cuda.current_stream(self.device)
        self._thread = threading.current_thread()
        self._lock = threading.Lock()
        self._failed = self._closed = False
        self.configs = {}
        self.calls = self.rows = 0

    def _validate(self, x):
        import torch
        if self._closed or self._failed:
            raise RuntimeError("mean_square runner closed/failed; discard runner")
        if threading.current_thread() is not self._thread:
            raise RuntimeError("mean_square requires its construction CPU thread")
        if (not isinstance(x, torch.Tensor) or x.device != self.device or x.dtype not in
                (torch.float16, torch.bfloat16, torch.float32) or x.ndim < 2 or x.shape[-1] not in WIDTHS
                or not 1 <= x.numel()//x.shape[-1] <= MAX_ROWS or not x.is_contiguous()
                or x.is_conj() or x.is_neg() or x.requires_grad):
            raise ValueError("mean_square requires contiguous inference FP16/BF16/FP32 rows1..256 and an allowed width")
        if x.dtype == torch.float32 and x.data_ptr() % 16:
            raise ValueError("mean_square requires actual 16B-aligned float rows; no implicit copy/fallback")
        stream = torch.cuda.current_stream(self.device)
        if stream.cuda_stream != self._stream.cuda_stream:
            raise RuntimeError("mean_square requires its construction CUDA stream")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("mean_square graph capture unsupported")
        return stream

    def run(self, x):
        """Return mean_square only. Keep source immutable and order its producer.

        record_stream is a lifetime lease, not a producer/consumer wait. Calls
        and close are serialized on the constructor's CPU thread/CUDA stream.
        No finite-input D2H check or silent fallback is inserted in this path.
        """
        import torch
        stream = self._validate(x)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("mean_square runner is already in use")
        try:
            flat = x.reshape(-1, x.shape[-1]).float()  # Same original conversion.
            for tensor in (x, flat): tensor.record_stream(stream)
            width = int(flat.shape[-1])
            if width not in self.configs:
                actual = dict(self._backend.aten_m1_config(flat[:1]))
                expected = m1_geometry(width)
                if actual != expected or any(type(actual[k]) is not type(expected[k]) for k in expected):
                    raise RuntimeError("Actual installed ATen M1 configuration differs from candidate schedule")
                self.configs[width] = actual
            output = self._backend.run(flat)
            output.record_stream(stream)
            self.calls += 1;self.rows += flat.shape[0]
            return output.reshape(*x.shape[:-1], 1)
        except BaseException:
            self._failed = True
            raise
        finally:
            self._lock.release()

    def close(self):
        if self._closed:return
        if threading.current_thread() is not self._thread or not self._lock.acquire(blocking=False):
            raise RuntimeError("mean_square close needs idle original CPU owner")
        try:
            self._stream.synchronize()
            self._closed = True
        except BaseException:
            self._failed = True
            raise
        finally:
            self._lock.release()
