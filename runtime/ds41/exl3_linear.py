"""EXL3 linears with M1/M2 arithmetic and a canonical M32 execution path."""
from __future__ import annotations

import os
from functools import lru_cache

import torch

from ds41.verification import MAX_VERIFICATION_ROWS, verification_max_rows


# Hardware-specific EXL3 tuning: mode 2, default max-K=6.
# Excludes head (5120,129280,6) and K4 Engram (6144,25600,4): both
# failed M1/M2 bitwise parity. Never widen by a K range.
DENSE_M2_SAFE_SIGNATURES = frozenset({
    (5120, 1280, 6),
    (1280, 32768, 5),
    (5120, 512, 6),
    (8192, 5120, 5),
    (5120, 2304, 5),
    (2304, 5120, 5),
    (4096, 1024, 5),
    (5120, 1280, 5),
    (5120, 512, 5),
    (6144, 25600, 5),
    (1280, 4096, 5),
    (512, 128, 8),
    (5120, 2304, 4),
    (2304, 5120, 4),
})

# These include the two M2-unsafe projections: M32 is a NEW numerical
# target, not an extension of the old M1/M2 equivalence certificate.
DENSE_CANONICAL_TILE_SIGNATURES = DENSE_M2_SAFE_SIGNATURES | {
    (5120, 129280, 6),
    (6144, 25600, 4),
}


def _parse_dense_canonical_tile(value):
    if not isinstance(value, str) or value.strip() not in ("0", "32"):
        raise ValueError("DS41_DENSE_CANONICAL_TILE must be 0 or 32")
    return int(value.strip())


def _dense_m2_precision_config_supported():
    # EXL3 caches these environment options. Set them before constructing any
    # linear / loading the extension, and leave them fixed for the process.
    return (os.environ.get("EXL3_INT8_GEMV", "2") == "2"
            and os.environ.get("EXL3_INT8_GEMV_MAX_K", "0") in ("0", "6"))


@lru_cache(maxsize=None)
def _dense_m2_device_supported(device):
    if device.type != "cuda":
        return False
    props = torch.cuda.get_device_properties(device)
    # The SQ decomposition depends on SM count and occupancy. Evidence for
    # this device must not silently become a claim for another Blackwell GPU.
    return (props.name == "NVIDIA GeForce RTX 5090 D"
            and (props.major, props.minor, props.multi_processor_count) == (12, 0, 170))


class ELinear:
    def __init__(self, lin, *, allow_m2: bool | None = None):
        if allow_m2 is not None and not isinstance(allow_m2, bool):
            raise TypeError("allow_m2 must be bool or None")
        self.lin = lin
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.k = lin.K
        self.strict_small_rows = os.environ.get("DS41_STRICT_DENSE", "1") != "0"
        self.canonical_tile = _parse_dense_canonical_tile(os.environ.get("DS41_DENSE_CANONICAL_TILE", "0"))
        if self.canonical_tile and not self.strict_small_rows:
            raise ValueError("DS41_DENSE_CANONICAL_TILE=32 requires DS41_STRICT_DENSE=1")
        if self.canonical_tile and not 1 <= verification_max_rows() <= MAX_VERIFICATION_ROWS:
            raise ValueError(f"DS41_DENSE_CANONICAL_TILE=32 requires verification_max_rows in 1..{MAX_VERIFICATION_ROWS}")
        self.allow_m2 = os.environ.get("DS41_DENSE_M2", "0") == "1" if allow_m2 is None else allow_m2
        self._tile_metadata_supported = (
            (self.in_features, self.out_features, self.k) in DENSE_CANONICAL_TILE_SIGNATURES
            and getattr(lin, "quant_type", None) == "exl3"
            and getattr(lin, "mul1", False) is True
            and not getattr(lin, "mcg", False)
            and getattr(lin, "bias", None) is None
            and getattr(lin, "default_out_dtype", None) == torch.float16
        )
        self._m2_metadata_supported = (
            self.allow_m2
            and (self.in_features, self.out_features, self.k) in DENSE_M2_SAFE_SIGNATURES
            and _dense_m2_precision_config_supported()
            and getattr(lin, "mul1", False) is True
            and not getattr(lin, "mcg", False)
            and getattr(lin, "bias", None) is None
            and getattr(lin, "default_out_dtype", None) == torch.float16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).to(torch.float16).contiguous()
        rows, limit = flat.shape[0], verification_max_rows()
        if self.strict_small_rows and self.canonical_tile and 1 <= rows <= limit:
            if not self._tile_metadata_supported:
                raise RuntimeError("Experimental dense tile32 requires a known checkpoint EXL3 mul1 signature, "
                                   "no MCG/bias, and Half output")
            if flat.shape[-1] != self.in_features:
                raise ValueError("canonical dense tile input width differs from EXL3 metadata")
            outputs = []
            for start in range(0, rows, self.canonical_tile):
                part = flat[start:start + self.canonical_tile]
                if part.shape[0] < self.canonical_tile:
                    padded = flat.new_zeros((self.canonical_tile, self.in_features))
                    padded[:part.shape[0]].copy_(part)
                else:
                    padded = part
                # Each tile uses exactly native M32, including sequential
                # decode/head and an incomplete final tile. EXL3 run_alloc
                # owns fresh outputs, retained here until concatenation.
                outputs.append(self.lin.forward(padded, {})[:part.shape[0]])
            y = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        elif self.strict_small_rows and 1 < rows <= limit:
            # The upstream M<=2 INT8 / M>=3 FP16 dispatch changes activation
            # precision. Even its all-FP16 M1/M3 paths need not round equally.
            # Default to the single-token kernel. M2 is opt-in, restricted to
            # individually measured signatures/configuration/device; the odd
            # tail remains M1. The full target still needs its parity gate.
            chunk = 2 if self._m2_metadata_supported and _dense_m2_device_supported(flat.device) else 1
            y = torch.cat([self.lin.forward(flat[i:i + chunk], {})
                           for i in range(0, rows, chunk)], dim=0)
        else:
            y = self.lin.forward(flat, {})
        return y.reshape(*shape[:-1], self.out_features)


def build_linear(suh, svh, trellis, mul1, device: str, *, allow_m2: bool | None = None) -> ELinear:
    from exllamav3.modules.quant.exl3 import LinearEXL3

    suh = suh.to(device=device, dtype=torch.float16).contiguous()
    svh = svh.to(device=device, dtype=torch.float16).contiguous()
    trellis = trellis.to(device=device, dtype=torch.int16).contiguous()
    mul1 = mul1.to(device=device, dtype=torch.int32)
    k = trellis.shape[-1] / 16
    lin = LinearEXL3(
        config=None,
        in_features=suh.shape[0],
        out_features=svh.shape[0],
        suh=suh,
        svh=svh,
        trellis=trellis,
        mul1=mul1,
        key="ds41",
    )
    if abs(float(lin.K) - float(k)) > 1e-3:
        raise RuntimeError(f"LinearEXL3 K {lin.K} != tile K {k}")
    return ELinear(lin, allow_m2=allow_m2)


def grouped_gemm(x: torch.Tensor, linears: list[ELinear]) -> torch.Tensor:
    """Same input, several experts. x is [n, in]; return [e, n, out]."""
    from exllamav3.ext import exllamav3_ext as ext

    n = x.shape[0]
    inn = linears[0].in_features
    out = linears[0].out_features
    k = float(linears[0].k)
    xt = x.reshape(1, n, inn).contiguous()
    c = torch.empty(len(linears), n, out, dtype=torch.float16, device=x.device)
    had = torch.empty(len(linears), n, inn, dtype=torch.float16, device=x.device)
    dev = x.device
    ptrs_t = torch.tensor([L.lin.trellis.data_ptr() for L in linears], dtype=torch.long, device=dev)
    ptrs_su = torch.tensor([L.lin.suh.data_ptr() for L in linears], dtype=torch.long, device=dev)
    ptrs_sv = torch.tensor([L.lin.svh.data_ptr() for L in linears], dtype=torch.long, device=dev)
    ext.exl3_mgemm(
        xt, ptrs_t, c, ptrs_su, had, ptrs_sv,
        None, None, k, -1, False, True, -1, -1, 0, n, None, None,
    )
    return c
