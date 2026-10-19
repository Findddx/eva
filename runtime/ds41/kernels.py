"""PyTorch stand-ins for the pieces the official graph runs in tilelang.

Compressed KV is stored in the paper's FP4 layout: E2M1 nibbles plus one
E4M3 scale per 16 channels. Attention dequantizes that buffer when it reads.
"""
import math
import os

import torch
import torch.nn.functional as F

FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_LEVELS = {}
_EAGER_QUANT = os.environ.get("DS41_EAGER_QUANT", "0") == "1"


def _fp4_levels(device):
    key = (device.type, device.index)
    table = _FP4_LEVELS.get(key)
    if table is None:
        table = torch.tensor(FP4_LEVELS, device=device)
        _FP4_LEVELS[key] = table
    return table


def _power2_scale_ceil(x):
    # Mirror the reference kernel's integer exponent calculation. log2 followed
    # by ceil can round DOWN for floats just above an exact power of two.
    bits = x.contiguous().view(torch.int32)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (exponent << 23).view(torch.float32)


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, device=device, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen, device=device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    y = x
    z = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if z.ndim == 3:
        freqs_cis = freqs_cis.view(1, z.size(1), z.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, z.size(1), 1, z.size(-1))
    z = torch.view_as_real(z * freqs_cis).flatten(-2)
    y.copy_(z.to(y.dtype))
    return y


_HC_WARNED = False


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps):
    global _HC_WARNED
    try:
        from ds41.hc_cuda import hc_split_sinkhorn_cuda

        fast = hc_split_sinkhorn_cuda(mixes, hc_scale, hc_base, hc_mult, iters, eps)
        if fast is not None:
            return fast
    except Exception as exc:
        if not _HC_WARNED:
            print(f"hc cuda fallback: {type(exc).__name__}: {exc}", flush=True)
            _HC_WARNED = True
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = mixes[..., 2 * hc :].reshape(*mixes.shape[:-1], hc, hc)
    comb = comb * hc_scale[2] + hc_base[2 * hc :].reshape(hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def fake_quant_fp8(x: torch.Tensor, group: int = 32) -> torch.Tensor:
    """Training-compatible MXFP8 activation roundtrip, with per-group scales."""
    if x.shape[-1] % group:
        raise ValueError("FP8 activation width must be divisible by its group size")
    if not _EAGER_QUANT:
        from ds41.quant_cuda import supported, fp8_roundtrip_
        if supported(x, group):
            return fp8_roundtrip_(x, group)
    grouped = x.float().reshape(*x.shape[:-1], -1, group)
    amax = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = _power2_scale_ceil(amax / 448.0)
    q = (grouped / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    x.copy_(q.reshape_as(x).to(x.dtype))
    return x


def quant_fp4_group(x: torch.Tensor, group: int = 16, scale_fmt: str = "e4m3"):
    """Pack E2M1 nibbles with E4M3 KV or E8M0 indexer group scales.

    Returns codes uint8 [..., d/2], scales float8 [..., d/group], and the
    dequantized tensor in the input dtype (what attention multiplies).
    """
    dtype = x.dtype
    d = x.shape[-1]
    assert d % group == 0 and group % 2 == 0
    if scale_fmt not in ("e4m3", "ue8m0", "e8m0"):
        raise ValueError(f"unknown FP4 scale format: {scale_fmt}")
    if not _EAGER_QUANT and (scale_fmt == "e4m3" or hasattr(torch, "float8_e8m0fnu")):
        from ds41.quant_cuda import supported, fp4_quant
        if supported(x, group):
            return fp4_quant(x, group, scale_fmt != "e4m3")
    flat = x.float().reshape(-1, d // group, group)
    if scale_fmt == "e4m3":
        amax = flat.abs().amax(-1).clamp_min(6 * 2**-9)
        scale = (amax / 6).to(torch.float8_e4m3fn).float()
        scale_dtype = torch.float8_e4m3fn
    elif scale_fmt in ("ue8m0", "e8m0"):
        amax = flat.abs().amax(-1).clamp_min(6 * 2**-126)
        scale = _power2_scale_ceil(amax / 6)
        scale_dtype = getattr(torch, "float8_e8m0fnu", torch.float32)
    else:
        raise ValueError(f"unknown FP4 scale format: {scale_fmt}")
    norm = flat / scale.unsqueeze(-1)
    sign = torch.where(norm < 0, torch.ones_like(norm), torch.zeros_like(norm))
    levels = _fp4_levels(x.device)
    distance = (norm.abs().unsqueeze(-1) - levels).abs()
    idx = distance.argmin(-1)
    # E2M1 uses round-to-nearest, ties-to-even. argmin alone always selects
    # the lower code, which is wrong at .75, 1.75 and 3.5.
    next_idx = (idx + 1).clamp_max(7)
    ties = distance.gather(-1, idx.unsqueeze(-1)) == distance.gather(-1, next_idx.unsqueeze(-1))
    idx = (idx + ((idx < 7) & (idx & 1).bool() & ties.squeeze(-1)).long()).to(torch.uint8)
    nib = (sign.to(torch.uint8) << 3) | idx
    nib = nib.reshape(-1, d)
    codes = (nib[:, 0::2] << 4) | nib[:, 1::2]
    codes = codes.reshape(*x.shape[:-1], d // 2)
    scales = scale.reshape(*x.shape[:-1], d // group).to(scale_dtype)
    recon = (torch.where(sign > 0, -levels[idx.long()], levels[idx.long()]) * scale.unsqueeze(-1)).reshape(x.shape)
    return codes, scales, recon.to(dtype)


def dequant_fp4_group(codes: torch.Tensor, scales: torch.Tensor, group: int = 16,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if codes.dtype != torch.uint8 or codes.shape[-1] * 2 % group or scales.numel() != codes.numel() * 2 // group:
        raise ValueError("FP4 codes/scales do not match the group layout")
    if not _EAGER_QUANT and codes.is_cuda and codes.numel() and codes.is_contiguous() and scales.is_contiguous():
        allowed = (torch.float32, torch.float8_e4m3fn, getattr(torch, "float8_e8m0fnu", torch.float32))
        if scales.device == codes.device and scales.dtype in allowed and dtype in (torch.float32, torch.float16, torch.bfloat16):
            from ds41.quant_cuda import fp4_dequant
            return fp4_dequant(codes, scales, group, dtype)
    d = codes.shape[-1] * 2
    hi = codes >> 4
    lo = codes & 0x0F
    nib = torch.stack((hi, lo), dim=-1).reshape(*codes.shape[:-1], d)
    sign = nib >> 3
    idx = (nib & 0x07).long()
    levels = _fp4_levels(codes.device)
    mag = levels[idx]
    signed = torch.where(sign > 0, -mag, mag)
    scale = scales.float()
    return (signed.reshape(*codes.shape[:-1], d // group, group) * scale.unsqueeze(-1)).reshape(*codes.shape[:-1], d).to(dtype)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q [b,s,h,d], kv [b,n,d], idxs [b,s,topk]. Sink adds to the denominator only."""
    b, s, h, d = q.shape
    topk = topk_idxs.shape[-1]
    if topk == 0:
        return q.new_zeros(b, s, h, d)
    n = kv.shape[1]
    if n == 0:
        return q.new_zeros(b, s, h, d)
    # Compressed indices are absolute into the concatenated cache. Anything that
    # does not land in this step's kv is an empty slot, same as -1.
    valid = (topk_idxs >= 0) & (topk_idxs < n)
    safe = topk_idxs.clamp(0, max(n - 1, 0))
    gathered = torch.gather(kv, 1, safe.reshape(b, s * topk, 1).expand(b, s * topk, d))
    gathered = gathered.reshape(b, s, topk, d)
    scores = torch.einsum("bshd,bstd->bsht", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
    m = torch.maximum(scores.amax(dim=-1), attn_sink.view(1, 1, h).float())
    exp = torch.exp(scores - m.unsqueeze(-1))
    denom = exp.sum(-1) + torch.exp(attn_sink.view(1, 1, h).float() - m)
    out = torch.einsum("bsht,bstd->bshd", exp / denom.unsqueeze(-1), gathered.float())
    return out.to(q.dtype)


def window_indices(window_size, bsz, seqlen, start_pos, device):
    if start_pos == 0:
        end = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size), device=device)
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat(
            [torch.arange(oldest, window_size, device=device), torch.arange(oldest, device=device)]
        )
        idxs = torch.where(idxs > start_pos, -1, idxs)
        return idxs.int().view(1, 1, -1).expand(bsz, 1, -1).contiguous()
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def swiglu(gate, up, limit: float):
    if limit > 0:
        up = up.clamp(min=-limit, max=limit)
        gate = gate.clamp(max=limit)
    return F.silu(gate) * up
