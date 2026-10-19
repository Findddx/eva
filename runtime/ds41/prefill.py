"""Explicit prompt prefill chunking; no truncation and no whole-pass parity claim.

All chunks execute at their absolute token offsets. Attention/compressor/hash
state remains owned by the model. A failed chunk leaves a partially consumed
model; callers must abort, not retry generation from that state.
"""
from __future__ import annotations

import math


DEFAULT_MAX_HIDDEN_BYTES = 256 * 1024**2


def prefill_alignment(args):
    alignment = 1
    for ratio in getattr(args, "compress_ratios", ()):
        if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio < 0:
            raise ValueError("compress_ratios must contain nonnegative integers")
        if ratio:
            alignment = math.lcm(alignment, ratio)
    return alignment


def _estimated_hidden_bytes(model, batch, length):
    args = model.args
    targets, dim = getattr(args, "dspark_target_layer_ids", ()), getattr(args, "dim", None)
    if not targets or not isinstance(dim, int) or dim < 1:
        return None
    # This checkpoint records Half/BF16 target residuals. A wider embedding is
    # conservatively charged too; the first actual chunk is checked again.
    embed = getattr(model, "embed", None)
    element = max(2, embed.element_size()) if embed is not None else 2
    return batch * length * dim * len(targets) * element


def prefill(model, tokens, chunk_size=0, retain_full_hidden=False, *,
            max_hidden_bytes=DEFAULT_MAX_HIDDEN_BYTES, retain_tail_hidden=0):
    """Return last-prompt logits, preserving every supplied input token.

    ``chunk_size=0`` is the original whole call. Positive sizes must be a
    multiple of all configured positive compression ratios; the final tail is
    never padded or dropped. Chunking changes floating point call geometry and
    must be matched between sequential/speculative benchmark variants.

    DSpark's existing warm() requires full-prompt main_hidden. Explicit chunked
    retention uses one bounded preallocation before scratch reuse. Whole mode
    retains the model's original hidden allocation without this new byte cap.
    ``retain_tail_hidden=W`` instead copies the final min(N,W) target rows into
    independent bounded storage, with absolute interval metadata. It requires
    explicit chunking and is mutually exclusive with full retention. It does
    not change the target forward schedule or certify long-context inference.
    Metadata is attached as ``model.prefill_metadata`` only on success.
    """
    import torch

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 0:
        raise ValueError("prefill chunk_size must be a nonnegative integer (0 means whole)")
    if not isinstance(retain_full_hidden, bool):
        raise TypeError("retain_full_hidden must be bool")
    if type(retain_tail_hidden) is not int or retain_tail_hidden < 0:
        raise ValueError("retain_tail_hidden must be a nonnegative integer")
    if retain_tail_hidden and (retain_full_hidden or not chunk_size):
        raise ValueError("tail retention requires chunk_size > 0 and excludes full retention")
    if isinstance(max_hidden_bytes, bool) or not isinstance(max_hidden_bytes, int) or max_hidden_bytes < 1:
        raise ValueError("max_hidden_bytes must be a positive byte limit")
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or min(tokens.shape) < 1:
        raise ValueError("prefill requires a nonempty input token tensor [batch,tokens]")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("prefill token ids must have integer embedding dtype")
    batch, length = tokens.shape
    if int(getattr(model, "position", 0)) != 0:
        raise ValueError("prefill requires a fresh model at position 0")
    if length > model.args.max_seq_len:
        raise ValueError("complete prompt exceeds the reserved context capacity")
    if batch > getattr(model.args, "max_batch_size", batch):
        raise ValueError("prefill batch exceeds the reserved batch capacity")
    alignment = prefill_alignment(model.args)
    if chunk_size and chunk_size % alignment:
        raise ValueError(f"prefill chunk_size must be a multiple of compression alignment {alignment}")
    tail_rows = min(length, retain_tail_hidden)
    retained_rows = tail_rows if retain_tail_hidden else length
    hidden_start = length - tail_rows if retain_tail_hidden else 0
    limit_hidden = bool(chunk_size and (retain_full_hidden or retain_tail_hidden))
    estimated = _estimated_hidden_bytes(model, batch, retained_rows) if limit_hidden else None
    if estimated is not None and estimated > max_hidden_bytes:
        raise ValueError(f"retained DSpark hidden estimate {estimated} exceeds limit {max_hidden_bytes}")
    size = chunk_size or length
    spans = [(offset, min(size, length-offset)) for offset in range(0, length, size)]
    full_hidden = None
    logits = None
    hidden_bytes = 0
    with torch.inference_mode():
        for offset, count in spans:
            logits = None  # Release the prior chunk's head output before forwarding.
            logits = model.forward(tokens[:, offset:offset+count].contiguous(), offset, full_logits=False)
            if int(model.position) != offset+count:
                raise RuntimeError("model did not consume the complete prefill chunk at its absolute offset")
            if retain_full_hidden or retain_tail_hidden:
                hidden = getattr(model, "main_hidden", None)
                if hidden is None or hidden.ndim != 3 or hidden.shape[:2] != (batch, count):
                    raise RuntimeError("DSpark prefill requires target main_hidden for every input row")
                required = batch * retained_rows * hidden.shape[-1] * hidden.element_size()
                if limit_hidden and required > max_hidden_bytes:
                    raise ValueError(f"retained DSpark hidden {required} exceeds limit {max_hidden_bytes}")
                hidden_bytes = required
                if len(spans) == 1 and not retain_tail_hidden:
                    full_hidden = hidden  # Preserve the original whole-pass ownership.
                else:
                    if full_hidden is None:
                        full_hidden = hidden.new_empty((batch, retained_rows, hidden.shape[-1]))
                    if (hidden.shape[-1] != full_hidden.shape[-1] or hidden.dtype != full_hidden.dtype
                            or hidden.device != full_hidden.device):
                        raise RuntimeError("prefill main_hidden width, dtype or device changed between chunks")
                    lo, hi = max(offset, hidden_start), offset + count
                    if lo < hi:
                        full_hidden[:, lo-hidden_start:hi-hidden_start].copy_(hidden[:, lo-offset:hi-offset])
        if retain_full_hidden or retain_tail_hidden:
            model.main_hidden = full_hidden
    hidden = getattr(model, "main_hidden", None)
    has_hidden = (isinstance(hidden, torch.Tensor) and hidden.ndim == 3)
    actual_rows = int(hidden.shape[1]) if has_hidden else 0
    actual_bytes = hidden.numel() * hidden.element_size() if has_hidden else 0
    model.prefill_metadata = {
        "chunk_size": chunk_size, "chunked": len(spans) > 1,
        "compression_alignment": alignment, "prompt_tokens": length,
        "processed_tokens": sum(count for _, count in spans), "chunks": len(spans),
        "offsets_and_lengths": [[offset, count] for offset, count in spans],
        "full_logits": False, "retain_full_hidden": retain_full_hidden,
        "retained_full_hidden_bytes": hidden_bytes if retain_full_hidden else 0,
        "retain_tail_hidden": retain_tail_hidden,
        "retention_mode": "tail" if retain_tail_hidden else "full" if retain_full_hidden else "last_chunk",
        "hidden_start_pos": length-actual_rows if has_hidden else None,
        "hidden_end_pos": length if has_hidden else None,
        "retained_rows": actual_rows, "retained_hidden_bytes": actual_bytes,
        "hidden_dtype": str(hidden.dtype) if has_hidden else None,
        "hidden_width": int(hidden.shape[-1]) if has_hidden else None,
        "max_hidden_bytes": max_hidden_bytes,
        "hidden_limit_applied": limit_hidden,
        "whole_vs_chunk_bitwise_validated": False,
        "numerical_scope": "Match prefill configuration across target comparisons; chunk geometry may change rounding.",
    }
    return logits
