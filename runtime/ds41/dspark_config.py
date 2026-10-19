"""Explicit proposal-only block configuration. Stdlib only; never mutates target."""
from dataclasses import dataclass, is_dataclass, replace


MIN_OVERRIDE, MAX_OVERRIDE, LEGACY_MAX_DEPTH = 5, 32, 5


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def _override(value):
    if value is not None and (type(value) is not int or not MIN_OVERRIDE <= value <= MAX_OVERRIDE):
        raise ValueError("DSpark block-size override must be integer5..32 or None")
    return value


@dataclass(frozen=True)
class ProposalConfiguration:
    trained_block_size: int
    effective_block_size: int
    block_size_override: int | None = None

    def __post_init__(self):
        _positive(self.trained_block_size, "checkpoint DSpark block size")
        _positive(self.effective_block_size, "effective DSpark block size")
        _override(self.block_size_override)
        expected = self.trained_block_size if self.block_size_override is None else self.block_size_override
        if self.effective_block_size != expected:
            raise ValueError("Inconsistent DSpark proposal configuration receipt")

    def as_dict(self):
        return dict(trained_block_size=self.trained_block_size,
                    effective_block_size=self.effective_block_size,
                    block_size_override=self.block_size_override,
                    proposal_variant=("checkpoint_block_v1" if self.block_size_override is None
                                      else "explicit_block_override_v1"),
                    checkpoint_length_extrapolated=self.effective_block_size != self.trained_block_size,
                    trained_size_scope="Checkpoint-declared config, not an independent training-data audit",
                    first_five_invariance_claimed=False)


def configure_draft(target_args, block_size_override=None):
    """Return (draft_args, frozen metadata); unset keeps the exact old object."""
    _override(block_size_override)
    trained = _positive(target_args.dspark_block_size, "checkpoint DSpark block size")
    if block_size_override is None:
        return target_args, ProposalConfiguration(trained, trained)
    if not is_dataclass(target_args) or isinstance(target_args, type):
        raise ValueError("Explicit DSpark override requires a dataclass config clone")
    return (replace(target_args, dspark_block_size=block_size_override),
            ProposalConfiguration(trained, block_size_override, block_size_override))


def validate_depth(depth, effective_block_size, target_cap, block_size_override=None):
    _override(block_size_override)
    _positive(effective_block_size, "effective DSpark block size")
    _positive(target_cap, "target verification capacity")
    limit = min(LEGACY_MAX_DEPTH, effective_block_size) if block_size_override is None else effective_block_size
    if type(depth) is not int or not 0 <= depth <= limit:
        raise ValueError("DSpark depth must fit the explicit proposal block; unset override retains0..5")
    if depth + 1 > target_cap:
        raise ValueError("DSpark pending+depth exceeds target verification capacity")


def validate_decode_configuration(draft, depth, target_cap, block_size_override=None):
    """Old direct callers remain legacy; extended depth requires explicit receipt."""
    _override(block_size_override)
    args = getattr(draft, "args", None)
    effective = getattr(args, "dspark_block_size", LEGACY_MAX_DEPTH)
    metadata = getattr(draft, "proposal_configuration", None)
    if metadata is not None and not isinstance(metadata, ProposalConfiguration):
        raise ValueError("Invalid DSpark proposal configuration receipt")
    if block_size_override is not None:
        if (metadata is None or metadata.block_size_override != block_size_override
                or metadata.effective_block_size != block_size_override or effective != block_size_override):
            raise ValueError("DSpark override requires its explicit matching draft configuration receipt")
        cache = getattr(draft, "expert_cache", None)
        if (getattr(cache, "args", None) is not args or getattr(cache, "mode", None) not in ("ram", "resident")
                or getattr(args, "dspark_n_routed_experts", None) != 128
                or getattr(args, "dspark_n_activated_experts", None) != 3
                or getattr(args, "n_mtp_layers", None) != 3
                or type(getattr(cache, "gpu_slots", None)) is not int or cache.gpu_slots < 3 * effective):
            raise ValueError("DSpark override needs its independent128/top3/three-stage cache with enough slots")
    elif metadata is not None and metadata.block_size_override is not None:
        raise ValueError("Explicit DSpark block configuration requires the same decode override keyword")
    validate_depth(depth, effective, target_cap, block_size_override)
    if metadata is not None:
        if metadata.effective_block_size != effective or metadata.trained_block_size < 1:
            raise ValueError("DSpark config changed after configuration receipt")
        return metadata
    return ProposalConfiguration(effective, effective)


def validate_override_anchor(draft, target, pending_pos, effective_block_size):
    """Metadata-only absolute interval checks; no frequency/KV readback or copy."""
    import torch
    args = draft.args
    if (type(pending_pos) is not int or pending_pos < 2 or pending_pos != target.position
            or args.dspark_block_size != effective_block_size):
        raise ValueError("DSpark override anchor/config changed")
    target_limit = getattr(getattr(target, "args", None), "max_seq_len", None)
    if target_limit != args.max_seq_len or not 0 < pending_pos < args.max_seq_len:
        raise ValueError("DSpark override target/draft absolute context capacities differ")
    trained = getattr(getattr(target, "args", None), "dspark_block_size", None)
    if trained is not None and trained != draft.proposal_configuration.trained_block_size:
        raise ValueError("DSpark override changed the main checkpoint block configuration")
    blocks = getattr(draft, "blocks", ())
    if len(blocks) != 3:
        raise ValueError("DSpark override requires three real draft stages")
    for block in blocks:
        freqs = block.attn.freqs
        if (len(freqs.shape) != 2 or freqs.shape[1] != args.rope_head_dim // 2
                or freqs.shape[0] < pending_pos + effective_block_size
                or freqs.dtype != torch.complex64 or not freqs.is_contiguous() or freqs.storage_offset() != 0
                or freqs.device != target.main_hidden.device):
            raise ValueError("DSpark full noisy-future block exceeds its absolute RoPE table")


def validate_override_output(ids, logits, confidence, seed, args):
    """Validate full forward geometry, without an extra logits D2H/reduction."""
    import torch
    length = args.dspark_block_size
    if (not all(isinstance(value, torch.Tensor) for value in (ids, logits, confidence))
            or tuple(ids.shape) != (1, length + 1) or ids.dtype != torch.long
            or tuple(logits.shape) != (1, length, args.vocab_size) or logits.dtype != torch.float32
            or tuple(confidence.shape) != (1, length) or confidence.dtype != torch.float32
            or any(value.device != seed.device for value in (ids, logits, confidence))):
        raise ValueError("DSpark override must return full seed+L IDs, Float[1,L,V] logits and Float[1,L] raw confidence")
