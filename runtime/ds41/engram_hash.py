"""Official Engram hash layout, kept as the reference for ds41.engram.

Copied from deepseek-ai/DeepSeek-V4.1-Flash inference/engram.py.
"""
from dataclasses import dataclass

import numpy as np
import torch


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(layer_ids, max_ngram_size, tokenizer_vocab_size) -> torch.Tensor:
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args):
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = args.engram_max_ngram_size, args.engram_n_heads
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(args.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=args.engram_head_dim,
        )


class NgramHashState(torch.nn.Module):
    DEAD = -1

    def __init__(self, args, layout: EngramLayout, tokenizer):
        super().__init__()
        self.layout = layout
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != args.engram_compressed_vocab_size:
            raise ValueError(f"compressed vocab {vocab_size} != {args.engram_compressed_vocab_size}")
        self.pad_id = token_map[args.engram_pad_id]
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size)
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets)), persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
        self.register_buffer(
            "cache", torch.zeros(args.max_batch_size, args.max_seq_len, dtype=torch.int64), persistent=False
        )

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int, token_mask: torch.Tensor | None = None):
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)
        self.cache[:batch, start_pos : start_pos + seqlen] = compressed
        positions = torch.arange(start_pos, start_pos + seqlen, device=input_ids.device).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source = self.cache[:batch].gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)
        products = tokens.unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets
