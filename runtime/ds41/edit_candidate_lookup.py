"""Prompt-only edit proposals aligned to already emitted assistant tokens.

Proposals are matched against the assistant prefix before target verification.
The caller supplies the real tokenizer and must verify every proposed token
using the unchanged causal target, including rejection and state rollback.
"""
from dataclasses import dataclass
import hashlib
import re
from typing import Callable, Sequence

from ds41.edit_proposal import propose_default_edit

MAX_PROMPT_BYTES = 1024 * 1024 + 8192
MAX_VARIANT_TOKENS = 32768
MAX_TOTAL_TOKENS = 131072
MAX_DRAFT_TOKENS = 1023  # Proposal bound only; does not extend engine capacity.
MAX_MATCHES = 256
PROTOCOL = "prompt_only_literal_default_candidate_v1"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _ids(values):
    if not isinstance(values, (list, tuple)) or any(type(v) is not int or not 0 <= v < 2**32 for v in values):
        raise ValueError("Require a bounded sequence of uint32 token IDs")
    return tuple(values)


@dataclass(frozen=True)
class CandidateSelection:
    tokens: tuple[int, ...]
    alignment: str
    matched_tokens: int
    matching_positions: int
    common_continuation_tokens: int
    candidate_only: bool = True


@dataclass(frozen=True)
class PreparedEditCandidate:
    variants: tuple[tuple[int, ...], ...]
    tokenizer_sha256: str
    prompt_sha256: str
    instruction_sha256: str
    source_sha256: str
    edited_source_sha256: str

    def __post_init__(self):
        if (type(self.variants) is not tuple or not 1 <= len(self.variants) <= 4
                or any(type(v) is not tuple or not 1 <= len(v) <= MAX_VARIANT_TOKENS for v in self.variants)
                or sum(map(len, self.variants)) > MAX_TOTAL_TOKENS
                or len(set(self.variants)) != len(self.variants)):
            raise ValueError("Require distinct, immutable and bounded candidate variants")
        for v in self.variants:
            _ids(v)
        for value in (self.tokenizer_sha256, self.prompt_sha256, self.instruction_sha256,
                      self.source_sha256, self.edited_source_sha256):
            if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
                raise ValueError("Candidate identity requires SHA256 fields")

    def metadata(self):
        return dict(protocol=PROTOCOL, tokenizer_sha256=self.tokenizer_sha256,
            prompt_sha256=self.prompt_sha256, instruction_sha256=self.instruction_sha256,
            source_sha256=self.source_sha256, edited_source_sha256=self.edited_source_sha256,
            variant_lengths=[len(v) for v in self.variants],
            variant_token_sha256=[_sha(b''.join(t.to_bytes(4, 'little') for t in v)) for v in self.variants],
            candidate_only=True, target_verification_required=True, production_integrated=False,
            selection_inputs="Prepared prompt-derived candidates and already emitted assistant IDs only")

    def select(self, generated: Sequence[int], max_tokens: int):
        """Return a common candidate continuation; never inspect target answers.

        Whole emitted-prefix alignment takes priority. If the target supplied
        introductory text, a suffix of at least three emitted tokens may align
        to a source span. Ambiguous continuations end at their common prefix.
        """
        if type(max_tokens) is not int or not 0 <= max_tokens <= MAX_DRAFT_TOKENS:
            raise ValueError("Proposal length must be within the caller's bounded capacity")
        if len(generated) > MAX_TOTAL_TOKENS:
            raise ValueError("Emitted-history proposal input exceeds the independent bound")
        history = _ids(generated)
        if not max_tokens or not history:
            return None
        matches = [(v, len(history)) for v in self.variants
                   if len(history) < len(v) and v[:len(history)] == history]
        alignment, matched = 'assistant_prefix', len(history)
        if not matches:
            alignment = 'assistant_suffix'
            for n in range(min(16, len(history)), 2, -1):
                suffix = history[-n:]
                matches = []
                for v in self.variants:
                    for end in range(n, len(v)):
                        if v[end-n:end] == suffix:
                            matches.append((v, end))
                            if len(matches) > MAX_MATCHES:
                                return None
                if matches:
                    matched = n
                    break
        if not matches:
            return None
        common = list(matches[0][0][matches[0][1]:matches[0][1]+max_tokens])
        for v, end in matches[1:]:
            other = v[end:end+max_tokens]
            limit = min(len(common), len(other))
            i = 0
            while i < limit and common[i] == other[i]:
                i += 1
            del common[i:]
            if not common:
                return None
        return CandidateSelection(tuple(common), alignment, matched, len(matches), len(common))


def _prompt_parts(prompt):
    """One instruction paragraph followed by one complete Python module."""
    if type(prompt) is not str or len(prompt) > MAX_PROMPT_BYTES:
        return None
    try:
        if len(prompt.encode('utf8')) > MAX_PROMPT_BYTES:
            return None
    except UnicodeError:
        return None
    separator = re.search(r'(?:\r\n|\n)[ \t]*(?:\r\n|\n)', prompt)
    if separator is None:
        return None
    instruction, source = prompt[:separator.start()], prompt[separator.end():]
    if source.startswith('```'):
        opening = re.match(r'```(?:python|py)?[ \t]*\r?\n', source)
        closing = re.search(r'\r?\n```[ \t]*(?:\r?\n)?\Z', source)
        if opening is None or closing is None or opening.end() > closing.start():
            return None
        newline_bytes = 2 if source[closing.start():].startswith('\r\n') else 1
        source = source[opening.end():closing.start()+newline_bytes]
    return instruction, source


def prepare_default_edit_candidate(prompt: str, encode: Callable[[str], Sequence[int]], *, tokenizer_sha256: str):
    """Bound source parsing and tokenization; unsupported prompts return None.

    `encode` must be the caller's actual tokenizer with special insertion
    disabled. The identity is explicit; this module does not load a tokenizer,
    files, quality assertions, historical completions or a model.
    """
    if type(tokenizer_sha256) is not str or re.fullmatch('[0-9a-f]{64}', tokenizer_sha256) is None:
        raise ValueError("Require an explicit tokenizer SHA256")
    parts = _prompt_parts(prompt)
    if parts is None:
        return None
    instruction, source = parts
    proposal = propose_default_edit(instruction, source.encode('utf8'))
    if proposal is None:
        return None
    body = proposal.proposed_source.decode('utf8')
    # Formatting choices are hypotheses. Actual emitted tokens decide which
    # prefix/suffix aligns, and target logits decide every accepted token.
    content = body if body.endswith(('\n', '\r')) else body + '\n'
    texts = [body] + ['```' + language + '\n' + content + '```\n' for language in ('python', 'py', '')]
    variants, total = [], 0
    for text in texts:
        values = _ids(encode(text))
        if not values or len(values) > MAX_VARIANT_TOKENS:
            return None
        if values not in variants:
            variants.append(values)
            total += len(values)
        if total > MAX_TOTAL_TOKENS:
            return None
    return PreparedEditCandidate(tuple(variants), tokenizer_sha256, _sha(prompt.encode('utf8')),
        proposal.instruction_sha256, proposal.source_sha256, proposal.proposal_sha256)
