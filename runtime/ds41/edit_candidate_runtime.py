"""Opt-in prompt-derived proposals with an explicitly separate lookup report.

No Torch, model, file history, expected answer or tokenizer loader is imported.
The caller supplies its normalized prompt and actual special-token-free encoder.
Every proposal still requires the original target verifier and commit path.
"""
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from ds41.edit_candidate_lookup import (MAX_TOTAL_TOKENS, PreparedEditCandidate,
                                        prepare_default_edit_candidate)
from ds41 import lookup_policy


FLAG = "DS41_EDIT_CANDIDATE"
PROTOCOL = "prompt_only_default_edit_with_history_fallback_v1"
SOURCE_NAMES = ("edit_candidate_runtime.py", "edit_candidate_lookup.py", "edit_proposal.py")


def requested(environ=None):
    value = (os.environ if environ is None else environ).get(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(FLAG+" must be 0 or 1")
    return value == "1"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sources():
    parent = Path(__file__).parent
    return {name: hashlib.sha256((parent/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def _finite_duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _same_decision(value, issued):
    return (type(value) is dict and set(value) == set(issued)
            and all(type(value[key]) is type(original) and value[key] == original
                    for key, original in issued.items()))


def _same_identity(value, actual):
    return (type(value) is dict and json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
            == json.dumps(actual,sort_keys=True,separators=(",",":"),allow_nan=False))


class EditCandidateSession:
    """One decode's prepared candidate and metadata-only selection ledger.

    Construct only under the explicit flag. Unsupported edit syntax is an
    ordinary prepared=None fallback; invalid identity/tokenizer output is an
    error. No per-cycle source hashing or tokenizer calls occur.
    """
    def __init__(self, prompt, encode, *, tokenizer_sha256, lookup_identity):
        _require(requested(), "Edit candidate sessions require the explicit enabled flag")
        _require(type(prompt) is str and callable(encode), "Need normalized prompt text and real encoder")
        _require(type(tokenizer_sha256) is str and re.fullmatch("[0-9a-f]{64}", tokenizer_sha256) is not None,
                 "Need actual tokenizer SHA256")
        _require(_same_identity(lookup_identity, lookup_policy.configuration()),
                 "Lookup identity differs from the actual startup policy")
        maximum = lookup_identity.get("max_draft_tokens")
        _require(type(maximum) is int and maximum in (127, 255, 511, 1023), "Need the explicit lookup proposal capacity")
        try:
            prompt_bytes = prompt.encode("utf8")
        except UnicodeError as error:
            raise ValueError("Normalized prompt must be valid UTF8") from error
        start = time.perf_counter()
        prepared = prepare_default_edit_candidate(prompt, encode, tokenizer_sha256=tokenizer_sha256)
        self._preparation_cpu_s = time.perf_counter()-start
        _require(prepared is None or type(prepared) is PreparedEditCandidate, "Invalid prepared candidate owner")
        prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
        _require(prepared is None or (prepared.prompt_sha256 == prompt_sha
                 and prepared.tokenizer_sha256 == tokenizer_sha256), "Candidate prompt/tokenizer identity differs")
        self._prepared = prepared
        self._lookup_identity = copy.deepcopy(lookup_identity)
        self._maximum = maximum
        self._issued = []
        self._failed = False
        self._identity = dict(protocol=PROTOCOL, requested=True, enabled=True,
            source_sha256=_sources(), lookup_identity=copy.deepcopy(lookup_identity),
            tokenizer_sha256=tokenizer_sha256,
            normalized_prompt_sha256=prompt_sha,
            normalized_prompt_bytes=len(prompt_bytes), supported=prepared is not None,
            candidate_metadata=None if prepared is None else prepared.metadata(), max_draft_tokens=maximum,
            selection_inputs="Current normalized prompt-derived candidates and already emitted assistant token IDs only",
            fallback="Original identity-bound history-only lookup policy on fallback cycles only",
            target_math_modified=False, target_commit_modified=False, target_verification_required=True,
            scheduling_GPU_verified=False, full_model_quality_verified=False, GPU_certificate=False,
            stochastic_sampling_certified=False)

    @property
    def prepared(self):
        return self._prepared

    def configuration(self):
        return copy.deepcopy(self._identity)

    def execution_identity(self):
        """Stable across requests; prompt/candidate identities stay in raw config."""
        names = ("protocol", "requested", "enabled", "source_sha256", "lookup_identity",
                 "tokenizer_sha256", "max_draft_tokens", "target_math_modified", "target_commit_modified",
                 "target_verification_required", "scheduling_GPU_verified", "full_model_quality_verified",
                 "GPU_certificate", "stochastic_sampling_certified")
        return {name: copy.deepcopy(self._identity[name]) for name in names}

    def verify_metadata(self):
        if self._failed:
            raise RuntimeError("Edit candidate session failed; discard it")
        try:
            if (not requested() or not _same_identity(lookup_policy.configuration(), self._lookup_identity)
                    or _sources() != self._identity["source_sha256"]
                    or (None if self._prepared is None else self._prepared.metadata())
                        != self._identity["candidate_metadata"]):
                raise RuntimeError("Edit candidate flag/source/tokenizer-candidate/lookup identity changed")
        except BaseException:
            self._failed = True
            raise
        return True

    def select(self, generated, capacity):
        if self._failed:
            raise RuntimeError("Edit candidate session failed; discard it")
        _require(type(capacity) is int and 0 <= capacity <= self._maximum,
                 "Candidate capacity exceeds the existing identity-bound lookup limit")
        _require(type(generated) in (list, tuple) and len(generated) <= MAX_TOTAL_TOKENS
                 and all(type(token) is int and 0 <= token < 2**32 for token in generated),
                 "Need bounded already emitted assistant uint32 token IDs")
        _require(len(self._issued) < MAX_TOTAL_TOKENS, "Too many candidate selection calls in one decode")
        start = time.perf_counter()
        chosen = None if self._prepared is None else self._prepared.select(generated, capacity)
        draft = None if chosen is None else list(chosen.tokens)
        reason = ("unsupported_prompt" if self._prepared is None else "zero_capacity" if not capacity
                  else "empty_history" if not generated else "no_common_alignment") if chosen is None else None
        meta = dict(protocol=PROTOCOL, chosen="fallback" if chosen is None else "candidate",
            drafted=0 if chosen is None else len(draft), requested_capacity=capacity,
            max_draft_tokens=self._maximum, known_generated_tokens=len(generated),
            alignment="none" if chosen is None else chosen.alignment,
            matched_tokens=0 if chosen is None else chosen.matched_tokens,
            matching_positions=0 if chosen is None else chosen.matching_positions,
            common_continuation_tokens=0 if chosen is None else chosen.common_continuation_tokens,
            selection_cpu_s=time.perf_counter()-start, fallback_reason=reason)
        self._issued.append(copy.deepcopy(meta))
        return draft, meta

    def decode_report(self, cycles):
        """Validate real selected/candidate/fallback scopes, never relabel one.

        accepted counts are verifier matches before EOS truncation. Emitted
        tokens may be a shorter prefix; target correctness is certified by the
        separate unchanged target gates, not these metadata assertions.
        """
        self.verify_metadata()  # Output boundary only, never select's hot path.
        _require(type(cycles) in (list, tuple) and len(cycles) == len(self._issued),
                 "Decode cycles do not match this session's actual selections")
        fallback_cycles = []
        totals = Counter()
        alignments, reasons = Counter(), Counter()
        selection_cpu_s = 0.
        for cycle, issued in zip(cycles, self._issued):
            _require(type(cycle) is dict and _same_decision(cycle.get("edit_candidate"), issued),
                     "Candidate cycle metadata differs from its actual selection")
            drafted, accepted, emitted = (cycle.get(name) for name in ("drafted", "accepted", "emitted"))
            _require(type(drafted) is int and 0 <= drafted <= issued["requested_capacity"] <= self._maximum
                     and type(accepted) is int and 0 <= accepted <= drafted
                     and type(emitted) is int and 1 <= emitted <= accepted+1,
                     "Invalid actual draft/accepted/emitted cycle geometry")
            _require(_finite_duration(issued["selection_cpu_s"]), "Invalid candidate CPU selection duration")
            if "selected" in cycle:
                _require(type(cycle["selected"]) is int and cycle["selected"] == drafted,
                         "Actual selected count differs from drafted count")
            selected = issued["chosen"] == "candidate"
            if selected:
                _require(self._prepared is not None and drafted == issued["drafted"] > 0
                         and issued["common_continuation_tokens"] == drafted
                         and issued["alignment"] in ("assistant_prefix", "assistant_suffix")
                         and issued["fallback_reason"] is None and "proposal_policy" not in cycle,
                         "Candidate cycle claims fallback policy or mismatched selection")
                alignments[issued["alignment"]] += 1
            else:
                _require(issued["chosen"] == "fallback" and issued["drafted"] == 0
                         and issued["alignment"] == "none" and issued["fallback_reason"] in
                             ("unsupported_prompt", "zero_capacity", "empty_history", "no_common_alignment"),
                         "Invalid fallback selection metadata")
                fallback_cycles.append(cycle)
                reasons[issued["fallback_reason"]] += 1
            kind = "candidate" if selected else "fallback"
            totals[kind+"_cycles"] += 1
            totals[kind+"_drafted"] += drafted
            totals[kind+"_accepted"] += accepted
            totals[kind+"_emitted"] += emitted
            totals["cycles"] += 1
            selection_cpu_s += issued["selection_cpu_s"]
        fallback = lookup_policy.decode_report(self._lookup_identity, fallback_cycles, "lookup")
        summary = {key: totals[key] for key in ("cycles", "candidate_cycles", "fallback_cycles",
            "candidate_drafted", "candidate_accepted", "candidate_emitted",
            "fallback_drafted", "fallback_accepted", "fallback_emitted")}
        summary.update(candidate_alignments=dict(alignments), fallback_reasons=dict(reasons),
            candidate_selection_cpu_s=selection_cpu_s, preparation_cpu_s=self._preparation_cpu_s)
        return dict(protocol=PROTOCOL, identity=self.configuration(), applicable=True, scheduling_enabled=True,
            metadata_rechecked_at_output=True, candidate_selector_executed=bool(cycles),
            fallback_lookup_report=fallback, decode_summary=summary,
            counter_scope="Actual candidate and fallback decisions; verifier acceptance before EOS truncation, no counterfactual speed claim",
            target_math_modified=False, target_commit_modified=False, GPU_certificate=False)
