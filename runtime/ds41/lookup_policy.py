"""Default-off, history-only lookup scheduling; no Torch/CUDA/import effects.

Only proposals change. Every emitted token still requires the original target
verification/commit. No current acceptance labels or model future are inputs.
"""
import ast
from collections import Counter
import copy
import hashlib
import math
import os
from pathlib import Path
import time

from ds41.verification import verification_max_rows


FLAG = "DS41_LOOKUP_POLICY"
MODES = ("off", "ambiguity", "source-boundary", "both")


def max_draft_tokens():
    """Only explicit process cap256/512 extends this proposal-only bound.

    An environ argument to configuration selects the policy mode, not a second
    target capacity. Set DS41_VERIFY_MAX_ROWS before importing the engine.
    """
    cap = verification_max_rows()
    return cap - 1 if cap in (256, 512, 1024) else 127


def configuration(environ=None):
    environ = os.environ if environ is None else environ
    mode = environ.get(FLAG, "off")
    if mode not in MODES:
        raise ValueError(FLAG + " must be off, ambiguity, source-boundary or both")
    source = Path(__file__).with_name("speculate.py").read_bytes()
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name == "lookup_draft"]
    if len(nodes) != 1:
        raise RuntimeError("Original lookup_draft source identity is unavailable")
    return dict(mode=mode, protocol="history_only_lookup_policy_v1",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        legacy_lookup_ast_sha256=hashlib.sha256(ast.dump(nodes[0], include_attributes=False).encode()).hexdigest(),
        selection_inputs="Already-known prompt/generated token IDs and source boundary only; no acceptance outcome",
        min_ngram=3, max_ngram=16, max_draft_tokens=max_draft_tokens(),
        target_math_modified=False, stochastic_sampling_certified=False,
        scheduling_GPU_verified=False, full_model_quality_verified=False)


def validate_for_speculation(speculation, environ=None):
    identity = configuration(environ)
    if identity["mode"] != "off" and speculation != "lookup":
        raise ValueError(FLAG + " applies only to --speculation lookup")
    return identity


def decode_report(identity, cycles, speculation):
    """Bind actual cycle diagnostics to the fixed startup identity, no TPS math."""
    enabled = identity["mode"] != "off"
    maximum = identity.get("max_draft_tokens")
    if identity["mode"] not in MODES or type(maximum) is not int or maximum != max_draft_tokens():
        raise RuntimeError("Lookup policy identity has the wrong process proposal capacity")
    if enabled and speculation != "lookup":
        raise ValueError("Active lookup policy attached to a different decoder")
    if any(type(cycle.get("drafted")) is not int or not 0 <= cycle["drafted"] <= maximum for cycle in cycles):
        raise RuntimeError("Actual lookup proposal exceeds the identity-bound capacity")
    metadata = [cycle.get("proposal_policy") for cycle in cycles]
    if not enabled:
        if any(meta is not None for meta in metadata):
            raise RuntimeError("Off policy result unexpectedly contains active selector decisions")
        return dict(identity=copy.deepcopy(identity), applicable=speculation == "lookup",
            scheduling_enabled=False, selector_executed=False, proposal_changed=False,
            metadata_rechecked_at_output=True, decode_summary=None,
            counter_scope="No new selector diagnostics under off; original lookup CPU search is not timed here")
    chosen = Counter();scouts = caps = suppressed = 0;cpu_s = 0.
    for cycle, meta in zip(cycles, metadata):
        if (not isinstance(meta, dict) or meta.get("mode") != identity["mode"]
                or meta.get("selected_draft_tokens") != cycle["drafted"]
                or meta.get("chosen_verifier_rows") != cycle["drafted"]+1
                or type(meta.get("original_draft_tokens")) is not int
                or not cycle["drafted"] <= meta["original_draft_tokens"] <= maximum
                or meta.get("max_draft_tokens",127) != maximum):
            raise RuntimeError("Lookup cycle policy/geometry is absent or differs from startup identity")
        value = meta.get("selection_cpu_s")
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
            raise RuntimeError("Invalid lookup CPU selection duration")
        cpu_s += value;chosen[str(meta["chosen_verifier_rows"])] += 1
        scouts += int(meta["ambiguity_scout"]);caps += int(meta["source_boundary_capped"])
        suppressed += meta["original_draft_tokens"]-meta["selected_draft_tokens"]
    return dict(identity=copy.deepcopy(identity), applicable=True, scheduling_enabled=True,
        selector_executed=bool(cycles), proposal_changed=suppressed > 0,
        metadata_rechecked_at_output=True, decode_summary=dict(cycles=len(cycles),
            ambiguity_scouts=scouts, source_boundary_caps=caps,
            proposal_tokens_suppressed_at_visited_histories=suppressed,
            chosen_verifier_rows=dict(chosen), policy_selection_cpu_s=cpu_s),
        counter_scope="Actual decode decisions after checks. Suppressed proposals describe visited histories, not counterfactual GPU row/time savings")


class LookupProposalPolicy:
    """Propose solely from immutable-at-call known history; no target state.

    `ambiguity` suppresses a draft if longest-ngram matches disagree on their
    next ID. Existing decode then performs an ordinary M1 and replans next loop.
    `source-boundary` caps the original winner at the prompt/generated boundary.
    `both` gives the ambiguity M1 precedence. Off callers retain lookup_draft.
    """
    def __init__(self, mode="off"):
        if mode not in MODES:
            raise ValueError("Unknown lookup proposal policy")
        self.mode = mode
        self.max_draft_tokens = max_draft_tokens()
        self.last = None

    @classmethod
    def from_environment(cls, environ=None):
        return cls(configuration(environ)["mode"])

    @property
    def enabled(self):
        return self.mode != "off"

    def select(self, history, max_tokens, *, prompt_tokens):
        if not self.enabled:
            raise RuntimeError("Off policy must retain the original lookup_draft call")
        if self.max_draft_tokens != max_draft_tokens():
            raise RuntimeError("Lookup policy process capacity changed after construction")
        start = time.perf_counter()
        if (type(history) is not list or type(prompt_tokens) is not int
                or not 0 <= prompt_tokens <= len(history)
                or type(max_tokens) is not int or not 0 <= max_tokens <= self.max_draft_tokens
                or any(type(token) is not int or not 0 <= token < 2**32 for token in history)):
            raise ValueError("Require known Python token IDs, a valid prompt boundary and capacity0.."
                             + str(self.max_draft_tokens))
        self.last = None
        meta = dict(mode=self.mode, known_history_tokens=len(history), prompt_tokens=prompt_tokens,
            matched_ngram=0, matching_sources=0, distinct_next_ids=0, selected_source_end=None,
            source_segment_remaining=None, original_draft_tokens=0, selected_draft_tokens=0,
            crosses_prompt_output_boundary=False, ambiguity_scout=False, source_boundary_capped=False,
            action="zero_budget" if max_tokens == 0 else "no_match",
            decision_scope="Computed before target forward; acceptance is never a selection input")
        selected = []
        if max_tokens:
            for n in range(min(16, len(history)-1), 2, -1):
                suffix = history[-n:]
                ends = [end for end in range(len(history)-1, n-1, -1)
                        if history[end-n:end] == suffix]
                if not ends:
                    continue
                end = ends[0]  # Keep original longest-ngram/latest-match winner.
                original = list(history[end:end+max_tokens])
                distinct = len({history[e] for e in ends})
                remaining = prompt_tokens-end if end < prompt_tokens else len(history)-end
                meta.update(matched_ngram=n, matching_sources=len(ends), distinct_next_ids=distinct,
                    selected_source_end=end, source_segment_remaining=remaining,
                    original_draft_tokens=len(original),
                    crosses_prompt_output_boundary=end < prompt_tokens < end+len(original))
                if self.mode in ("ambiguity", "both") and distinct > 1:
                    meta.update(ambiguity_scout=True, action="M1_ambiguity_scout")
                else:
                    selected = original
                    if self.mode in ("source-boundary", "both") and remaining < len(original):
                        selected = original[:remaining]
                        meta.update(source_boundary_capped=True, action="source_boundary_cap")
                    else:
                        meta["action"] = "original_draft"
                break
        meta["selected_draft_tokens"] = len(selected)
        if self.max_draft_tokens in (255, 511, 1023):
            # Keep legacy <=128 metadata unchanged. The new tier must carry
            # its explicit capacity even for a scout/no-match/zero-budget.
            meta["max_draft_tokens"] = self.max_draft_tokens
        chosen_rows = len(selected)+1  # The current pending input is always real.
        meta["chosen_verifier_rows"] = chosen_rows
        meta["shape_matches_HC_32_64_128_preset_or_M1"] = chosen_rows in (1,32,64,128)
        meta["shape_hint_scope"] = "Requested shape compatibility only, not proof of actual graph hits"
        meta["real_rows_per_M32_tile_if_enabled"] = ([32]*(chosen_rows//32)
            + ([chosen_rows%32] if chosen_rows%32 else []))
        meta["selection_cpu_s"] = time.perf_counter()-start
        self.last = meta
        return selected

    def decision(self):
        return copy.deepcopy(self.last)
