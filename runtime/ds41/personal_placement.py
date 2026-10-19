"""Turn a past request boundary into an optional startup placement prior."""
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np


PROTECTION_PROTOCOL = "personal_prompt_frozen_byte_protection_v1"
GIB = 1024**3


def prompt_protection_configuration(gib, *, apply, state, prompt_gpu_reseed, counts,
                                    gpu_policy, cpu_experts, adaptive_gpu, verification_cap):
    """Explicit opt-in metadata; zero preserves the old configuration schema."""
    if type(gib) not in (int, float) or not math.isfinite(gib) or not 0 <= gib <= 32:
        raise ValueError("personal-prompt-retain-gib must be finite and in [0,32]")
    if gib == 0:
        return None
    budget = int(gib * GIB)
    if not budget:
        raise ValueError("personal-prompt-retain-gib must represent at least one byte")
    if (not apply or not state or not prompt_gpu_reseed or not counts
            or gpu_policy not in ("auto", "profile") or cpu_experts or adaptive_gpu
            or type(verification_cap) is not int
            or not (1 <= verification_cap <= 128 or verification_cap == 256)):
        raise ValueError("Personal prompt protection requires apply/state, prompt GPU reseed, counts profile, GPU-only, adaptive OFF and cap<=128 or256")
    return dict(protocol=PROTECTION_PROTOCOL, requested=True, requested_bytes=budget,
        quota_unit="actual_existing_frozen_slot_storage_bytes",
        ranking="applied_historical_prior_per_slot_byte_stable_key",
        eligible="startup_actual_frozen_compatible_K3_positive_prior",
        application="post_prefill_prompt_gpu_reseed_only", history_prompt_scores_blended=False,
        extra_weight_allocations=False, production_GPU_verified=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def _keys_digest(keys):
    return hashlib.sha256(json.dumps([list(k) for k in sorted(keys)],
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def frozen_snapshot(cache):
    """CPU owner metadata only. No tensor clone, GPU readback or addresses."""
    records = []
    for key, slot in sorted(cache.frozen.items()):
        payload, capacity = int(cache._fields(*key)[1]), int(slot.storage.numel())
        if payload <= 0 or capacity < payload:
            raise ValueError("Invalid actual frozen slot capacity")
        records.append([int(key[0]), int(key[1]), payload, capacity])
    ram_keys = list(cache.ram)
    return dict(keys=[r[:2] for r in records], records=records, experts=len(records),
        keys_sha256=_keys_digest(tuple(r[:2]) for r in records),
        records_sha256=hashlib.sha256(json.dumps(records,separators=(",", ":")).encode()).hexdigest(),
        payload_bytes=sum(r[2] for r in records), slot_capacity_bytes=sum(r[3] for r in records),
        RAM=dict(experts=len(ram_keys), keys_sha256=_keys_digest(ram_keys),
            LRU_keys_sha256=hashlib.sha256(json.dumps([list(k) for k in ram_keys],separators=(",", ":")).encode()).hexdigest(),
            payload_bytes=int(cache.ram_nbytes), allocation_charge_bytes=int(cache.ram_charge_bytes)))


def select_prompt_protection(cache, applied_prior, budget, *, inactive_reason="no_candidate"):
    """Choose a bounded subset of actual startup frozen weights, not new IO.

    Prior is the already-applied integer byte-value prior, not prompt counts.
    Exact rational density ordering is deterministic but not exact knapsack.
    """
    if type(budget) is not int or not 0 < budget <= 32 * GIB:
        raise ValueError("Invalid personal frozen byte quota")
    report = dict(requested_bytes=budget, active=False, reason=inactive_reason,
        eligible_experts=0, eligible_slot_capacity_bytes=0, protected_keys=[],
        protected_keys_sha256=_keys_digest(()), protected_experts=0,
        protected_slot_capacity_bytes=0, protected_payload_bytes=0,
        unfilled_bytes=budget, selection="density_greedy_existing_frozen_not_exact_knapsack")
    if applied_prior is None:
        return frozenset(), report
    if (not isinstance(applied_prior,np.ndarray) or applied_prior.shape != (cache.n_layers,384)
            or applied_prior.dtype.kind != "i" or np.any(applied_prior < 0)
            or np.any(applied_prior > 1_000_000_000)):
        raise ValueError("Expected the applied signed nonnegative bounded historical prior")
    signatures = {cache._sig(*key) for key,slot in cache.vram.items()
                  if slot.storage.numel() == cache._volatile_slot_bytes}
    signatures.update(sig for sig,pool in cache.pools.items()
                      if any(slot.storage.numel() == cache._volatile_slot_bytes for slot in pool))
    candidates = []
    for key,slot in cache.frozen.items():
        fields = cache._fields(*key)
        capacity, payload, value = int(slot.storage.numel()), int(fields[1]), int(applied_prior[key])
        if (int(fields[2]) == 3 and capacity == cache._volatile_slot_bytes
                and cache._sig(*key) in signatures and value > 0):
            if not 0 < payload <= capacity:
                raise ValueError("Invalid eligible frozen payload/capacity")
            candidates.append((-Fraction(value,capacity),key,payload,capacity))
    candidates.sort(key=lambda row:(row[0],row[1]))
    selected,spent,payload = [],0,0
    for _,key,size,capacity in candidates:
        if spent + capacity <= budget:
            selected.append(key); spent += capacity; payload += size
    protected = frozenset(selected)
    report.update(active=bool(protected),reason="selected_existing_history" if protected else "no_eligible_history_fits_quota",
        applied_prior_int64_sha256=hashlib.sha256(applied_prior.astype('<i8').tobytes()).hexdigest(),
        eligible_experts=len(candidates),eligible_slot_capacity_bytes=sum(r[3] for r in candidates),
        protected_keys=[list(k) for k in sorted(protected)],protected_keys_sha256=_keys_digest(protected),
        protected_experts=len(protected),protected_slot_capacity_bytes=spent,
        protected_payload_bytes=payload,unfilled_bytes=budget-spent)
    return protected,report


def snapshot_retention(startup, current, protected_keys, *, prompt_counts=None):
    """Set-retention evidence; no claim about decode hits or same slot pointer."""
    before = {tuple(r[:2]):(r[2],r[3]) for r in startup["records"]}
    after = {tuple(r[:2]):(r[2],r[3]) for r in current["records"]}
    kept,removed,added = before.keys() & after.keys(),before.keys()-after.keys(),after.keys()-before.keys()
    protected = frozenset(protected_keys)
    intact = {k for k in protected if k in before and k in after and before[k] == after[k]}
    result = dict(retained_experts=len(kept),removed_experts=len(removed),added_experts=len(added),
        retained_startup_payload_bytes=sum(before[k][0] for k in kept),
        retained_startup_slot_capacity_bytes=sum(before[k][1] for k in kept),
        removed_startup_payload_bytes=sum(before[k][0] for k in removed),
        removed_startup_slot_capacity_bytes=sum(before[k][1] for k in removed),
        added_current_payload_bytes=sum(after[k][0] for k in added),
        added_current_slot_capacity_bytes=sum(after[k][1] for k in added),
        protected_retained_experts=len(intact),protected_retained_slot_capacity_bytes=sum(after[k][1] for k in intact),
        protected_missing_or_changed_keys=[list(k) for k in sorted(protected-intact)],
        all_protected_keys_and_capacities_retained=intact == protected,
        scope="logical_frozen_key_and_byte_retention_not_decode_coverage_or_pointer_identity")
    if prompt_counts is not None:
        if (not isinstance(prompt_counts,np.ndarray) or prompt_counts.dtype.kind != 'i'
                or prompt_counts.ndim != 2 or prompt_counts.shape[1] != 384 or np.any(prompt_counts<0)):
            raise ValueError("Expected fresh nonnegative prompt counts")
        result['prompt_positive_frozen_experts'] = sum(int(prompt_counts[k]) > 0 for k in after)
        result['prompt_route_uses_in_frozen'] = sum(int(prompt_counts[k]) for k in after)
    return result


def startup_prior(profile, payload_bytes):
    """Adapt call presence to the existing counts/bytes density ranker.

    No weights, routes or live GPU owners change here. The caller must load a
    completed boundary snapshot, never a future/partial evaluation window.
    Integer normalization is a bounded ranking approximation, not raw uses.
    """
    sizes = np.asarray(payload_bytes)
    scores = profile.scores('decode')
    if (sizes.shape != scores.shape or sizes.dtype.kind not in 'iu'
            or np.any(sizes <= 0) or not np.isfinite(scores).all() or np.any(scores < 0)):
        raise ValueError('invalid profile/payload placement geometry')
    if not profile.completed_requests or profile.completed_requests % profile.update_every:
        raise ValueError('startup placement requires a completed request boundary')
    value = scores * sizes
    if not np.isfinite(value).all():
        raise ValueError('nonfinite byte value')
    if not np.any(value):
        return None, dict(applied=False, reason='no_completed_decode_signal')
    # ranked_sets divides counts by payload/host charge. Multiplying presence
    # by payload first estimates bytes saved rather than access hits per byte.
    prior = np.rint(value / value.max() * 1_000_000_000).astype(np.int64)
    prior[(value > 0) & (prior == 0)] = 1
    report = dict(applied=True, scope='past_completed_boundary_call_union',
                  trained_through_request=profile.completed_requests, phase='decode',
                  objective='call_presence_times_payload_then_existing_density_ranker',
                  normalization='maximum_prior_1000000000_minimum_positive_1',
                  prior_int64_sha256=hashlib.sha256(prior.astype('<i8').tobytes()).hexdigest(),
                  gpu_policy='existing_per_layer_floor_then_density',
                  ram_policy='existing_unconditional_density_not_joint_optimal',
                  application='next_process_startup_preload_only')
    return prior, report
