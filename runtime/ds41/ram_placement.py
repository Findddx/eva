"""Pure metadata planning for explicit prompt-only host expert placement.

No Torch, weight IO, environment lookup or cache mutation. Density selection is
a deterministic heuristic for indivisible records, not an exact knapsack solver.
"""
from dataclasses import dataclass
from functools import cmp_to_key
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class RamCandidate:
    key: tuple
    nbytes: int
    charge: int
    frequency: int
    value: int
    resident: bool


@dataclass(frozen=True)
class RamPlacementPlan:
    selected: tuple
    candidate_count: int
    eligible_value: int
    selected_value: int
    predicted_charge_bytes: int
    objective: str

    @property
    def keys(self):
        return tuple(candidate.key for candidate in self.selected)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_prompt_counts(counts, n_layers=None):
    if (not isinstance(counts, np.ndarray) or counts.ndim != 2 or counts.shape[1] != 384
            or counts.shape[0] < 1 or (n_layers is not None and counts.shape[0] != n_layers)
            or counts.dtype.kind != "i" or np.any(counts < 0)):
        raise ValueError("prompt counts must be a fresh nonnegative signed integer [layers,384] array")


def plan_prompt_ram(counts, *, record_bytes, allocation_charges, resident_keys,
                    frozen_keys, budget_bytes, minimum_count=1, objective="bytes"):
    """Select prompt-positive non-frozen records under charged host capacity.

    value = prompt frequency * payload bytes (bytes), or frequency (hits).
    Existing residents must pass their actual stored charge; absent records use
    an allocator-config estimate. Freshness is a caller sampling-boundary
    contract: a histogram alone cannot identify leaked future routing.
    """
    validate_prompt_counts(counts)
    budget_bytes = _integer(budget_bytes, "budget_bytes")
    minimum_count = _integer(minimum_count, "minimum_count", 1)
    if objective not in ("bytes", "hits"):
        raise ValueError("RAM placement objective must be 'bytes' or 'hits'")
    frozen, resident = set(frozen_keys), set(resident_keys)
    candidates = []
    for layer in range(counts.shape[0]):
        for expert in range(384):
            key, frequency = (layer, expert), int(counts[layer, expert])
            if key in frozen or frequency < minimum_count:
                continue
            try:
                nbytes = _integer(record_bytes[key], "record_bytes", 1)
                charge = _integer(allocation_charges[key], "allocation_charges", 1)
            except KeyError as error:
                raise ValueError(f"missing RAM record metadata for {key}") from error
            if charge < nbytes:
                raise ValueError("allocation charge cannot be smaller than record payload")
            value = frequency * nbytes if objective == "bytes" else frequency
            candidates.append(RamCandidate(key, nbytes, charge, frequency, value, key in resident))

    def compare(a, b):
        # Python integers preserve exact density ordering for large histograms.
        left, right = a.value * b.charge, b.value * a.charge
        if left != right:
            return -1 if left > right else 1
        if a.resident != b.resident:
            return -1 if a.resident else 1
        return (a.key > b.key) - (a.key < b.key)

    candidates.sort(key=cmp_to_key(compare))
    selected, spent = [], 0
    for candidate in candidates:
        if spent + candidate.charge <= budget_bytes:
            selected.append(candidate)
            spent += candidate.charge
    return RamPlacementPlan(tuple(selected), len(candidates), sum(c.value for c in candidates),
                            sum(c.value for c in selected), spent, objective)
