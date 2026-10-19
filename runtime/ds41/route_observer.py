"""Opt-in CPU adapter for actual target MoE-call route statistics.

This module never imports Torch/numpy, converts tensors, synchronizes devices,
or retains input routes. The caller supplies an existing CPU list once per real
cache.moe invocation, not once per whole layer or predicted draft. The sink is
RequestRoutes.record_target_call (or the same protocol).
"""
from contextlib import contextmanager
from contextvars import ContextVar
from numbers import Integral


PHASES = ("prefill", "decode")
MAX_ROUTED_SLOTS = 256


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


class TargetRouteObserver:
    """A request-scoped callable; disabled and unscoped by default.

    ``observer(layer, routed_ids)`` consumes an existing flat Python list/tuple.
    ``record_rows`` consumes an existing rectangular CPU route list plus an
    actual MoE-call's [start,end) row span. Both preserve repeated IDs for
    route-use counts; the sink deduplicates presence within each call.

    Call only for actual target executions. Rejected verification/replay work
    is real execution; prewarm/parity/probes/draft calls require suspend(). The
    request lifecycle owner must discard failed requests, complete successful
    requests, and set the actual token exposures independently.
    """

    def __init__(self, request_routes, *, enabled=False):
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        record = getattr(request_routes, "record_target_call", None)
        if not callable(record):
            raise TypeError("request_routes must expose record_target_call")
        identity = getattr(request_routes, "identity", None)
        self.layers = _integer(getattr(identity, "layers", None), "layers", 1)
        self.experts = _integer(getattr(identity, "experts", None), "experts", 1)
        capacity = getattr(identity, "max_routed_slots", MAX_ROUTED_SLOTS)
        if type(capacity) is not int or capacity not in (256, 384, 1536):
            raise ValueError("profile identity must select 256, 384 or 1536 routed slots")
        if getattr(request_routes, "max_routed_slots", capacity) != capacity:
            raise ValueError("sink capacity must match its profile identity")
        self._max_routed_slots = capacity
        self._record = record
        self._enabled = enabled
        self._phase = ContextVar(f"target_route_phase_{id(self)}", default=None)
        self._suspended = ContextVar(f"target_route_suspended_{id(self)}", default=0)

    @property
    def enabled(self):
        return self._enabled

    @property
    def max_routed_slots(self):
        """Maximum for one real call, bound to the request's identity."""
        return self._max_routed_slots

    @property
    def active(self):
        return self._enabled and not self._suspended.get() and self._phase.get() in PHASES

    @contextmanager
    def scope(self, phase):
        """Set prefill/decode, or None to disable; always restore on exit.

        A nested scope cannot override suspend(). Contexts are task/thread
        local; an unrelated worker thread starts unscoped rather than silently
        inheriting the inference thread's phase.
        """
        if phase is not None and phase not in PHASES:
            raise ValueError("phase must be prefill, decode, or None")
        token = self._phase.set(phase)
        try:
            yield self
        finally:
            self._phase.reset(token)

    @contextmanager
    def suspend(self):
        """Exclude diagnostic/draft/prewarm work, including nested scopes."""
        token = self._suspended.set(self._suspended.get() + 1)
        try:
            yield self
        finally:
            self._suspended.reset(token)

    def __call__(self, layer, routed_ids):
        """Record one real call; return False when inactive, True on success.

        Inactive calls do not inspect inputs. Active calls reject tensor-like
        inputs instead of calling .tolist/.cpu or iterating device scalars.
        All input validation completes before the sink is allowed to mutate.
        """
        if not self.active:
            return False
        layer = _integer(layer, "layer")
        if layer >= self.layers:
            raise ValueError("layer outside profile identity")
        if not isinstance(routed_ids, (list, tuple)):
            raise TypeError("routed_ids must be an existing flat CPU list/tuple")
        if not 1 <= len(routed_ids) <= self.max_routed_slots:
            raise ValueError(f"actual MoE call must contain 1..{self.max_routed_slots} routed slots, including duplicates")
        ids = []
        for value in routed_ids:
            expert = _integer(value, "expert")
            if expert >= self.experts:
                raise ValueError("expert outside profile identity")
            ids.append(expert)
        self._record(self._phase.get(), layer, ids)
        return True

    def record_rows(self, layer, cpu_rows, start=0, end=None):
        """Record only an executed chunk from an existing indices.tolist list.

        This never re-chunks routes. The caller supplies the exact span passed
        to one cache.moe call, including frozen/hot/cold experts alike.
        """
        if not self.active:
            return False
        if not isinstance(cpu_rows, (list, tuple)):
            raise TypeError("cpu_rows must be an existing rectangular CPU list/tuple")
        start = _integer(start, "start")
        end = len(cpu_rows) if end is None else _integer(end, "end")
        if not 0 <= start < end <= len(cpu_rows):
            raise ValueError("chunk must be a nonempty in-bounds [start,end) range")
        first = cpu_rows[start]
        if not isinstance(first, (list, tuple)) or not first:
            raise ValueError("route rows must be nonempty CPU lists/tuples")
        width = len(first)
        if (end - start) * width > self.max_routed_slots:
            raise ValueError(f"actual chunk exceeds {self.max_routed_slots} routed slots, including duplicates")
        flat = []
        for index in range(start, end):
            row = cpu_rows[index]
            if not isinstance(row, (list, tuple)) or len(row) != width:
                raise ValueError("chunk route rows must have the same topk")
            flat.extend(row)
        return self(layer, flat)
