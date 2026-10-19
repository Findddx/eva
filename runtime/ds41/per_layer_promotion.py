"""Default-off per-request, past-only resident K3 ownership exchanges.

This changes placement only. Executed target-call union presence is not a fresh
accepted-token profile or the persistent fifty-request personal EMA. CUDA and
weight math remain in the caller. The optional host-backing helper enqueues
placement-only DMA; default promotion never copies or binds weight bytes.
"""
import hashlib
import os
from pathlib import Path
import threading
import time
from ds41.promotion_host_backing import HostBackingMigration, configuration as host_backing_configuration

FLAG = "DS41_LAYER_PROMOTION"
MAX_SWAPS = "DS41_LAYER_PROMOTION_MAX_SWAPS"
COUNTERS = ("calls", "rows", "routed_slots", "union_credits", "boundaries",
            "eligible_boundaries", "candidates", "swaps", "protected_skips",
            "failures", "close_drains", "boundary_s", "activation_s")


def requested(environ=None):
    env = os.environ if environ is None else environ
    flag = env.get(FLAG, "0")
    if flag not in ("0", "1"):
        raise ValueError(FLAG + " must be 0 or 1")
    return flag == "1"


def configuration(environ=None):
    env = os.environ if environ is None else environ
    enabled = requested(env)
    host_backing = host_backing_configuration(env)
    if not enabled:
        return None
    value = env.get(MAX_SWAPS, "1")
    if value not in ("1", "2", "4"):
        raise ValueError(MAX_SWAPS + " must be 1, 2 or 4")
    value = dict(protocol="past_only_per_layer_resident_K3_v1", enabled=True,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                max_swaps_per_layer_boundary=int(value), minimum_call_presence=3,
                frozen_layer_floor=6, counting_unit="one_per_expert_per_successful_actual_target_MoE_call",
                scope="per_request_decode_executed_target_routes_including_rejected_rows",
                boundary="after_complete_Block_forward_HC_post_enqueue_and_MoE_cleanup",
                candidate_order="descending_past_presence_then_key",
                victim_order="ascending_past_presence_then_key_same_layer_unprotected",
                demotion_order="volatile_oldest", persistent_personal_EMA=False,
                new_GPU_slots=0, weight_DMA=False, host_backing_migration=False,
                target_math_changed=False, supported_batch=1, full_model_graph_capture_supported=False,
                GPU_validated=False, full_model_validated=False)
    if host_backing is not None:
        value.update(protocol="past_only_per_layer_resident_K3_host_backing_v1",
                     weight_DMA=True, host_backing_migration=True, host_backing=host_backing)
    return value


def validate_options(enabled, *, cpu_experts=False, adaptive=False, next_chunk=False,
                     lookahead=False):
    if enabled and (cpu_experts or adaptive or next_chunk or lookahead):
        raise ValueError("Layer promotion excludes CPU experts, continuous adaptive placement, next-chunk prefetch and lookahead")


class LayerPromotion:
    """One cache/thread/stream owner. Inactive until explicit post-check start.

    Every demotion records a distinct event generation before publishing its
    new volatile owner. Old/new events and both slots remain strongly owned
    until the construction compute stream drains, including partial failures.
    """
    def __init__(self, cache):
        self.cache = cache
        self.identity = configuration()
        if self.identity is None:
            raise ValueError("Construct LayerPromotion only when explicitly requested")
        self.owner = threading.get_ident()
        self.active = self.started = self.closed = self.failed = False
        self.stream = self.event_factory = self.current_stream = None
        self.pending_layer = None
        self.executing_layer = None
        self.counts = {}
        self.counters = dict.fromkeys(COUNTERS, 0)
        self.geometries = {}
        self.event_leases = []
        self.protected = frozenset()
        self.protected_slots = {}
        self.quotas = {}
        self.pointer_table = None
        self.stage_limit = None
        self.host_backing = HostBackingMigration(self) if self.identity.get("host_backing_migration", False) else None

    def _inc(self, name, value=1):
        self.counters[name] += value
        key = "layer_promotion_" + name
        self.cache.stats[key] = self.cache.stats.get(key, 0) + value

    def poison(self):
        if not self.failed:
            self._inc("failures")
        self.failed = True
        self.cache._layer_promotion_failed = True
        backing = getattr(self, "host_backing", None)
        if backing is not None and not backing.failed:
            backing.failed = True
            backing._inc("failures")

    def _check(self, *, stream=True):
        if threading.get_ident() != self.owner or self.failed or self.closed:
            raise RuntimeError("Layer promotion owner failed/closed or changed thread; discard cache")
        if stream and self.current_stream() != self.stream:
            raise RuntimeError("Layer promotion requires its original compute stream")
        cache = self.cache
        validate_options(True, cpu_experts=getattr(cache, "cpu_backend", None) is not None,
                         adaptive=getattr(cache, "adaptive_gpu_enabled", False),
                         next_chunk=getattr(getattr(cache, "_chunk_prefetch_config", None), "enabled", False))
        if self.stage_limit is not None and cache.stage_limit != self.stage_limit:
            raise RuntimeError("Layer promotion volatile capacity changed")
        table = getattr(cache, "_fast_pointer_table", None)
        if (not getattr(cache, "_fast_pointer_requested", False) or table is None
                or self.pointer_table is not None and table is not self.pointer_table
                or getattr(table, "closed", False)):
            raise RuntimeError("Layer promotion requires the same cache's initialized fast pointer table")

    def start(self, stream, event_factory, current_stream, *, protected_frozen_keys=frozenset()):
        if self.started or self.closed or self.failed or threading.get_ident() != self.owner:
            raise RuntimeError("Start layer promotion exactly once on its cache owner")
        if type(protected_frozen_keys) is not frozenset:
            raise ValueError("Layer promotion protected keys must be a frozenset")
        cache = self.cache
        if cache._active_keys:
            raise RuntimeError("Start layer promotion only after MoE active leases finish")
        pipeline = getattr(cache, "_demand_h2d_pipeline", None)
        if pipeline is not None and (pipeline.current is not None or pipeline.quarantined or pipeline.failed or pipeline.closed):
            raise RuntimeError("Start layer promotion only with an idle healthy pipeline")
        combined = protected_frozen_keys | frozenset(getattr(cache, "_prompt_protected_frozen_keys", ()))
        if not combined.issubset(cache.frozen):
            raise ValueError("Protected layer-promotion keys must already be frozen")
        self.stream, self.event_factory, self.current_stream = stream, event_factory, current_stream
        self.pointer_table = getattr(cache, "_fast_pointer_table", None)
        self.stage_limit = cache.stage_limit
        self._check()
        if set(cache.frozen).intersection(cache.vram):
            raise RuntimeError("Duplicate GPU ownership before layer promotion")
        self.protected = combined
        self.protected_slots = {key:(cache.frozen[key], cache.frozen[key].storage,
                                    cache.frozen[key].storage.numel(), cache.frozen[key].storage.data_ptr())
                                for key in combined}
        self.quotas = {layer:sum(key[0] == layer for key in cache.frozen) for layer in range(cache.n_layers)}
        self.active = self.started = True

    def begin_layer(self, layer):
        if self.closed or self.failed:
            raise RuntimeError("Layer promotion owner failed/closed; discard cache")
        if not self.active:
            return
        try:
            self._check()
            if self.executing_layer is not None or self.pending_layer is not None:
                raise RuntimeError("An unfinished layer must not be retried or resumed")
            self.executing_layer = layer
        except BaseException:
            self.poison()
            raise

    def before_call(self, layer):
        if self.closed or self.failed:
            raise RuntimeError("Layer promotion owner failed/closed; discard cache")
        if not self.active:
            return
        try:
            self._check()
            if self.executing_layer != layer:
                raise RuntimeError("Layer promotion requires its complete Block.forward scope")
        except BaseException:
            self.poison()
            raise

    def record_call(self, layer, ids, rows, topk):
        if self.closed or self.failed:
            raise RuntimeError("Layer promotion owner failed/closed; discard cache")
        if not self.active:
            return
        try:
            self._check()
            if (type(layer) is not int or not 0 <= layer < self.cache.n_layers
                    or type(rows) is not int or rows < 1 or topk != 6
                    or not ids or len(ids) != len(set(ids))
                    or any(type(expert) is not int or not 0 <= expert < 384 for expert in ids)
                    or len(ids) > rows * topk):
                raise ValueError("Layer promotion requires a validated actual top6 target-call union")
            if self.executing_layer != layer or self.pending_layer not in (None, layer) or self.cache._active_keys:
                raise RuntimeError("Finish the previous whole layer before recording another layer")
            pipeline = getattr(self.cache, "_demand_h2d_pipeline", None)
            if pipeline is not None and (pipeline.current is not None or pipeline.quarantined):
                raise RuntimeError("Record promotion counts after pipeline source cleanup")
            self.pending_layer = layer
            for expert in ids:
                key = layer, expert
                self.counts[key] = self.counts.get(key, 0) + 1
            for name, value in (("calls", 1), ("rows", rows), ("routed_slots", rows*topk), ("union_credits", len(ids))):
                self._inc(name, value)
            self.geometries[str(rows)] = self.geometries.get(str(rows), 0) + 1
        except BaseException:
            self.poison()
            raise

    def _k3(self, key):
        return all(part[2] == 3 for part in self.cache._sig(*key))

    def finish_layer(self, layer):
        if self.closed or self.failed:
            raise RuntimeError("Layer promotion owner failed/closed; discard cache")
        if not self.active:
            return 0
        started = time.perf_counter()
        try:
            self._check()
            cache = self.cache
            if self.executing_layer != layer or self.pending_layer != layer or cache._active_keys:
                raise RuntimeError("Layer promotion requires a completed observed whole layer")
            pipeline = getattr(cache, "_demand_h2d_pipeline", None)
            if pipeline is not None and (pipeline.current is not None or pipeline.quarantined or pipeline.failed):
                raise RuntimeError("Layer promotion cannot overlap pipeline cleanup")
            if self.host_backing is not None:
                self.host_backing.poll()
                self.event_leases = self.host_backing.poll_promotion_events(self.event_leases)
            self._inc("boundaries")
            quota = self.quotas[layer]
            frozen = [key for key in cache.frozen if key[0] == layer]
            if len(frozen) != quota:
                raise RuntimeError("Layer promotion frozen quota changed externally")
            for key, (slot, storage, size, pointer) in self.protected_slots.items():
                if key[0] == layer and (cache.frozen.get(key) is not slot or slot.storage is not storage):
                    raise RuntimeError("A protected frozen slot changed owner")
            self.pending_layer = None
            self.executing_layer = None
            if quota < 6:
                return 0
            protected = self.protected | frozenset(getattr(cache, "_prompt_protected_frozen_keys", ()))
            candidates = sorted((key for key in cache.vram if key[0] == layer
                                 and self.counts.get(key, 0) >= 3 and self._k3(key)),
                                key=lambda key:(-self.counts[key], key))
            self._inc("eligible_boundaries")
            self._inc("candidates", len(candidates))
            swaps = 0
            for hot in candidates:
                hot_slot = cache.vram[hot]
                compatible = [key for key in frozen if self._k3(key)
                              and cache._sig(*key) == cache._sig(*hot)
                              and cache.frozen[key].storage.numel() == hot_slot.storage.numel()]
                self._inc("protected_skips", sum(key in protected for key in compatible))
                victims = [key for key in compatible if key not in protected]
                cold = min(victims, key=lambda key:(self.counts.get(key, 0), key)) if victims else None
                if cold is None or self.counts.get(cold, 0) >= self.counts[hot]:
                    continue
                cold_slot = cache.frozen[cold]
                event = self.event_factory()
                # Retain BEFORE record: enqueue can partially succeed then throw.
                self.event_leases.append((cold_slot.event, event, hot_slot, cold_slot))
                migrated = (self.host_backing is not None and
                            self.host_backing.exchange(hot, cold, hot_slot, cold_slot, event))
                if not migrated:
                    event.record(self.stream)
                cold_slot.event, cold_slot.seen = event, True
                del cache.vram[hot]
                del cache.frozen[cold]
                cache.frozen[hot] = hot_slot
                frozen.remove(cold)
                frozen.append(hot)
                cache.vram[cold] = cold_slot
                cache.vram.move_to_end(cold, last=False)
                swaps += 1
                self._inc("swaps")
                if swaps == self.identity["max_swaps_per_layer_boundary"]:
                    break
            return swaps
        except BaseException:
            self.poison()
            raise
        finally:
            self._inc("boundary_s", time.perf_counter() - started)

    def verify_metadata(self):
        self._check(stream=False)
        if configuration() != self.identity:
            self.poison()
            raise RuntimeError("Layer promotion configuration/source changed")
        if self.pending_layer is not None or self.executing_layer is not None or self.cache._active_keys:
            self.poison()
            raise RuntimeError("Incomplete layer at layer-promotion output boundary")
        actual_quotas = {layer:sum(key[0] == layer for key in self.cache.frozen) for layer in self.quotas}
        if actual_quotas != self.quotas or set(self.cache.frozen).intersection(self.cache.vram):
            self.poison()
            raise RuntimeError("Layer promotion GPU ownership/quota changed")
        if any(self.cache.frozen.get(key) is not slot or slot.storage is not storage
               or storage.numel() != size or storage.data_ptr() != pointer
               for key, (slot, storage, size, pointer) in self.protected_slots.items()):
            self.poison()
            raise RuntimeError("Layer promotion lost protected frozen ownership")

    def close(self):
        if self.closed:
            return
        if threading.get_ident() != self.owner:
            self.poison()
            raise RuntimeError("Close layer promotion on its owner thread")
        if self.executing_layer is not None or self.pending_layer is not None:
            self.poison()
        try:
            if self.started:
                self.stream.synchronize()
                self._inc("close_drains")
            if self.host_backing is not None:
                self.host_backing.close_after_compute_drain()
        except BaseException:
            self.poison()
            raise
        self.active = False
        self.closed = True
        self.event_leases.clear()

    def status(self):
        value = dict(configuration=dict(self.identity), active=self.active, started=self.started,
                    closed=self.closed, failed=self.failed, pending_layer=self.pending_layer,
                    executing_layer=self.executing_layer,
                    counters=dict(self.counters), actual_call_rows=dict(self.geometries),
                    count_entries=len(self.counts), protected_frozen_experts=len(self.protected),
                    event_generation_leases=len(self.event_leases), counts_reset_at_decode_start=True,
                    volatile_capacity_at_start=self.stage_limit,
                    initial_frozen_layer_quotas={str(k):v for k,v in self.quotas.items()},
                    current_frozen_experts=len(self.cache.frozen), current_volatile_experts=len(self.cache.vram),
                    counter_scope="One explicit per-request decode session; activation_s is before timed decode; boundary_s is within it; close_drains is after output",
                    count_scope=self.identity["scope"], native_fastpath_preserved=True)
        if self.host_backing is not None:
            value["configuration"]["host_backing"] = dict(self.identity["host_backing"])
            value["host_backing"] = self.host_backing.status()
        return value
