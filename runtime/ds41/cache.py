"""Three-tier routed experts: frozen VRAM, a volatile VRAM LRU, pinned RAM, NVMe.

Routed GEMMs go through the fused coop kernel. A miss copies bytes into a
preallocated slot. The slot's address is published in a GPU pointer table.

The optional host allocator reserve is an unmeasured allocation-policy
hypothesis. It partitions the existing host budget; it creates no buffer pool.
"""
import os
import struct
import time
import heapq
import hashlib
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch

from ds41.verification import verification_max_rows
from ds41.ram_placement import plan_prompt_ram, validate_prompt_counts
from ds41.direct_io import DIRECT_READ_STATS, DirectExpertReader, requested as direct_io_requested
from ds41.pointer_table import FastPointerTable, requested as fast_pointer_requested
from ds41.chunk_prefetch import (COUNTERS as CHUNK_PREFETCH_COUNTERS, ChunkPrefetchConfig,
                                 ChunkPrefetchState, PendingOrigin, GIB, validate_next_ids)
from ds41.route_snapshot_runtime import (RouteSnapshotRuntime, requested as route_snapshot_requested,
                                        configuration as route_snapshot_configuration, disabled_status)
from ds41.demand_h2d_pipeline import (DemandH2DPipeline, COUNTERS as DEMAND_H2D_COUNTERS,
                                     requested as demand_h2d_requested,
                                     configuration as demand_h2d_configuration)
from ds41.native2048_runtime import (requested as coop2048_requested, initialize as initialize_coop2048,
                                    status as coop2048_status, validate_configuration as validate_coop2048)
from ds41.native3072_runtime import (requested as coop3072_requested, initialize as initialize_coop3072,
                                    status as coop3072_status, validate_configuration as validate_coop3072)
from ds41.per_layer_promotion import (LayerPromotion, COUNTERS as LAYER_PROMOTION_COUNTERS,
                                      requested as layer_promotion_requested)
from ds41.promotion_host_backing import requested as promotion_host_backing_requested

SLOT = 128
SLOT_FMT = struct.Struct("<QIHH" + "HHHHIIIiIII" * 3)
N_EXPERTS = 384
PTR_ROWS = 9


def pinned_allocation_charge(nbytes):
    """Mirror installed PyTorch's pinned allocator rounding configuration.

    The legacy CUDA/HIP env keys take precedence over PYTORCH_ALLOC_CONF.
    Unrounded cached blocks may be larger than a request; aggregate host stats
    and the dynamic MemAvailable guard cover that residual fragmentation.
    """
    config = next((os.environ[key] for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
                   if os.environ.get(key)), "")
    settings = {}
    for part in config.split(","):
        if ":" in part:
            key, value = part.split(":", 1)
            settings[key.strip()] = value.strip()
    threshold = int(settings.get("pinned_max_round_threshold_mb", str(1 << 40))) * 1024**2
    cached = int(settings.get("pinned_max_cached_size_mb", str(1 << 40))) * 1024**2
    if nbytes and nbytes <= min(threshold, cached):
        return 1 << (int(nbytes) - 1).bit_length()
    return int(nbytes)


def mem_available_bytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def _signature(fields):
    sig = []
    for pi in range(3):
        base = 4 + pi * 11
        in_f, out_f, k, _pad, suh_b, svh_b, tr_b = fields[base : base + 7]
        sig.append((int(in_f), int(out_f), int(k), int(suh_b), int(svh_b), int(tr_b)))
    return tuple(sig)


class GpuExpert:
    def __init__(self, parts, storage=None):
        self.parts = parts
        self.storage = storage
        self.event = torch.cuda.Event()
        self.seen = False

    def retire(self):
        if self.seen:
            self.event.synchronize()
            self.seen = False


class PinnedExpert:
    def __init__(self, host, parts, nbytes, allocation_bytes=None):
        self.host = host
        self.parts = parts
        self.nbytes = nbytes
        self.allocation_bytes = nbytes if allocation_bytes is None else allocation_bytes
        self.alloc_s = self.pread_s = self.views_s = 0.
        self.direct_io_stats = {}


class ExpertCache:
    def __init__(self, pack_path: str, kmap: dict, vram_experts: int, ram_experts: int):
        self._route_snapshot_runtime = RouteSnapshotRuntime(self) if route_snapshot_requested(os.environ) else None
        self._route_snapshot_failed = False
        self._chunk_prefetch_config = ChunkPrefetchConfig.from_env(os.environ)
        self._chunk_prefetch = (ChunkPrefetchState(self._chunk_prefetch_config)
                                if self._chunk_prefetch_config.enabled else None)
        self._demand_h2d_pipeline = DemandH2DPipeline() if demand_h2d_requested(os.environ) else None
        self._demand_h2d_failed = False
        self._demand_h2d_closed = False
        if self._demand_h2d_pipeline is not None and self._chunk_prefetch_config.enabled:
            raise ValueError("demand H2D pipeline cannot be combined with next-chunk prefetch")
        coop512 = os.environ.get("DS41_MOE_COOP512", "0")
        if coop512 not in ("0", "1"):
            raise ValueError("DS41_MOE_COOP512 must be 0 or 1")
        self._coop512_requested = coop512 == "1"
        self._coop512_failed = False
        self._coop512_devices = {}
        self._coop2048_requested = coop2048_requested()
        self._coop2048_failed = False
        self._native2048_runtime = None
        if self._coop2048_requested:
            validate_coop2048(cap=verification_max_rows())
        self._coop3072_requested = coop3072_requested()
        self._coop3072_failed = False
        self._native3072_runtime = None
        if self._coop3072_requested:
            validate_coop3072(cap=verification_max_rows())
        self._expert_direct_requested = direct_io_requested()
        self._direct_reader = None
        self._fast_pointer_requested = fast_pointer_requested(os.environ)
        if self._route_snapshot_runtime is not None and not self._fast_pointer_requested:
            raise ValueError("DS41_ROUTE_SNAPSHOT=1 requires DS41_FAST_POINTER_BIND=1 for host-DMA retirement")
        self._layer_promotion_failed = False
        promotion_host_backing_requested()  # Validate the opt-in dependency before opening the pack.
        self._layer_promotion = LayerPromotion(self) if layer_promotion_requested() else None
        if self._layer_promotion is not None and (not self._fast_pointer_requested or not self._coop512_requested):
            raise ValueError("Layer promotion requires FAST_POINTER_BIND=1 and MOE_COOP512=1")
        self._fast_pointer_table = None
        self.pack = os.open(pack_path, os.O_RDONLY)
        self.kmap = kmap
        self.vram_experts = vram_experts
        self.ram_experts = ram_experts
        magic = os.pread(self.pack, 8, 0)
        if magic != b"DS41PK01":
            raise RuntimeError(f"bad pack magic {magic!r}")
        _version, n_layers, n_experts, slot = struct.unpack("<IIII", os.pread(self.pack, 16, 8))
        if (n_experts, slot) != (N_EXPERTS, SLOT):
            raise RuntimeError(f"unexpected pack {n_experts} {slot}")
        self.n_layers = n_layers
        raw = os.pread(self.pack, n_layers * N_EXPERTS * SLOT, 24)
        self.slots = [SLOT_FMT.unpack_from(raw, i * SLOT) for i in range(n_layers * N_EXPERTS)]
        self._volatile_slot_bytes = max(f[1] for f in self.slots)
        # A byte limit remains valid when EXL3 K (and record size) varies by layer.
        self.ram_budget_bytes = int(sum(f[1] for f in self.slots) / len(self.slots) * ram_experts)
        self.host_allocator_reserve_bytes = 0
        self.ram_nbytes = 0
        self.ram_charge_bytes = 0
        self.minimum_available_bytes = int(float(os.environ.get("DS41_HOST_RESERVE_GIB", "8")) * 1024**3)
        self.host_reclaim_margin_bytes = int(float(os.environ.get("DS41_HOST_RECLAIM_GIB", "1")) * 1024**3)
        self.minimum_vram_available_bytes = int(float(os.environ.get("DS41_VRAM_RESERVE_GIB", "0.8")) * 1024**3)
        self.warm_unseen = os.environ.get("DS41_WARM_UNSEEN", "0") == "1"
        self.prefetch_layers = max(0, int(os.environ.get("DS41_DISK_PREFETCH_LAYERS", "2")))
        self.prefetch_limit = max(1, int(os.environ.get("DS41_DISK_PREFETCH_LIMIT", "12")))
        self.vram = OrderedDict()
        self.ram = OrderedDict()
        self.frozen = {}
        self.pools = defaultdict(list)
        self.sigs = {}
        self.stage_limit = 512
        self.last_ids = {}
        self.stats = {"vram": 0, "frozen_hits": 0, "volatile_hits": 0,
                      "ram": 0, "nvme": 0, "ram_bytes": 0, "nvme_bytes": 0,
                      "reuse": 0, "reuse_n": 0, "io": 0.0, "io_wait": 0.0,
                      "io_reads": 0, "io_bytes": 0, "prefetch_hits": 0,
                      "prefetch_reads": 0, "prefetch_bytes": 0, "h2d_bytes": 0,
                      "io_alloc": 0., "io_pread": 0., "io_views": 0.}
        self.counts = np.zeros((n_layers, N_EXPERTS), np.int32)
        self._checked = set()
        self.dirty = set()
        self.bump_weight = 1
        self.retain_host = True
        self.ram_admission_enabled = True
        self.copy_stream = None
        self.copy_event = None
        self.coop = None
        self._coops = {}
        self._io_pool = None
        self._pending = {}
        self._prefetched = set()
        self._active_keys = set()
        self._ram_insertions = 0
        self._host_baseline_bytes = 0
        self.cpu_backend = None
        self.cpu_private_host_budget_bytes = 0
        self.execution_precision = "gpu_exl3_fp16"
        self.adaptive_gpu_enabled = False
        self.adaptive_migrate_host_backing = False
        self._adaptive_backing_failed = False
        self._gpu_reseed_failed = False
        self._ram_reseed_failed = False
        self.adaptive_profile_scope = None
        self._adaptive_counts = None
        self._adaptive_last_boundary = 0
        self.stats.update(dict(cpu_experts=0, cpu_layers=0, cpu_weight_bytes=0,
                               cpu_fallback=0, cpu_compute_s=0., cpu_wait_s=0.,
                               cpu_d2h_bytes=0, cpu_result_h2d_bytes=0))
        self.stats.update(dict(host_guard_calls=0, host_guard_s=0., host_guard_scan_s=0.,
                               host_guard_reclaims=0, host_guard_evictions=0, host_guard_evicted_bytes=0,
                               host_cached_bytes_seen_peak=0, host_empty_cache_calls=0, host_empty_cache_s=0.,
                               host_trim_calls=0, host_trim_reclaims=0, host_trim_s=0., io_submit=0.))
        self.stats.update(dict(adaptive_gpu_rounds=0, adaptive_gpu_candidates=0,
                               adaptive_gpu_swaps=0, adaptive_gpu_seconds=0.,
                               adaptive_gpu_executed_layer_rows=0, adaptive_gpu_executed_expert_uses=0,
                               backing_swaps=0, backing_d2h_bytes=0, backing_s=0.,
                               backing_skips=0, backing_failures=0))
        self.stats.update({name: 0 for name in DIRECT_READ_STATS})
        self.stats.update(dict(moe_coop512_calls=0, moe_coop512_rows=0, moe_coop512_routed_slots=0,
                               moe_coop512_fallback_calls=0, moe_coop512_fallback_rows=0))
        if self._coop2048_requested:
            self.stats.update(dict(moe_coop2048_calls=0, moe_coop2048_rows=0, moe_coop2048_routed_slots=0,
                                   moe_coop2048_fallback_calls=0, moe_coop2048_fallback_rows=0))
        if self._coop3072_requested:
            self.stats.update(dict(moe_coop3072_calls=0, moe_coop3072_rows=0, moe_coop3072_routed_slots=0,
                                   moe_coop3072_fallback_calls=0, moe_coop3072_fallback_rows=0))
        self.stats.update({"next_chunk_" + name: 0 for name in CHUNK_PREFETCH_COUNTERS})
        if self._demand_h2d_pipeline is not None:
            self.stats.update({"demand_h2d_" + name: 0 for name in DEMAND_H2D_COUNTERS})
        if self._layer_promotion is not None:
            self.stats.update({"layer_promotion_" + name: 0 for name in LAYER_PROMOTION_COUNTERS})
        self.host_ptrs = torch.zeros(n_layers, PTR_ROWS, N_EXPERTS, dtype=torch.long, pin_memory=True)
        self.packed = torch.zeros(n_layers, PTR_ROWS, N_EXPERTS, dtype=torch.long, device="cuda")
        if self._fast_pointer_requested:
            try:
                self._fast_pointer_table = FastPointerTable(self.host_ptrs, n_layers, lambda: torch.cuda.Event())
                if self._route_snapshot_runtime is not None:
                    self._route_snapshot_runtime.validate_dependency()
            except BaseException:
                os.close(self.pack)
                self.pack = None
                raise
        if self._expert_direct_requested:
            try:
                self._direct_reader = DirectExpertReader(pack_path, self.pack, self.slots,
                                                         pack_version=_version)
            except BaseException:
                os.close(self.pack)
                self.pack = None
                raise
            status = self._direct_reader.status()
            if not status["active"]:
                print("expert direct IO inactive: " + str(status["disabled_reason"]), flush=True)

    def close(self):
        if getattr(self, "_layer_promotion", None) is not None:
            self._retire_layer_promotion_resources()
        if getattr(self, "_native3072_runtime", None) is not None:
            # This ON-only chain must retain GPU/source/pack owners when ANY
            # drain fails, including pipeline cleanup before the native drain.
            self._close_native3072_resources()
            return
        try:
            try:
                if self.cpu_backend is not None:
                    self.cpu_backend.close()
                    self.cpu_backend = None
            finally:
                self.finish_next_chunk_prefetch()
        finally:
            if self._io_pool is not None:
                # Both descriptors remain valid until every worker completes.
                self._io_pool.shutdown(wait=True, cancel_futures=True)
                self._io_pool = None
            try:
                # A prior interrupted join may have left an owned live future.
                # Shutdown has now retired workers; account it before closing.
                self.finish_next_chunk_prefetch()
            finally:
                try:
                    pipeline = getattr(self, "_demand_h2d_pipeline", None)
                    if pipeline is not None:
                        pipeline.close()
                finally:
                    native2048 = getattr(self, "_native2048_runtime", None)
                    if native2048 is not None:
                        # Failure keeps table/slot/runner owners and pack alive;
                        # a later close may retry the SAME recorded CUDA stream.
                        try:
                            native2048.close()
                        except BaseException:
                            self._coop2048_failed = True
                            raise
                    self._pending.clear()
                    pointers = getattr(self, "_fast_pointer_table", None)
                    reader = getattr(self, "_direct_reader", None)
                    if native2048 is not None:
                        # Its owner-stream drain does not cover an outstanding
                        # copy-stream table DMA. Retire that source before any
                        # descriptors are released; retain all owners on error.
                        try:
                            if pointers is not None:
                                pointers.close()
                        except BaseException:
                            self._coop2048_failed = True
                            native2048.poison()
                            raise
                        pointers = None
                    try:
                        if pointers is not None:
                            pointers.close()
                    finally:
                        try:
                            if reader is not None:
                                reader.close()
                        finally:
                            if self.pack is not None:
                                os.close(self.pack)
                                self.pack = None
        self._demand_h2d_closed = True

    def _close_native3072_resources(self):
        runtime = self._native3072_runtime
        try:
            if self.cpu_backend is not None:
                raise RuntimeError("coop3072 cannot close a cache with CPU experts")
            if self._io_pool is not None:
                self._io_pool.shutdown(wait=True, cancel_futures=True)
                self._io_pool = None
            self.finish_next_chunk_prefetch()
            pipeline = getattr(self, "_demand_h2d_pipeline", None)
            if pipeline is not None:
                pipeline.close()
            runtime.close()
            pointers = getattr(self, "_fast_pointer_table", None)
            if pointers is not None:
                pointers.close()
            # Pending worker sources may be discarded only after both compute
            # and pointer-table copy streams have actually drained.
            self._pending.clear()
            reader = getattr(self, "_direct_reader", None)
            if reader is not None:
                reader.close()
            if self.pack is not None:
                os.close(self.pack)
                self.pack = None
        except BaseException:
            self._coop3072_failed = True
            runtime.poison()
            raise
        self._demand_h2d_closed = True

    def start_layer_promotion(self, *, protected_frozen_keys=frozenset()):
        runtime = getattr(self, "_layer_promotion", None)
        if runtime is None:
            return
        self._check_placement_health()
        # A one-time boundary drains prior prefill/copy streams before the
        # decode-only policy binds its compute stream. No steady-state sync.
        started = time.perf_counter()
        try:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Start layer promotion outside CUDA graph capture")
            torch.cuda.synchronize(self.packed.device)
            runtime.start(torch.cuda.current_stream(self.packed.device), lambda: torch.cuda.Event(),
                          lambda: torch.cuda.current_stream(self.packed.device),
                          protected_frozen_keys=protected_frozen_keys)
            runtime._inc("activation_s", time.perf_counter() - started)
        except BaseException:
            runtime.poison()
            raise

    def begin_layer_promotion(self, layer):
        runtime = getattr(self, "_layer_promotion", None)
        if runtime is not None:
            runtime.begin_layer(layer)

    def finish_layer_promotion(self, layer):
        runtime = getattr(self, "_layer_promotion", None)
        return 0 if runtime is None else runtime.finish_layer(layer)

    def abort_layer_promotion(self):
        runtime = getattr(self, "_layer_promotion", None)
        if runtime is not None and runtime.active:
            runtime.poison()

    def layer_promotion_status(self):
        runtime = getattr(self, "_layer_promotion", None)
        return None if runtime is None else runtime.status()

    def verify_layer_promotion_metadata(self):
        runtime = getattr(self, "_layer_promotion", None)
        if runtime is not None:
            runtime.verify_metadata()

    def _retire_layer_promotion_resources(self):
        runtime = self._layer_promotion
        try:
            # This ON-only pre-retirement runs before legacy finally blocks can
            # close descriptors. Any failed drain retains pack/table/slot owners.
            if self._io_pool is not None:
                self._io_pool.shutdown(wait=True, cancel_futures=True)
                self._io_pool = None
            self.finish_next_chunk_prefetch()
            pipeline = getattr(self, "_demand_h2d_pipeline", None)
            if pipeline is not None:
                pipeline.close()
            for name in ("_native3072_runtime", "_native2048_runtime"):
                native = getattr(self, name, None)
                if native is not None:
                    native.close()
            if not runtime.started:
                # An early prefill/activation failure has no recorded decode
                # stream yet. Retain all owners unless every device stream drains.
                torch.cuda.synchronize(self.packed.device)
            runtime.close()
            if self.copy_stream is not None:
                self.copy_stream.synchronize()
            pointers = getattr(self, "_fast_pointer_table", None)
            if pointers is not None:
                pointers.close()
        except BaseException:
            runtime.poison()
            raise

    def initialize_native3072(self, device, *, volatile_experts):
        self._check_placement_health()
        initialize_coop3072(self, device, volatile_experts=volatile_experts)

    def native3072_status(self):
        return coop3072_status(self)

    def validate_native3072_placement(self):
        if getattr(self, "_coop3072_requested", False):
            if type(self.stage_limit) is not int or self.stage_limit != 384:
                raise RuntimeError("coop3072 actual volatile capacity must remain exactly384")
            runtime = getattr(self, "_native3072_runtime", None)
            if runtime is None:
                raise RuntimeError("coop3072 must be initialized before expert GPU budgeting")
            runtime._validate_cache(self)

    def verify_native3072_metadata(self):
        runtime = getattr(self, "_native3072_runtime", None)
        if runtime is not None:
            try:
                runtime._validate_cache(self)
                runtime.verify_metadata()
            except BaseException:
                self._coop3072_failed = True
                runtime.poison()
                raise

    def initialize_native2048(self, device, *, volatile_experts):
        self._check_placement_health()
        initialize_coop2048(self, device, volatile_experts=volatile_experts)

    def native2048_status(self):
        return coop2048_status(self)

    def validate_native2048_placement(self):
        if getattr(self, "_coop2048_requested", False):
            if type(self.stage_limit) is not int or not 6 <= self.stage_limit <= 256:
                raise RuntimeError("coop2048 actual volatile capacity must remain6..256")
            if getattr(self, "_native2048_runtime", None) is None:
                raise RuntimeError("coop2048 must be initialized before expert GPU budgeting")

    def verify_native2048_metadata(self):
        runtime = getattr(self, "_native2048_runtime", None)
        if runtime is not None:
            try:
                runtime.verify_metadata()
            except BaseException:
                self._coop2048_failed = True
                runtime.poison()
                raise

    def direct_io_status(self):
        reader = getattr(self, "_direct_reader", None)
        if reader is not None:
            return reader.status()
        return dict(requested=getattr(self, "_expert_direct_requested", False), active=False,
                    disabled_reason="disabled", scope="_read_pinned_timed_payload_only",
                    policy_alignment=4096, counters_including_startup={name: 0 for name in DIRECT_READ_STATS})

    def pointer_bind_status(self):
        table = getattr(self, "_fast_pointer_table", None)
        return dict(enabled=False, events_created=0) if table is None else table.status()

    def demand_h2d_pipeline_status(self):
        pipeline = getattr(self, "_demand_h2d_pipeline", None)
        if pipeline is not None:
            value=pipeline.status(self.stats)
            if verification_max_rows()==1024:
                value.update(scope='actual_MoE_staging_transactions_including_prefill_and_diagnostics',
                    joint_layer_native_kernel_calls=2,
                    joint_layer_scope='One stage transaction forM513..1024; two actual checked3072 kernels')
            return value
        return dict(demand_h2d_configuration(False), failed=False,
                    closed=getattr(self, "_demand_h2d_closed", False), active=False, quarantined_calls=0,
                    counters=dict.fromkeys(DEMAND_H2D_COUNTERS, 0),
                    counters_since_reset=dict.fromkeys(DEMAND_H2D_COUNTERS, 0),
                    scoped_counter_scope="since_latest_cache_reset_stats")

    def demand_h2d_pipeline_configuration(self):
        return demand_h2d_configuration(getattr(self, "_demand_h2d_pipeline", None) is not None)

    def route_snapshot_scope(self, layer, flat, indices, weights, k, inter):
        runtime = getattr(self, "_route_snapshot_runtime", None)
        return nullcontext(None) if runtime is None else runtime.scope(self, layer, flat, indices, weights, k, inter)

    def route_snapshot_configuration(self):
        return route_snapshot_configuration(getattr(self, "_route_snapshot_runtime", None) is not None)

    def route_snapshot_status(self):
        runtime = getattr(self, "_route_snapshot_runtime", None)
        return disabled_status() if runtime is None else runtime.status()

    @property
    def next_chunk_prefetch_active(self):
        state = getattr(self, "_chunk_prefetch", None)
        return state is not None and state.active

    def next_chunk_prefetch_scope(self):
        state = getattr(self, "_chunk_prefetch", None)
        return nullcontext() if state is None else state.scope(self.finish_next_chunk_prefetch)

    def next_chunk_prefetch_configuration(self):
        config = getattr(self, "_chunk_prefetch_config", ChunkPrefetchConfig())
        result = config.as_dict()
        result["minimum_available_bytes"] = max(8 * GIB, getattr(self, "minimum_available_bytes", 0))
        return result

    def next_chunk_prefetch_status(self):
        state = getattr(self, "_chunk_prefetch", None)
        return dict(**self.next_chunk_prefetch_configuration(), active=False if state is None else state.active,
                    pending_records=0 if state is None else len(state.entries),
                    pending_charge_bytes=0 if state is None else state.charge,
                    counters={name: self.stats.get("next_chunk_" + name, 0)
                              for name in CHUNK_PREFETCH_COUNTERS})

    def _chunk_stat(self, name, value=1):
        key = "next_chunk_" + name
        self.stats[key] = self.stats.get(key, 0) + value

    def _prefetch_next_chunk(self, layer, next_ids):
        if next_ids is None or not self.next_chunk_prefetch_active or not self.retain_host:
            return
        state = self._chunk_prefetch
        start = time.perf_counter()
        try:
            self._check_placement_health()
            outstanding = None
            for expert in validate_next_ids(next_ids):
                key = (layer, expert)
                if key in self.frozen or key in self.vram or key in self.ram:
                    self._chunk_stat("skip_resident")
                    continue
                if key in self._pending:
                    self._chunk_stat("skip_pending")
                    continue
                if len(state.entries) >= state.config.max_records:
                    self._chunk_stat("skip_budget")
                    continue
                nbytes = self._fields(*key)[1]
                charge = pinned_allocation_charge(nbytes)
                if state.charge + charge > state.config.max_charge_bytes:
                    self._chunk_stat("skip_budget")
                    continue
                # No speculative reclaim, admission, pointer updates or GPUcopy.
                # Include ALL pending demand/legacy/prefetch allocations, even
                # completed-but-unconsumed ones, in the conservative guard.
                if outstanding is None:
                    outstanding = sum(pinned_allocation_charge(self._fields(*k)[1]) for k in self._pending)
                reserve = max(8 * GIB, self.minimum_available_bytes)
                if mem_available_bytes() < reserve + outstanding + charge:
                    self._chunk_stat("skip_headroom")
                    continue
                if self._io_pool is None:
                    self._io_pool = ThreadPoolExecutor(max_workers=4)
                # Called only AFTER the current chunk's complete demand submit.
                future = self._io_pool.submit(self._read_pinned_timed, *key)
                self._pending[key] = future
                state.entries[key] = PendingOrigin(nbytes, charge)
                state.charge += charge
                outstanding += charge
                self._chunk_stat("submitted_reads")
                self._chunk_stat("submitted_bytes", nbytes)
                self.stats["next_chunk_peak_charge_bytes"] = max(
                    self.stats.get("next_chunk_peak_charge_bytes", 0), state.charge)
        finally:
            self._chunk_stat("submit_s", time.perf_counter() - start)

    def _claim_next_chunk_read(self, key):
        state = getattr(self, "_chunk_prefetch", None)
        entry = None if state is None else state.entries.get(key)
        if entry is not None and not entry.claimed:
            entry.claimed = True
            entry.ready_at_demand = self._pending[key].done()

    def _take_pending_read(self, key):
        state = getattr(self, "_chunk_prefetch", None)
        entry = None if state is None else state.entries.get(key)
        if entry is None:
            return self._pending.pop(key).result()  # Original demand path.
        future = self._pending[key]
        try:
            result = future.result()
            self._chunk_stat("completed_reads")
            self._chunk_stat("completed_bytes", result[0].nbytes)
            if entry.claimed:
                name = "timely" if entry.ready_at_demand else "late"
                self._chunk_stat(name + "_reads")
                self._chunk_stat(name + "_bytes", result[0].nbytes)
            return result
        except BaseException:
            self._chunk_prefetch_failed = True
            raise
        finally:
            # Keep ownership/charge if the WAIT was interrupted while IO runs.
            if future.done():
                self._pending.pop(key, None)
                state.pop(key)

    def finish_next_chunk_prefetch(self, layer=None):
        """Discard unneeded owned reads safely; never admit them to RAM."""
        state = getattr(self, "_chunk_prefetch", None)
        if state is None or not state.entries:
            return
        _ = state.active  # Reject cleanup from a competing decode owner.
        start, failure = time.perf_counter(), None
        for key in list(state.entries):
            if layer is not None and key[0] != layer:
                continue
            entry = state.entries[key]
            future = self._pending.get(key)
            try:
                if future is None:
                    raise RuntimeError("next-chunk future ownership was lost")
                if future.cancel():
                    self._pending.pop(key)
                    state.pop(key)
                    self._chunk_stat("cancelled_reads")
                    self._chunk_stat("cancelled_bytes", entry.nbytes)
                else:
                    pinned, elapsed = self._take_pending_read(key)
                    self._account_read(pinned, elapsed)
                    self._chunk_stat("unused_reads")
                    self._chunk_stat("unused_bytes", pinned.nbytes)
            except BaseException as exc:
                if future is None or future.done():
                    self._pending.pop(key, None)
                    state.pop(key)
                self._chunk_prefetch_failed = True
                if failure is None:
                    failure = exc
        self._chunk_stat("cleanup_s", time.perf_counter() - start)
        if failure is not None:
            raise failure

    def _check_placement_health(self):
        if any(getattr(self, flag, False) for flag in
               ("_adaptive_backing_failed", "_gpu_reseed_failed", "_ram_reseed_failed", "_fast_pointer_failed",
                "_coop512_failed", "_coop2048_failed", "_coop3072_failed", "_chunk_prefetch_failed", "_route_snapshot_failed", "_demand_h2d_failed", "_layer_promotion_failed")):
            raise RuntimeError("expert backing/reseed failed; discard this cache instance")

    def configure_host_allocator_reserve(self, reserve_bytes):
        """Opt-in hypothesis: split existing main-host T into live RAM L=T-R.

        Configure before placement, after the normal shared-budget clamp and
        optional CPU-backend deduction. ranked_sets receives the original shared
        budget and recomputes that deduction once. No allocation or DMA changes.
        """
        if type(reserve_bytes) is not int or reserve_bytes not in (0, 512 * 1024**2, 1024**3):
            raise ValueError("Host allocator reserve must be0,512MiB or1GiB")
        previous = getattr(self, "host_allocator_reserve_bytes", 0)
        if previous:
            if reserve_bytes != previous:
                raise RuntimeError("Host allocator reserve cannot be changed after configuration")
            return
        if reserve_bytes == 0:
            return  # Preserve legacy budgets, statistics and host-query path.
        self._check_placement_health()
        if (not self.retain_host or self.ram or self.vram or self.frozen
                or self._pending or self._active_keys or any(self.pools.values())
                or self.ram_charge_bytes):
            raise RuntimeError("Configure host allocator reserve before expert placement/reads")
        total = self.ram_budget_bytes
        if type(total) is not int or total < reserve_bytes:
            raise ValueError("Host allocator reserve exceeds the actual main-host budget")
        setup_snapshot = self._host_allocator_snapshot()
        self.host_allocator_reserve_bytes = reserve_bytes
        self._host_allocator_total_budget_bytes = total
        self.ram_budget_bytes = total - reserve_bytes
        self._host_allocator_last_trim = None
        self._host_allocator_setup_snapshot = setup_snapshot
        self.stats.update(dict(host_allocator_reserve_samples=0,
                               host_allocator_reserve_observation_failures=0,
                               host_allocator_reserve_peak_reserved_bytes=0,
                               host_allocator_reserve_peak_active_bytes=0,
                               host_allocator_reserve_peak_cached_bytes=0))

    def _host_allocator_snapshot(self):
        """Observed Torch allocator bytes; no allocation, reclaim or CUDA sync."""
        try:
            stats = torch.cuda.memory.host_memory_stats()
        except (AttributeError, RuntimeError):
            stats = {}
        reserved = stats.get("allocated_bytes.current")
        active = stats.get("active_bytes.current")
        reserved = int(reserved) if reserved is not None else None
        active = int(active) if active is not None else None
        return dict(reserved_bytes=reserved, active_bytes=active,
                    cached_bytes=None if reserved is None or active is None else max(0, reserved-active),
                    ram_charge_bytes=self.ram_charge_bytes,
                    pending_requested_charge_bytes=sum(pinned_allocation_charge(self._fields(*key)[1])
                                                       for key in self._pending))

    def _host_allocator_observe_trim(self):
        sample = self._host_allocator_snapshot()
        self.stats["host_allocator_reserve_samples"] += 1
        for name in ("reserved", "active", "cached"):
            value = sample[name + "_bytes"]
            if value is not None:
                key = "host_allocator_reserve_peak_" + name + "_bytes"
                self.stats[key] = max(self.stats[key], value)
        return sample

    def host_allocator_reserve_status(self):
        """ON-only observation; closing the cache does not release RAM tensors."""
        reserve = getattr(self, "host_allocator_reserve_bytes", 0)
        if not reserve:
            return None  # No extra host queries or report field for default0.
        private = getattr(self, "cpu_private_host_budget_bytes", 0)
        total = self._host_allocator_total_budget_bytes
        if self.ram_budget_bytes + reserve != total:
            raise RuntimeError("Host allocator total/live/reserve accounting changed")
        last = self._host_allocator_last_trim
        return dict(protocol="shared_host_budget_allocator_reserve_v1", hypothesis=True,
                    measured_speed_gain=False, total_main_host_budget_bytes=total,
                    live_ram_budget_bytes=self.ram_budget_bytes, reserve_bytes=reserve,
                    cpu_private_host_budget_bytes=private, baseline_bytes=self._host_baseline_bytes,
                    trim_slack_bytes=256 * 1024**2,
                    trim_limit_bytes=total + self._host_baseline_bytes + private + 256 * 1024**2,
                    minimum_available_bytes=self.minimum_available_bytes,
                    highwater_margin_bytes=self.host_reclaim_margin_bytes,
                    new_pools=0, new_streams=0, manual_storage_overwrite=False,
                    reserve_is_not_extra_memory=True, reserve_is_not_guaranteed_idle_bytes=True,
                    counter_scope="trim observations since latest cache.reset_stats",
                    counters={key: value for key, value in self.stats.items()
                              if key.startswith("host_allocator_reserve_") or key.startswith("host_trim_")
                              or key.startswith("host_empty_cache_") or key.startswith("host_guard_")},
                    setup_snapshot=dict(self._host_allocator_setup_snapshot),
                    last_trim=None if last is None else {"before":dict(last["before"]),
                        "after":None if last["after"] is None else dict(last["after"]),
                        "reclaim_attempted":last["reclaim_attempted"]},
                    current_snapshot=self._host_allocator_snapshot(),
                    cache_descriptor_closed=getattr(self, "pack", None) is None,
                    RAM_owners_retained=bool(self.ram))

    def attach_cpu_backend(self, backend, *, allow_approx=False, speculative=False):
        """Opt in to decode-only CPU cold experts and reserve their private RAM.

        Call after configuring ram_budget_bytes (or before ranked_sets, whose
        RAM argument is the total shared budget). Backend tickets borrow native
        records; only private fixed buffers and an explicit repack allowance are
        deducted. AVX activation INT8 requires the caller's visible consent.
        """
        if speculative:
            raise ValueError("CPU experts are disabled for speculative target verification")
        if backend is not None and getattr(self, "_demand_h2d_pipeline", None) is not None:
            raise ValueError("demand H2D pipeline cannot be combined with CPU experts")
        if backend is not None and getattr(self, "_layer_promotion", None) is not None:
            raise ValueError("Layer promotion cannot be combined with CPU experts")
        if backend is not None and getattr(self, "adaptive_gpu_enabled", False):
            raise ValueError("adaptive GPU placement cannot be combined with CPU experts")
        if backend is not None and not allow_approx:
            raise ValueError("CPU cold experts require allow_approx=True; their arithmetic differs from GPU")
        old = getattr(self, "cpu_backend", None)
        if old is not None and old is not backend:
            raise ValueError("close the existing CPU backend before attaching another")
        if old is backend:
            return
        charge = int(backend.total_private_host_budget_bytes)
        if charge > self.ram_budget_bytes:
            raise ValueError("CPU backend private buffers exceed the shared host budget")
        self.cpu_backend = backend
        self.cpu_private_host_budget_bytes = charge
        self.execution_precision = str(backend.precision)
        self.ram_budget_bytes -= charge
        if getattr(self, "host_allocator_reserve_bytes", 0):
            self._host_allocator_total_budget_bytes = self.ram_budget_bytes + self.host_allocator_reserve_bytes
        for key in list(self.ram):
            if self.ram_charge_bytes <= self.ram_budget_bytes:
                break
            if key not in self._active_keys:
                self._evict_ram(key)

    def enable_adaptive_gpu(self, interval=8, max_swaps=16, minimum_count=3, *, migrate_host_backing=False):
        """Start an independent post-prefill target-route profile; default off.

        Call rebalance_gpu(completed_tokens) after an entire decode/speculative
        cycle is committed and synchronized, never between verification chunks.
        All executed target rows count, including rejected proposals and replay.
        The caller's model batch must be one; flat MoE rows do not encode B.
        Default promotions retain RAM backing and cause no weight DMA. Optional
        migration reuses that pinned buffer for the demoted GPU expert via D2H.
        """
        if getattr(self, "_layer_promotion", None) is not None:
            raise ValueError("Layer promotion excludes continuous adaptive GPU placement")
        if getattr(self, "cpu_backend", None) is not None:
            raise ValueError("adaptive GPU placement cannot be combined with CPU experts")
        if getattr(self, "_demand_h2d_pipeline", None) is not None:
            raise ValueError("demand H2D pipeline cannot be combined with adaptive placement")
        self._configure_adaptive_gpu(interval, max_swaps, minimum_count,
                                     migrate_host_backing=migrate_host_backing)

    def _configure_adaptive_gpu(self, interval, max_swaps, minimum_count, *, migrate_host_backing):
        """Initialize placement bookkeeping after the caller's compatibility checks.

        Only public continuous-adaptive setup and the synchronized one-shot
        prompt reseed transaction may call this private configuration helper.
        It does not detach, disable, or authorize a demand pipeline call.
        """
        self._check_placement_health()
        if not isinstance(migrate_host_backing, bool):
            raise TypeError("migrate_host_backing must be bool")
        if interval < 1 or max_swaps < 1 or minimum_count < 1:
            raise ValueError("adaptive GPU settings must be positive")
        self.adaptive_gpu_enabled = True
        self.adaptive_migrate_host_backing = migrate_host_backing
        self.adaptive_profile_scope = "executed_target_routes"
        self._adaptive_interval = int(interval)
        self._adaptive_max_swaps = int(max_swaps)
        self._adaptive_minimum_count = int(minimum_count)
        self._adaptive_counts = np.zeros((self.n_layers, N_EXPERTS), dtype=np.int64)
        self._adaptive_last_boundary = 0
        for name in ("adaptive_gpu_rounds", "adaptive_gpu_candidates", "adaptive_gpu_swaps", "adaptive_gpu_seconds",
                     "adaptive_gpu_executed_layer_rows", "adaptive_gpu_executed_expert_uses"):
            self.stats[name] = 0
        for name in ("backing_swaps", "backing_d2h_bytes", "backing_s", "backing_skips", "backing_failures"):
            self.stats[name] = 0

    def _exchange_adaptive_backing(self, hot_key, cold_key, cold_slot, synchronized_devices):
        """Reuse hot's pinned buffer, publishing cold only after blocking D2H.

        All cache mutation is on the inference thread. CPU tickets are excluded;
        per-device synchronization retires every prior GPU source borrow. IO
        workers allocate independent buffers and neither pair key may be pending.
        """
        start = time.perf_counter()
        try:
            source = self.ram.get(hot_key)
            pending = getattr(self, "_pending", {})
            if (source is None or cold_key in self.ram or self._active_keys
                    or getattr(self, "cpu_backend", None) is not None
                    or hot_key in pending or cold_key in pending):
                self.stats["backing_skips"] += 1
                return False
            hot_slot = self.vram.get(hot_key)
            fields = self._fields(*cold_key)
            nbytes = int(fields[1])
            host = source.host
            if (hot_slot is None or self._sig(*hot_key) != self._sig(*cold_key)
                    or hot_slot.storage.numel() != cold_slot.storage.numel()
                    or int(self._fields(*hot_key)[1]) != nbytes or source.nbytes != nbytes
                    or host.dtype != torch.uint8 or host.device.type != "cpu" or host.ndim != 1
                    or not host.is_contiguous() or not host.is_pinned() or host.numel() != nbytes
                    or cold_slot.storage.dtype != torch.uint8 or cold_slot.storage.ndim != 1
                    or not cold_slot.storage.is_contiguous() or cold_slot.storage.numel() < nbytes):
                self.stats["backing_skips"] += 1
                return False
            # All pack components are 16-bit. Preflight offsets to prevent
            # _view_pinned's misalignment fallback from cloning a projection.
            for pi in range(3):
                base = 4+pi*11
                sizes, offsets = fields[base+4:base+7], fields[base+8:base+11]
                if any(offset < 0 or size < 0 or offset+size > nbytes
                       or (host.data_ptr()+offset) % 2 or size % 2
                       for size, offset in zip(sizes, offsets)):
                    self.stats["backing_skips"] += 1
                    return False
            backed = self._pinned_views(host, fields)  # Validate before mutation.
            backed.allocation_bytes = source.allocation_bytes
            allocation = host.untyped_storage().data_ptr()
            if (backed.host is not host or backed.nbytes != nbytes
                    or any(part.untyped_storage().data_ptr() != allocation
                           for projection in backed.parts for part in projection)):
                raise RuntimeError("adaptive backing views must borrow the original pinned allocation")
            device = cold_slot.storage.device
            if device not in synchronized_devices:
                # The host allocator protects deallocation, not an in-place
                # overwrite while a different stream still reads this buffer.
                torch.cuda.synchronize(device)
                synchronized_devices.add(device)
            # Unpublish before mutable IO: a partial/failed copy must never
            # leave either key pointing at a corrupted CPU weight record.
            del self.ram[hot_key]
            self.ram_nbytes -= source.nbytes
            self.ram_charge_bytes -= source.allocation_bytes
            self._prefetched.discard(hot_key)
            try:
                host.copy_(cold_slot.storage[:nbytes], non_blocking=False)
            except BaseException:
                self._adaptive_backing_failed = True
                self.stats["backing_failures"] += 1
                raise
            self.ram[cold_key] = backed
            self.ram_nbytes += backed.nbytes
            self.ram_charge_bytes += backed.allocation_bytes
            self._prefetched.discard(cold_key)
            self.stats["backing_swaps"] += 1
            self.stats["backing_d2h_bytes"] += nbytes
            return True
        finally:
            self.stats["backing_s"] += time.perf_counter()-start

    def _check_prompt_gpu_reseed_quiescent(self):
        if self._active_keys or self._pending:
            raise ValueError("prompt GPU reseed requires no active keys or pending IO")
        pipeline = getattr(self, "_demand_h2d_pipeline", None)
        if pipeline is not None:
            pipeline.validate(self)  # Owner, healthy, open, idle, and compatible.
            if pipeline.quarantined:
                raise RuntimeError("prompt GPU reseed requires no quarantined demand H2D sources")

    def reseed_gpu_from_prompt(self, prompt_counts, *, max_loads=256, max_swaps=256,
                               minimum_count=1, migrate_host_backing=True, protected_frozen_keys=None):
        """Opt-in cold-start placement using ONLY fresh real-prompt routing.

        The caller captures the count delta around prefill, drains pending IO,
        and calls before parity/prefix probes or any generated-token routing.
        No new GPU slot is allocated. RAM admission suppression is respected;
        missing host backing makes migration skip, rather than forcing admit.
        """
        self._check_placement_health()
        if self.cpu_backend is not None or self.adaptive_gpu_enabled:
            raise ValueError("prompt GPU reseed requires GPU-only mode with online adaptive disabled")
        self._check_prompt_gpu_reseed_quiescent()
        if self.stage_limit < 1:
            raise ValueError("prompt GPU reseed requires an existing volatile capacity")
        if (not isinstance(prompt_counts, np.ndarray) or prompt_counts.shape != (self.n_layers, N_EXPERTS)
                or prompt_counts.dtype.kind != "i" or np.any(prompt_counts < 0)):
            raise ValueError("prompt counts must be a fresh nonnegative signed integer [layers,384] array")
        if (isinstance(max_loads, bool) or not isinstance(max_loads, (int, np.integer)) or max_loads < 0
                or isinstance(max_swaps, bool) or not isinstance(max_swaps, (int, np.integer)) or max_swaps < 1
                or isinstance(minimum_count, bool) or not isinstance(minimum_count, (int, np.integer)) or minimum_count < 1
                or not isinstance(migrate_host_backing, bool)):
            raise ValueError("invalid prompt GPU reseed limits")
        protected = frozenset()
        protected_owners = {}
        if protected_frozen_keys is not None:
            if type(protected_frozen_keys) is not frozenset:
                raise ValueError("protected frozen keys must be an immutable frozenset")
            protected_signatures = {self._sig(*key) for key,slot in self.vram.items()
                                    if slot.storage.numel() == self._volatile_slot_bytes}
            protected_signatures.update(sig for sig,pool in self.pools.items()
                if any(slot.storage.numel() == self._volatile_slot_bytes for slot in pool))
            for key in protected_frozen_keys:
                if (type(key) is not tuple or len(key) != 2 or any(type(n) is not int for n in key)
                        or not 0 <= key[0] < self.n_layers or not 0 <= key[1] < N_EXPERTS
                        or key not in self.frozen or int(self._fields(*key)[2]) != 3):
                    raise ValueError("protected keys must be existing compatible K3 frozen experts")
                slot = self.frozen[key]
                if (slot.storage.numel() != self._volatile_slot_bytes
                        or self._sig(*key) not in protected_signatures):
                    raise ValueError("protected frozen slot must match actual volatile capacity")
                protected_owners[key] = (slot, slot.storage.data_ptr(), slot.storage.numel())
            protected = protected_frozen_keys
        def check_protected_owners():
            for key,(slot,pointer,capacity) in protected_owners.items():
                if (self.frozen.get(key) is not slot or slot.storage.data_ptr() != pointer
                        or slot.storage.numel() != capacity):
                    raise RuntimeError("protected frozen owner changed during prompt placement")
        counts = np.array(prompt_counts, dtype="<i8", order="C", copy=True)
        def slots():
            owned = list(self.frozen.values()) + list(self.vram.values())
            owned += [slot for pool in self.pools.values() for slot in pool]
            if len({id(slot) for slot in owned}) != len(owned):
                raise RuntimeError("GPU slot has duplicate ownership")
            return {id(slot): (slot.storage.data_ptr(), slot.storage.numel()) for slot in owned}
        before_slots = slots()
        if any(slot.storage.numel() < self._volatile_slot_bytes
               for slot in list(self.vram.values()) + [s for p in self.pools.values() for s in p]):
            raise ValueError("prompt GPU reseed requires uniform existing volatile slot capacity")
        ranked = sorted(((int(counts[layer, expert]), (layer, expert))
                         for layer in range(self.n_layers) for expert in range(N_EXPERTS)
                         if counts[layer, expert] >= minimum_count), key=lambda item: (-item[0], item[1]))
        report = dict(kind="prompt_gpu_residency_reseed", profile_scope="prompt_prefill_only",
                      profile_sha256=hashlib.sha256(counts.tobytes(order="C")).hexdigest(),
                      profile_hash_encoding="raw_C_contiguous_little_endian_int64_payload_not_npy",
                      profile_shape=list(counts.shape), profile_expert_uses=sum(map(int, counts.flat)),
                      max_loads=int(max_loads), max_swaps=int(max_swaps), minimum_count=int(minimum_count),
                      migrate_host_backing=migrate_host_backing,
                      ram_admission_enabled=bool(getattr(self, "ram_admission_enabled", True)),
                      candidates=len(ranked), loads=0, swaps=0, resident_swaps=0,
                      loaded_without_ram_backing=0,
                      skipped_k2=0, skipped_no_victim=0, skipped_no_existing_slot=0,
                      load_limit_reached=False, swap_limit_reached=False)
        start = time.perf_counter()
        adaptive_names = ("adaptive_gpu_enabled", "adaptive_migrate_host_backing", "adaptive_profile_scope",
                          "_adaptive_counts", "_adaptive_last_boundary", "_adaptive_interval",
                          "_adaptive_max_swaps", "_adaptive_minimum_count")
        saved_adaptive = {name: (hasattr(self, name), getattr(self, name, None)) for name in adaptive_names}
        saved_protected = (hasattr(self, "_prompt_protected_frozen_keys"),
                           getattr(self, "_prompt_protected_frozen_keys", None))
        try:
            torch.cuda.synchronize()
            self._check_prompt_gpu_reseed_quiescent()
            check_protected_owners()
            if protected_frozen_keys is not None:
                self._prompt_protected_frozen_keys = protected
            self._configure_adaptive_gpu(interval=1, max_swaps=int(max_swaps), minimum_count=int(minimum_count),
                                         migrate_host_backing=migrate_host_backing)
            self._adaptive_counts[:] = counts
            self.adaptive_profile_scope = "prompt_prefill_only"
            baseline = dict(self.stats)
            boundary = 0
            def rebalance():
                nonlocal boundary
                remaining = int(max_swaps)-report["swaps"]
                if remaining <= 0:
                    return
                boundary += 1
                self._adaptive_max_swaps = remaining
                report["swaps"] += self.rebalance_gpu(boundary)
            rebalance()
            report["resident_swaps"] = report["swaps"]
            for frequency, key in ranked:
                if report["swaps"] >= max_swaps:
                    report["swap_limit_reached"] = True
                    break
                if key in self.frozen:
                    continue
                if int(self._fields(*key)[2]) != 3:
                    report["skipped_k2"] += 1
                    continue
                if key in self.vram:
                    continue  # Initial rebalance already considered this key.
                layer_counts = defaultdict(int)
                for frozen_key in self.frozen:
                    layer_counts[frozen_key[0]] += 1
                signature = self._sig(*key)
                eligible = any(victim not in protected and int(self._fields(*victim)[2]) == 3
                               and self._sig(*victim) == signature
                               and slot.storage.numel() == self._volatile_slot_bytes
                               and int(counts[victim]) < frequency
                               and (victim[0] == key[0] or layer_counts[victim[0]] > 6)
                               for victim, slot in self.frozen.items())
                if not eligible:
                    report["skipped_no_victim"] += 1
                    continue
                if report["loads"] >= max_loads:
                    report["load_limit_reached"] = True
                    break
                if len(self.vram) < self.stage_limit and not any(self.pools.values()):
                    # _checkout would allocate in this state. Never grow slots
                    # during reseed, even if the caller's limit has spare room.
                    report["skipped_no_existing_slot"] += 1
                    continue
                self._install_if_absent(*key, False, True)
                report["loads"] += 1
                report["loaded_without_ram_backing"] += key not in self.ram
                rebalance()
            dirty = list(self.dirty)
            for layer in dirty:
                self._flush(layer)
            torch.cuda.synchronize()
            if self._pending or self._active_keys or slots() != before_slots:
                raise RuntimeError("prompt GPU reseed violated IO/active/slot ownership invariants")
            check_protected_owners()
            if protected_frozen_keys is not None:
                report["protected_frozen"] = dict(keys=[list(k) for k in sorted(protected)],
                    experts=len(protected), slot_capacity_bytes=sum(v[2] for v in protected_owners.values()),
                    actual_original_owners_intact=True,
                    scope="only_this_quiescent_prompt_reseed_no_extra_weight_allocation")
            report["existing_gpu_slots"] = len(before_slots)
            report["gpu_slot_capacity_bytes"] = sum(capacity for _, capacity in before_slots.values())
            report["pointer_table_h2d_bytes"] = len(dirty)*PTR_ROWS*N_EXPERTS*8
            for name in ("h2d_bytes", "io_reads", "io_bytes", "io_wait", "io_alloc", "io_pread", "io_views",
                         "backing_swaps", "backing_d2h_bytes", "backing_s", "backing_skips", "backing_failures",
                         "host_guard_evictions", "host_guard_evicted_bytes"):
                report[name] = self.stats.get(name, 0)-baseline.get(name, 0)
            report["elapsed_s"] = time.perf_counter()-start
            return report
        except BaseException as error:
            self._gpu_reseed_failed = True
            raise RuntimeError("post-prefill GPU reseed failed; discard this cache instance") from error
        finally:
            for name, (present, value) in saved_adaptive.items():
                if present:
                    setattr(self, name, value)
                elif hasattr(self, name):
                    delattr(self, name)
            if protected_frozen_keys is not None:
                if saved_protected[0]:
                    self._prompt_protected_frozen_keys = saved_protected[1]
                elif hasattr(self, "_prompt_protected_frozen_keys"):
                    del self._prompt_protected_frozen_keys

    @staticmethod
    def _ram_reseed_fields_valid(fields):
        if len(fields) != 37 or int(fields[0]) < 0 or int(fields[1]) <= 0:
            raise ValueError("invalid RAM reseed pack descriptor")
        nbytes = int(fields[1])
        for pi in range(3):
            base = 4 + pi * 11
            sizes, offsets = fields[base + 4:base + 7], fields[base + 8:base + 11]
            if any(size < 0 or offset < 0 or offset + size > nbytes or size % 2 or offset % 2
                   for size, offset in zip(sizes, offsets)):
                raise ValueError("RAM reseed parts must be aligned views within the record")

    def _ram_reseed_reuse(self, key, donor_key):
        """The caller has synchronized every source borrow before this write.

        Keep the donor alive through preadv, but never publish partially replaced
        bytes under either logical owner. No new pinned allocation is made.
        """
        start = time.perf_counter()
        source = self.ram[donor_key]
        fields, host = self._fields(*key), source.host
        self._ram_reseed_fields_valid(fields)
        if source.nbytes != int(fields[1]) or host.numel() != int(fields[1]):
            raise ValueError("RAM donor logical capacity does not match payload")
        backed = self._pinned_views(host, fields)
        backed.allocation_bytes = source.allocation_bytes
        allocation = host.untyped_storage().data_ptr()
        if (backed.host is not host or any(part.untyped_storage().data_ptr() != allocation
                                         for projection in backed.parts for part in projection)):
            raise ValueError("RAM reseed views must borrow the original pinned allocation")
        target = memoryview(host.numpy())
        self._evict_ram(donor_key)
        done = 0
        io_start = time.perf_counter()
        while done < backed.nbytes:
            got = os.preadv(self.pack, [target[done:]], int(fields[0]) + done)
            if not 0 < got <= backed.nbytes - done:
                raise RuntimeError(f"short expert read {key[0]}:{key[1]}")
            done += got
        finished = time.perf_counter()
        backed.pread_s = finished - io_start
        backed.views_s = io_start - start
        # Ordinary admission remains disabled if the phase disabled it. This
        # explicit operation publishes only the planned, completely read key.
        self.ram[key] = backed
        self.ram_nbytes += backed.nbytes
        self.ram_charge_bytes += backed.allocation_bytes
        self._prefetched.discard(key)
        self._account_read(backed, finished - start)

    def reseed_ram_from_prompt(self, prompt_counts, *, max_loads=256, minimum_count=1,
                               objective="bytes"):
        """Explicit bounded RAM placement after GPU reseed and before probes.

        Default callers never invoke this API. Fresh complete-prompt sampling is
        a caller contract; no histogram can prove absence of future-token data.
        Retained selected records keep their original objects. Safety headroom
        may still evict them; the returned achieved set reports that shortfall.
        """
        self._check_placement_health()
        if self.cpu_backend is not None or self.adaptive_gpu_enabled or not self.retain_host:
            raise ValueError("prompt RAM reseed requires retained host records, GPU-only and adaptive off")
        if self._active_keys or self._pending:
            raise ValueError("prompt RAM reseed requires no active keys or pending IO")
        validate_prompt_counts(prompt_counts, self.n_layers)
        if (isinstance(max_loads, bool) or not isinstance(max_loads, (int, np.integer)) or max_loads < 0):
            raise ValueError("max_loads must be a nonnegative integer")
        counts = np.array(prompt_counts, dtype="<i8", order="C", copy=True)
        keys = [(l, e) for l in range(self.n_layers) for e in range(N_EXPERTS)]
        payloads = {key: int(self._fields(*key)[1]) for key in keys}
        charges = {key: pinned_allocation_charge(payloads[key]) for key in keys}
        before_ids, allocations = {}, set()
        for key, source in self.ram.items():
            host = source.host
            if (key not in payloads or source.nbytes != payloads[key]
                    or source.allocation_bytes < source.nbytes
                    or host.dtype != torch.uint8 or host.device.type != "cpu" or host.ndim != 1
                    or not host.is_contiguous() or not host.is_pinned() or host.numel() != source.nbytes):
                raise ValueError("invalid resident pinned RAM record")
            allocation = host.untyped_storage().data_ptr()
            if allocation in allocations:
                raise ValueError("RAM records must have unique allocation ownership")
            allocations.add(allocation)
            charges[key] = int(source.allocation_bytes)
            before_ids[key] = id(source)
        # Do not retain these loop locals across the headroom guard: an evicted
        # record must be physically releasable. The RAM map owns kept objects.
        source = host = None
        if (self.ram_nbytes != sum(obj.nbytes for obj in self.ram.values())
                or self.ram_charge_bytes != sum(obj.allocation_bytes for obj in self.ram.values())):
            raise ValueError("RAM accounting disagrees with its resident records")
        plan = plan_prompt_ram(counts, record_bytes=payloads, allocation_charges=charges,
                               resident_keys=self.ram, frozen_keys=self.frozen,
                               budget_bytes=self.ram_budget_bytes, minimum_count=minimum_count,
                               objective=objective)
        for candidate in plan.selected:
            self._ram_reseed_fields_valid(self._fields(*candidate.key))
        desired = set(plan.keys)

        def gpu_inventory():
            result = [("frozen", key, id(slot), slot.storage.data_ptr(), slot.storage.numel(), str(slot.storage.device))
                      for key, slot in self.frozen.items()]
            result += [("volatile", key, id(slot), slot.storage.data_ptr(), slot.storage.numel(), str(slot.storage.device))
                       for key, slot in self.vram.items()]
            result += [("pool", (repr(sig), i), id(slot), slot.storage.data_ptr(), slot.storage.numel(), str(slot.storage.device))
                       for sig, pool in self.pools.items() for i, slot in enumerate(pool)]
            return tuple(result)

        inventory = gpu_inventory()
        devices = {entry[-1] for entry in inventory if entry[-1].startswith("cuda")}
        def synchronize():
            if devices:
                for device in sorted(devices):
                    torch.cuda.synchronize(torch.device(device))
            else:  # Also allows CPU byte-buffer fixtures to mock the boundary.
                torch.cuda.synchronize()

        admission = self.ram_admission_enabled
        baseline = dict(self.stats)
        report = dict(kind="prompt_ram_residency_reseed", profile_scope="prompt_prefill_only",
                      profile_sha256=hashlib.sha256(counts.tobytes(order="C")).hexdigest(),
                      profile_hash_encoding="raw_C_contiguous_little_endian_int64_payload_not_npy",
                      profile_shape=list(counts.shape), profile_expert_uses=sum(map(int, counts.flat)),
                      objective=objective, density="frequency*payload_bytes/allocation_charge" if objective == "bytes"
                      else "frequency/allocation_charge", selection="deterministic_density_greedy_not_exact_knapsack",
                      max_loads=int(max_loads), minimum_count=int(minimum_count),
                      ram_admission_enabled=bool(admission), ram_budget_bytes=self.ram_budget_bytes,
                      planned_records=len(plan.selected), candidate_records=plan.candidate_count,
                      planned_charge_bytes=plan.predicted_charge_bytes, planned_value=plan.selected_value,
                      eligible_value=plan.eligible_value, loads=0, attempted_loads=0, rebound_records=0,
                      allocated_records=0, evicted_records=0, frozen_backing_evicted=0,
                      headroom_evicted_records=0, peak_live_charge_including_single_read=self.ram_charge_bytes,
                      max_new_record_charge_bytes=0, stop_reason=None, status="running")
        report.update(host_available_before_bytes=mem_available_bytes(),
                      host_reserved_before_bytes=self._host_reserved_bytes(),
                      host_cached_before_bytes=self._host_cached_bytes())
        report["host_available_min_sampled_bytes"] = report["host_available_before_bytes"]
        self._last_ram_reseed_report = report
        start = time.perf_counter()

        def evict(key):
            self._evict_ram(key)
            report["evicted_records"] += 1

        def make_room(charge):
            for key in sorted((key for key in self.ram if key not in desired), key=lambda k: (int(counts[k]), k)):
                if self.ram_charge_bytes + charge <= self.ram_budget_bytes:
                    break
                evict(key)
            return self.ram_charge_bytes + charge <= self.ram_budget_bytes

        def headroom(nbytes):
            before = set(self.ram)
            try:
                # Eviction alone can leave old pinned blocks reserved. Keep the
                # existing allocator-reservation cap as well as MemAvailable.
                self._trim_host_reservation()
                self._ensure_read_headroom(nbytes)
            finally:
                report["headroom_evicted_records"] += len(before - set(self.ram))
                report["host_available_min_sampled_bytes"] = min(
                    report["host_available_min_sampled_bytes"], mem_available_bytes())

        try:
            synchronize()  # Retire all prior H2D readers before any in-place IO.
            for key in list(self.ram):
                if key in self.frozen:
                    evict(key)
                    report["frozen_backing_evicted"] += 1
            if not make_room(0):
                raise RuntimeError("selected RAM records exceed their actual charged budget")
            for candidate in plan.selected:
                key = candidate.key
                if key in self.ram:
                    continue
                if report["attempted_loads"] >= max_loads:
                    report["stop_reason"] = "max_loads"
                    break
                try:
                    headroom(0)  # In-place writes allocate no new pinned record.
                except MemoryError:
                    report["stop_reason"] = "host_headroom"
                    break
                donor = next((old for old in sorted(self.ram, key=lambda k: (int(counts[k]), k))
                              if old not in desired and self.ram[old].nbytes == candidate.nbytes), None)
                if donor is not None:
                    report["attempted_loads"] += 1
                    self._ram_reseed_reuse(key, donor)
                    report["rebound_records"] += 1
                else:
                    charge = pinned_allocation_charge(candidate.nbytes)
                    if not make_room(charge):
                        report["stop_reason"] = "actual_charge_budget"
                        break
                    try:
                        headroom(candidate.nbytes)
                    except MemoryError:
                        report["stop_reason"] = "host_headroom"
                        break
                    report["attempted_loads"] += 1
                    report["max_new_record_charge_bytes"] = max(report["max_new_record_charge_bytes"], charge)
                    report["peak_live_charge_including_single_read"] = max(
                        report["peak_live_charge_including_single_read"], self.ram_charge_bytes + charge)
                    pinned, elapsed = self._read_pinned_timed(*key)
                    if pinned.nbytes != candidate.nbytes or pinned.allocation_bytes != charge:
                        raise RuntimeError("new RAM record disagrees with its planned allocation charge")
                    self._account_read(pinned, elapsed)
                    self.ram[key] = pinned
                    self.ram_nbytes += pinned.nbytes
                    self.ram_charge_bytes += pinned.allocation_bytes
                    self._prefetched.discard(key)
                    pinned = None  # No executor reference defeats later safety eviction.
                    report["allocated_records"] += 1
                report["loads"] += 1
                if self.ram_charge_bytes > self.ram_budget_bytes:
                    raise RuntimeError("RAM reseed exceeded its actual charged budget")
            try:
                headroom(0)
            except MemoryError:
                report["stop_reason"] = "host_headroom"
            for key in sorted(self.ram, key=lambda k: (int(counts[k]), k)):
                self.ram.move_to_end(key)
            synchronize()
            if (self._pending or self._active_keys or gpu_inventory() != inventory
                    or self.ram_admission_enabled != admission):
                raise RuntimeError("RAM reseed violated phase/GPU ownership invariants")
            if (self.ram_nbytes != sum(obj.nbytes for obj in self.ram.values())
                    or self.ram_charge_bytes != sum(obj.allocation_bytes for obj in self.ram.values())):
                raise RuntimeError("RAM reseed accounting mismatch")
            missing, extra = sorted(desired - set(self.ram)), sorted(set(self.ram) - desired)
            report.update(missing_target_keys=missing, extra_resident_keys=extra,
                          complete_target=not missing, matches_target=not missing and not extra,
                          retained_original_objects=sum(id(obj) == before_ids.get(key) for key, obj in self.ram.items()),
                          achieved_target_value=sum(c.value for c in plan.selected if c.key in self.ram),
                          ram_records=len(self.ram), ram_payload_bytes=self.ram_nbytes,
                          ram_charge_bytes=self.ram_charge_bytes, lru_order="ascending_prompt_frequency_then_key",
                          host_available_after_bytes=mem_available_bytes(),
                          host_reserved_after_bytes=self._host_reserved_bytes(),
                          host_cached_after_bytes=self._host_cached_bytes(),
                          status="complete" if not missing else "partial")
            if missing and report["stop_reason"] is None:
                report["stop_reason"] = "safety_evicted_target_or_inherited_charge"
            for name in ("io_reads", "io_bytes", "io_alloc", "io_pread", "io_views",
                         "host_guard_evictions", "host_guard_evicted_bytes", "host_empty_cache_calls",
                         "host_empty_cache_s", "host_trim_calls", "host_trim_reclaims", "host_trim_s"):
                report[name] = self.stats.get(name, 0) - baseline.get(name, 0)
            return report
        except BaseException as error:
            self._ram_reseed_failed = True
            report.update(status="failed", error_type=type(error).__name__)
            raise RuntimeError("post-prefill RAM reseed failed; discard this cache instance") from error
        finally:
            report["elapsed_s"] = time.perf_counter() - start

    def rebalance_gpu(self, completed_tokens):
        """Exchange GPU tier ownership without changing pointers or weight bytes."""
        self._check_placement_health()
        if not getattr(self, "adaptive_gpu_enabled", False):
            return 0
        boundary = int(completed_tokens) // self._adaptive_interval
        if boundary <= self._adaptive_last_boundary:
            return 0
        self._adaptive_last_boundary = boundary
        start = time.perf_counter()
        self.stats["adaptive_gpu_rounds"] += 1
        swaps = 0
        backing_devices = set()
        try:
            counts = self._adaptive_counts
            frozen_per_layer = defaultdict(int)
            victims = defaultdict(lambda: defaultdict(list))
            for key, slot in self.frozen.items():
                frozen_per_layer[key[0]] += 1
                if (key in self._active_keys or int(self._fields(*key)[2]) != 3
                        or key in getattr(self, "_prompt_protected_frozen_keys", ())):
                    continue
                signature = (self._sig(*key), slot.storage.numel())
                heapq.heappush(victims[signature][key[0]], (int(counts[key]), key))
            candidates = []
            for key, slot in self.vram.items():
                frequency = int(counts[key])
                if (key not in self._active_keys and frequency >= self._adaptive_minimum_count
                        and int(self._fields(*key)[2]) == 3):
                    candidates.append((-frequency, key, slot))
            candidates.sort(key=lambda item: (item[0], item[1]))
            self.stats["adaptive_gpu_candidates"] += len(candidates)
            for negative_frequency, hot_key, hot_slot in candidates:
                signature = (self._sig(*hot_key), hot_slot.storage.numel())
                options = victims.get(signature, {})
                best = None
                for layer, heap in options.items():
                    # A same-layer exchange preserves its quota, even at six.
                    if heap and (layer == hot_key[0] or frozen_per_layer[layer] > 6):
                        frequency, key = heap[0]
                        if frequency < -negative_frequency and (best is None or (frequency, key) < best):
                            best = (frequency, key)
                if best is None:
                    continue
                _, cold_key = best
                heapq.heappop(options[cold_key[0]])
                cold_slot = self.frozen[cold_key]
                if getattr(self, "adaptive_migrate_host_backing", False):
                    self._exchange_adaptive_backing(hot_key, cold_key, cold_slot, backing_devices)
                # Frozen slots originally needed no use tracking. After this
                # demotion the volatile recycler must wait for all earlier
                # compute on this stream before overwriting its storage.
                cold_slot.event.record(torch.cuda.current_stream())
                cold_slot.seen = True
                del self.vram[hot_key]
                del self.frozen[cold_key]
                self.frozen[hot_key] = hot_slot
                self.vram[cold_key] = cold_slot
                self.vram.move_to_end(cold_key, last=False)
                frozen_per_layer[cold_key[0]] -= 1
                frozen_per_layer[hot_key[0]] += 1
                swaps += 1
                if swaps >= self._adaptive_max_swaps:
                    break
            self.stats["adaptive_gpu_swaps"] += swaps
            return swaps
        finally:
            self.stats["adaptive_gpu_seconds"] += time.perf_counter() - start

    def _submit_read(self, key):
        self._check_placement_health()
        if self._io_pool is None:
            self._io_pool = ThreadPoolExecutor(max_workers=4)
        if key not in self._pending:
            self._ensure_read_headroom(self._fields(*key)[1])
            self._pending[key] = self._io_pool.submit(self._read_pinned_timed, *key)
        return self._pending[key]

    def _account_read(self, pinned, elapsed, prefetched=False):
        # io is a SUM of worker durations, not critical-path latency.
        self.stats["io"] += elapsed
        self.stats["io_reads"] += 1
        self.stats["io_bytes"] += pinned.nbytes
        self.stats["io_alloc"] += pinned.alloc_s
        self.stats["io_pread"] += pinned.pread_s
        self.stats["io_views"] += pinned.views_s
        # Worker counters travel with the completed record. Main-thread merge
        # keeps request/reset accounting separate from lifetime startup reads.
        for name, value in getattr(pinned, "direct_io_stats", {}).items():
            self.stats[name] += value
        if prefetched:
            self.stats["prefetch_reads"] += 1
            self.stats["prefetch_bytes"] += pinned.nbytes

    def _ram_insert(self, key, pinned):
        # Phase-scoped profile preservation: temporary staged sources still
        # serve IO/H2D, while the existing RAM membership remains untouched.
        # The independent low-headroom guard may always evict for safety.
        if not getattr(self, "ram_admission_enabled", True):
            return False
        old = self.ram.pop(key, None)
        if old is not None:
            self.ram_nbytes -= old.nbytes
            self.ram_charge_bytes -= old.allocation_bytes
        if not self.retain_host or pinned.allocation_bytes > self.ram_budget_bytes:
            return False
        while self.ram and self.ram_charge_bytes + pinned.allocation_bytes > self.ram_budget_bytes:
            victim = next((key for key in self.ram if key not in self._active_keys), None)
            if victim is None:
                # The caller retains an uncacheable current-token source until
                # its DMA/CPU ticket completes; do not evict another active row.
                return False
            self._evict_ram(victim)
        self.ram[key] = pinned
        self.ram_nbytes += pinned.nbytes
        self.ram_charge_bytes += pinned.allocation_bytes
        self._ram_insertions += 1
        if self._ram_insertions % 64 == 0:
            self._trim_host_reservation()
        return True

    def _evict_ram(self, key=None):
        key = next(iter(self.ram)) if key is None else key
        obj = self.ram.pop(key)
        self.ram_nbytes -= obj.nbytes
        self.ram_charge_bytes -= obj.allocation_bytes
        self._prefetched.discard(key)
        return obj.allocation_bytes

    @staticmethod
    def _host_reserved_bytes():
        try:
            return int(torch.cuda.memory.host_memory_stats().get("allocated_bytes.current", 0))
        except (AttributeError, RuntimeError):
            return 0

    @staticmethod
    def _host_cached_bytes():
        try:
            stats = torch.cuda.memory.host_memory_stats()
            # active includes in-use allocations AND blocks awaiting their DMA
            # events. Only the difference is currently reusable/freeable memory.
            if "active_bytes.current" not in stats:
                return 0
            return max(0, int(stats.get("allocated_bytes.current", 0)) - int(stats["active_bytes.current"]))
        except (AttributeError, RuntimeError):
            return 0

    def _empty_host_cache(self):
        # Unlike torch.cuda.empty_cache(), this releases unused PINNED blocks.
        empty = getattr(torch._C, "_host_emptyCache", None)
        if empty is not None:
            start = time.perf_counter()
            self.stats["host_empty_cache_calls"] += 1
            try:
                empty()
            finally:
                self.stats["host_empty_cache_s"] += time.perf_counter() - start

    def _trim_host_reservation(self):
        start = time.perf_counter()
        self.stats["host_trim_calls"] += 1
        reserve = getattr(self, "host_allocator_reserve_bytes", 0)
        before = None
        reclaim_attempted = False
        failure = None
        try:
            if reserve:
                if self.ram_budget_bytes + reserve != self._host_allocator_total_budget_bytes:
                    raise RuntimeError("Host allocator total/live/reserve accounting changed")
                before = self._host_allocator_observe_trim()
            reserved = (before["reserved_bytes"] if before is not None and before["reserved_bytes"] is not None
                        else self._host_reserved_bytes())
            # R is carved out of the old live budget, not the total trim cap.
            # Reducing this cap with L would leave no allocator reserve at all.
            limit = (self.ram_budget_bytes + reserve + self._host_baseline_bytes +
                     getattr(self, "cpu_private_host_budget_bytes", 0) + 256 * 1024**2)
            if reserved <= limit:
                return
            self.stats["host_trim_reclaims"] += 1
            reclaim_attempted = True
            # Prefer returning already-unused allocator blocks. Their physical
            # reservation does not justify evicting another live expert copy.
            cached = (before["cached_bytes"] if before is not None and before["cached_bytes"] is not None
                      else self._host_cached_bytes())
            over = max(0, reserved - limit - cached)
            released = 0
            for key in list(self.ram):
                if released >= over:
                    break
                if key not in self._active_keys:
                    released += self._evict_ram(key)
            self._empty_host_cache()
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                if before is not None:
                    self._host_allocator_last_trim = dict(before=before, after=None,
                                                         reclaim_attempted=reclaim_attempted)
                    try:
                        self._host_allocator_last_trim["after"] = self._host_allocator_observe_trim()
                    except BaseException as observation_error:
                        self.stats["host_allocator_reserve_observation_failures"] += 1
                        if failure is None:
                            raise
                        # Diagnostic failure must not replace an existing
                        # allocator/reclaim exception (or user interruption).
                        try:
                            failure.add_note("Host allocator after-trim observation failed: " + repr(observation_error))
                        except BaseException:
                            pass
            finally:
                self.stats["host_trim_s"] += time.perf_counter() - start

    def _ensure_read_headroom(self, nbytes):
        start = time.perf_counter()
        self.stats["host_guard_calls"] += 1
        try:
            outstanding = sum(pinned_allocation_charge(self._fields(*key)[1]) for key in self._pending)
            target = self.minimum_available_bytes + outstanding + pinned_allocation_charge(nbytes)
            available = mem_available_bytes()
            if available >= target:
                # Keep cached exact-size blocks for the allocator's normal reuse.
                return
            self.stats["host_guard_reclaims"] += 1
            high = target + max(0, getattr(self, "host_reclaim_margin_bytes", 1024**3))
            cached = self._host_cached_bytes()
            self.stats["host_cached_bytes_seen_peak"] = max(self.stats["host_cached_bytes_seen_peak"], cached)
            required = max(0, high - available - cached)
            scan_start = time.perf_counter()
            same, other = [], []
            for key, obj in self.ram.items():
                if key not in self._active_keys:
                    (same if obj.nbytes == nbytes else other).append(key)
            # A list-membership dedup here used to turn every pressure check
            # into quadratic work over thousands of RAM experts.
            self.stats["host_guard_scan_s"] += time.perf_counter() - scan_start
            released = 0
            for key in same + other:
                if released >= required:
                    break
                freed = self._evict_ram(key)
                released += freed
                self.stats["host_guard_evictions"] += 1
                self.stats["host_guard_evicted_bytes"] += freed
            # Reclaim to a high-water mark once, rather than free/register one
            # allocation on every subsequent miss close to the reserve floor.
            self._empty_host_cache()
            if mem_available_bytes() < max(512 * 1024**2, target):
                raise MemoryError("insufficient host headroom for expert IO; reduce the RAM cache budget")
        finally:
            self.stats["host_guard_s"] += time.perf_counter() - start

    def _ram_get(self, key):
        obj = self.ram.get(key)
        if obj is not None:
            self.ram.move_to_end(key)
        return obj

    def _drain_prefetch(self):
        # Only the inference thread mutates caches, pointers, and statistics.
        for key, future in list(self._pending.items()):
            state = getattr(self, "_chunk_prefetch", None)
            if state is not None and key in state.entries:
                continue  # Exact next-span reads retain their origin/ownership.
            if not future.done():
                continue
            pinned, elapsed = future.result()
            del self._pending[key]
            self._account_read(pinned, elapsed, prefetched=True)
            if key not in self.frozen and key not in self.vram and key not in self.ram:
                if self._ram_insert(key, pinned):
                    self._prefetched.add(key)

    def _prefetch_disk(self, layer):
        self._drain_prefetch()
        if not self.retain_host or not self.prefetch_layers:
            return
        # Previous-token routes are predictions only. Incorrect predictions never
        # change the gate result or cause an unavailable expert to be used.
        for ahead in range(1, self.prefetch_layers + 1):
            lid = (layer + ahead) % self.n_layers
            for expert in self.last_ids.get(lid, ()):
                if len(self._pending) >= self.prefetch_limit:
                    return
                key = (lid, expert)
                if key in self.frozen or key in self.vram or key in self.ram:
                    continue
                try:
                    self._submit_read(key)
                except MemoryError:
                    return

    def _stage_cold(self, layer, ids, next_ids=None):
        if not self.retain_host:
            return {}
        missing = []
        for expert in dict.fromkeys(ids):
            key = (layer, expert)
            if key in self.frozen or key in self.vram or key in self.ram:
                continue
            missing.append(expert)
        if not missing:
            self._prefetch_next_chunk(layer, next_ids)
            return {}
        submit_start = time.perf_counter()
        for expert in missing:
            self._claim_next_chunk_read((layer, expert))
            self._submit_read((layer, expert))
        self.stats["io_submit"] += time.perf_counter() - submit_start
        self._prefetch_next_chunk(layer, next_ids)
        staged = {}
        t0 = time.perf_counter()
        for expert in missing:
            key = (layer, expert)
            pinned, elapsed = self._take_pending_read(key)
            self._account_read(pinned, elapsed)
            self.stats["nvme"] += 1
            self.stats["nvme_bytes"] += pinned.nbytes
            self._ram_insert(key, pinned)
            # Hold requested sources even if an unusually small RAM budget cannot
            # retain all six experts at once. Never read the same miss twice.
            staged[expert] = pinned
        self.stats["io_wait"] += time.perf_counter() - t0
        return staged

    def bump(self, layer: int, ids):
        row = self.counts[layer]
        weight = int(self.bump_weight)
        for expert in ids:
            row[expert] += weight

    def reset_stats(self):
        for k in self.stats:
            self.stats[k] = 0
        if getattr(self, "host_allocator_reserve_bytes", 0):
            self._host_allocator_last_trim = None

    def _fields(self, layer, expert):
        return self.slots[layer * N_EXPERTS + expert]

    def _sig(self, layer, expert):
        key = (layer, expert)
        sig = self.sigs.get(key)
        if sig is None:
            sig = _signature(self._fields(layer, expert))
            self.sigs[key] = sig
        return sig

    def ranked_sets(self, frozen_bytes: int, ram_bytes: int):
        """A small per-layer floor, then global frequency/byte placement.

        Unseen RAM warming is optional; the byte capacity is preserved either way,
        so dynamically discovered experts can fill unused capacity after startup.
        """
        total = max(0, int(ram_bytes) - getattr(self, "cpu_private_host_budget_bytes", 0))
        reserve = getattr(self, "host_allocator_reserve_bytes", 0)
        if reserve:
            if total < reserve:
                raise ValueError("Host allocator reserve exceeds the actual main-host budget")
            self._host_allocator_total_budget_bytes = total
        self.ram_budget_bytes = total - reserve
        candidates = []
        by_layer = []
        unseen = []
        for layer in range(self.n_layers):
            order = np.argsort(-self.counts[layer], kind="stable")
            positives = [int(e) for e in order if self.counts[layer, e] > 0]
            by_layer.append(positives[:6])
            for expert in positives:
                cost = self._fields(layer, expert)[1]
                candidates.append((float(self.counts[layer, expert]) / cost, layer, expert))
            unseen.append([int(e) for e in order if self.counts[layer, e] == 0])
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        floor = [(l, row[rank]) for rank in range(6)
                 for l, row in enumerate(by_layer) if rank < len(row)]
        vram_pairs = []
        chosen = set()
        spent = 0
        for layer, expert in floor + [(l, e) for _, l, e in candidates]:
            key = (layer, expert)
            cost = self._fields(layer, expert)[1]
            if key not in chosen and spent + cost <= frozen_bytes:
                vram_pairs.append(key)
                chosen.add(key)
                spent += cost
        ram_pairs = []
        ram_spent = 0
        ram_order = [(l, e) for _, l, e in candidates if (l, e) not in chosen]
        if self.warm_unseen:
            ram_order += [(l, row[rank]) for rank in range(N_EXPERTS)
                          for l, row in enumerate(unseen) if rank < len(row)]
        ram_order.sort(key=lambda le: -float(self.counts[le]) / pinned_allocation_charge(self._fields(*le)[1]))
        for layer, expert in ram_order:
            cost = pinned_allocation_charge(self._fields(layer, expert)[1])
            if ram_spent + cost <= self.ram_budget_bytes:
                ram_pairs.append((layer, expert))
                ram_spent += cost
        return vram_pairs, ram_pairs

    def _check_k(self, layer, k):
        if layer in self._checked:
            return
        expect = self.kmap.get("routed", {}).get(str(layer))
        if expect is not None and int(k) != int(expect):
            raise RuntimeError(f"layer {layer} trellis K {k} != kmap {expect}")
        self._checked.add(layer)

    def _read_record(self, layer, expert):
        fields = self._fields(layer, expert)
        rec_off, nbytes = fields[0], fields[1]
        buf = os.pread(self.pack, nbytes, rec_off)
        if len(buf) != nbytes:
            raise RuntimeError(f"short expert read {layer}:{expert}")
        return fields, buf

    def _view_pinned(self, host, rel, nbytes, dtype):
        sl = host[rel : rel + nbytes]
        if sl.data_ptr() % torch.empty((), dtype=dtype).element_size():
            sl = sl.clone()
        return sl.view(dtype)

    def _pin(self, layer, fields, buf) -> PinnedExpert:
        host = torch.empty(len(buf), dtype=torch.uint8, pin_memory=True)
        host.numpy()[:] = np.frombuffer(buf, dtype=np.uint8)
        return self._pinned_views(host, fields)

    def _pinned_views(self, host, fields) -> PinnedExpert:
        parts = []
        for pi in range(3):
            base = 4 + pi * 11
            in_f, out_f, k, _pad, suh_b, svh_b, tr_b, _mul1, rel_suh, rel_svh, rel_trellis = fields[base : base + 11]
            suh = self._view_pinned(host, rel_suh, suh_b, torch.float16)
            svh = self._view_pinned(host, rel_svh, svh_b, torch.float16)
            tre = self._view_pinned(host, rel_trellis, tr_b, torch.int16)
            parts.append((suh, svh, tre))
        return PinnedExpert(host, parts, fields[1], pinned_allocation_charge(fields[1]))

    def _read_pinned_timed(self, layer, expert):
        t0 = time.perf_counter()
        fields = self._fields(layer, expert)
        host = torch.empty(fields[1], dtype=torch.uint8, pin_memory=True)
        allocated = time.perf_counter()
        target = memoryview(host.numpy())
        done = 0
        io_start = time.perf_counter()
        reader = getattr(self, "_direct_reader", None)
        read_stats = {}
        if reader is not None:
            # The existing allocation is used unchanged; no hidden alignment
            # over-allocation/slicing can escape pinned-memory accounting.
            read_stats = reader.read_into(target, int(fields[0]))
        else:
            while done < fields[1]:
                # Buffered preadv already avoids a Python bytes staging copy.
                got = os.preadv(self.pack, [target[done:]], fields[0] + done)
                if not 0 < got <= fields[1] - done:
                    raise RuntimeError(f"short expert read {layer}:{expert}")
                done += got
        io_end = time.perf_counter()
        pinned = self._pinned_views(host, fields)
        pinned.direct_io_stats = read_stats
        finished = time.perf_counter()
        pinned.alloc_s = allocated - t0
        pinned.pread_s = io_end - io_start
        pinned.views_s = (io_start - allocated) + (finished - io_end)
        return pinned, finished - t0

    def _alloc_slot(self, sig, capacity=None) -> GpuExpert:
        payload = sum(s + v + t for _, _, _, s, v, t in sig)
        nbytes = (payload + 4095) // 4096 * 4096
        storage = torch.empty(max(nbytes, capacity or 0), dtype=torch.uint8, device="cuda")
        slot = GpuExpert([], storage)
        self._rebind_slot(slot, sig)
        return slot

    def _rebind_slot(self, slot, sig):
        """Reinterpret a retired uniform volatile allocation for another EXL3 K."""
        storage = slot.storage
        parts = []
        offset = 0
        for _in_f, _out_f, _k, suh_b, svh_b, tr_b in sig:
            suh = storage[offset : offset + suh_b].view(torch.float16)
            offset += suh_b
            svh = storage[offset : offset + svh_b].view(torch.float16)
            offset += svh_b
            tre = storage[offset : offset + tr_b].view(torch.int16)
            offset += tr_b
            parts.append((suh, svh, tre))
        slot.parts = parts

    def _blit(self, slot: GpuExpert, pinned: PinnedExpert):
        promotion = getattr(self, "_layer_promotion", None)
        backing = None if promotion is None else getattr(promotion, "host_backing", None)
        if backing is not None and promotion.active:
            def enqueue():
                slot.storage[:pinned.nbytes].copy_(pinned.host, non_blocking=True)
                self.stats["h2d_bytes"] += pinned.nbytes
            backing.blit(slot, pinned, torch.cuda.current_stream(slot.storage.device), enqueue)
            return
        slot.storage[:pinned.nbytes].copy_(pinned.host, non_blocking=True)
        self.stats["h2d_bytes"] += pinned.nbytes

    def _bind(self, layer, expert, slot: GpuExpert):
        g, d, u = slot.parts
        fast = getattr(self, "_fast_pointer_table", None)
        if fast is not None:
            try:
                fast.bind(layer, expert, (g[2].data_ptr(), g[0].data_ptr(), g[1].data_ptr(),
                                          u[2].data_ptr(), u[0].data_ptr(), u[1].data_ptr(),
                                          d[2].data_ptr(), d[0].data_ptr(), d[1].data_ptr()))
            except BaseException:
                self._fast_pointer_failed = True
                raise
            self.dirty.add(layer)
            return
        h = self.host_ptrs[layer]
        h[0, expert] = g[2].data_ptr()
        h[1, expert] = g[0].data_ptr()
        h[2, expert] = g[1].data_ptr()
        h[3, expert] = u[2].data_ptr()
        h[4, expert] = u[0].data_ptr()
        h[5, expert] = u[1].data_ptr()
        h[6, expert] = d[2].data_ptr()
        h[7, expert] = d[0].data_ptr()
        h[8, expert] = d[1].data_ptr()
        self.dirty.add(layer)

    def _unbind(self, layer, expert):
        fast = getattr(self, "_fast_pointer_table", None)
        if fast is None:
            self.host_ptrs[layer, :, expert] = 0
        else:
            try:
                fast.unbind(layer, expert)
            except BaseException:
                self._fast_pointer_failed = True
                raise
        self.dirty.add(layer)

    def _flush(self, layer):
        if layer not in self.dirty:
            return
        fast = getattr(self, "_fast_pointer_table", None)
        try:
            if fast is not None:
                fast.retire(layer)
            self.packed[layer].copy_(self.host_ptrs[layer], non_blocking=True)
            if fast is not None:
                fast.record(layer, torch.cuda.current_stream())
        except BaseException:
            if fast is not None:
                self._fast_pointer_failed = True
            raise
        self.dirty.discard(layer)

    def _checkout(self, sig) -> GpuExpert:
        pool = self.pools[sig]
        if pool:
            return pool.pop()
        # All volatile allocations have the largest record capacity. A K=2 layer
        # can borrow K=3 storage (and vice versa) without growing GPU residency.
        for other_pool in self.pools.values():
            if other_pool:
                slot = other_pool.pop()
                self._rebind_slot(slot, sig)
                return slot
        allocated = len(self.vram) + sum(len(items) for items in self.pools.values())
        if allocated >= self.stage_limit:
            raise RuntimeError("volatile slot capacity exhausted; active experts must not be evicted")
        return self._alloc_slot(sig, capacity=self._volatile_slot_bytes)

    def _evict_one(self, sig=None):
        key = next((key for key in self.vram
                    if key not in self._active_keys and
                    (sig is None or self._sig(*key) == sig)), None)
        if key is None:
            return False
        slot = self.vram.pop(key)
        slot.retire()
        layer, expert = key
        self._unbind(layer, expert)
        self.pools[self._sig(layer, expert)].append(slot)
        return True

    def _install_if_absent(self, layer, expert, stat: bool, allow_evict: bool, source=None):
        key = (layer, expert)
        if key in self.frozen:
            if stat:
                self.stats["vram"] += 1
                self.stats["frozen_hits"] += 1
            return
        hit = self.vram.get(key)
        if hit is not None:
            self.vram.move_to_end(key)
            if stat:
                self.stats["vram"] += 1
                self.stats["volatile_hits"] += 1
            return
        sig = self._sig(layer, expert)
        if not allow_evict and (not any(self.pools.values()) or len(self.vram) >= self.stage_limit):
            return
        pinned = source if source is not None else self._ram_get(key)
        if pinned is None:
            # GPU lookahead must not synchronously read the disk before attention.
            if not allow_evict:
                return
            t0 = time.perf_counter()
            self._claim_next_chunk_read(key)
            self._submit_read(key)
            pinned, elapsed = self._take_pending_read(key)
            self.stats["io_wait"] += time.perf_counter() - t0
            self._account_read(pinned, elapsed)
            self._ram_insert(key, pinned)
            if stat:
                self.stats["nvme"] += 1
                self.stats["nvme_bytes"] += pinned.nbytes
        elif stat:
            self.stats["ram"] += 1
            self.stats["ram_bytes"] += pinned.nbytes
            if key in self._prefetched:
                self.stats["prefetch_hits"] += 1
                self._prefetched.discard(key)
        self._check_k(layer, int(self._fields(layer, expert)[2]))
        while allow_evict and len(self.vram) >= self.stage_limit and self.vram:
            if not self._evict_one():
                raise RuntimeError("volatile cache cannot hold the active expert set")
        slot = self._checkout(sig)
        self._blit(slot, pinned)
        if not self.retain_host:
            torch.cuda.current_stream().synchronize()
        if self.copy_stream is not None and torch.cuda.current_stream() == self.copy_stream:
            slot.event.record(self.copy_stream)
            slot.seen = True
        self.vram[key] = slot
        self._bind(layer, expert, slot)

    def preload(self, vram_pairs, ram_pairs):
        self.ram_nbytes = sum(obj.nbytes for obj in self.ram.values())
        self.ram_charge_bytes = sum(obj.allocation_bytes for obj in self.ram.values())
        order = sorted(vram_pairs, key=lambda le: self._fields(le[0], le[1])[0])
        for i, (layer, expert) in enumerate(order):
            if torch.cuda.mem_get_info()[0] < self.minimum_vram_available_bytes + self._fields(layer, expert)[1]:
                print(f"  stop vram preload at {i}: free memory low", flush=True)
                break
            pinned, _ = self._read_pinned_timed(layer, expert)
            slot = self._alloc_slot(self._sig(layer, expert))
            self._blit(slot, pinned)
            torch.cuda.current_stream().synchronize()
            self.frozen[(layer, expert)] = slot
            self._bind(layer, expert, slot)
            if i and i % 200 == 0:
                print(f"  vram preload {i}/{len(order)}", flush=True)
        for layer in range(self.n_layers):
            self._flush(layer)
        torch.cuda.synchronize()
        hist = defaultdict(int)
        for layer in range(self.n_layers):
            for expert in range(N_EXPERTS):
                hist[self._sig(layer, expert)] += 1
        total = sum(hist.values()) or 1
        target = self.stage_limit
        for sig, n in sorted(hist.items(), key=lambda kv: -kv[1]):
            if target <= 0:
                break
            quota = min(target, max(1, round(self.stage_limit * n / total)))
            made = 0
            for _ in range(quota):
                needed = self._volatile_slot_bytes
                if torch.cuda.mem_get_info()[0] < self.minimum_vram_available_bytes + needed:
                    target = 0
                    break
                self.pools[sig].append(self._alloc_slot(sig, capacity=self._volatile_slot_bytes))
                made += 1
            target -= made
        pooled = sum(len(v) for v in self.pools.values())
        self.stage_limit = pooled
        self._empty_host_cache()
        self._host_baseline_bytes = self._host_reserved_bytes()
        ram_order = sorted(ram_pairs, key=lambda le: self._fields(le[0], le[1])[0])
        for i, (layer, expert) in enumerate(ram_order):
            if mem_available_bytes() < self.minimum_available_bytes + pinned_allocation_charge(self._fields(layer, expert)[1]):
                print(f"  stop ram preload at {i}: host memory low", flush=True)
                break
            pinned, _ = self._read_pinned_timed(layer, expert)
            self._ram_insert((layer, expert), pinned)
            if i and i % 400 == 0:
                print(f"  ram preload {i}/{len(ram_order)}", flush=True)
        self.copy_stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        self._trim_host_reservation()
        # Keep the configured byte capacity; a sparse calibration must not shrink
        # the runtime cache to the number of experts observed during calibration.
        self.reset_stats()
        per = len(self.frozen) // max(self.n_layers, 1)
        print(
            f"pinned vram {len(self.frozen)}  volatile {self.stage_limit}  ram {len(self.ram)}  "
            f"per layer vram {per}  ram payload {self.ram_nbytes/1024**3:.2f} GiB "
            f"charge {self.ram_charge_bytes/1024**3:.2f}/{self.ram_budget_bytes/1024**3:.2f} GiB "
            f"host_reserved {self._host_reserved_bytes()/1024**3:.2f} GiB",
            flush=True,
        )

    def prefetch_gpu(self, layer: int):
        """Copy the previous token's experts for this layer while attention runs."""
        self._check_placement_health()
        promotion = getattr(self, "_layer_promotion", None)
        backing = None if promotion is None else getattr(promotion, "host_backing", None)
        if backing is not None and promotion.active:
            backing.skip_prefetch()
            return
        self._prefetch_disk(layer)
        ids = self.last_ids.get(layer)
        if not ids or self.copy_stream is None:
            return
        with torch.cuda.stream(self.copy_stream):
            for expert in ids:
                self._install_if_absent(layer, expert, False, False)
            self._flush(layer)
        self.copy_event = self.copy_stream.record_event()

    def _coop512_capable(self, x, topk, k, inter):
        """Physical/backend contract only, without routes, sync or allocations."""
        if (not getattr(self, "_coop512_requested", False) or getattr(self, "cpu_backend", None) is not None
                or topk != 6 or k not in (2, 3) or inter != 2304 or x.shape[-1] != 5120
                or x.dtype != torch.float16 or x.device.type != "cuda"):
            return False
        strict = os.environ.get("DS41_STRICT_BATCH", "1") == "1"
        if (os.environ.get("EXL3_MOE_COOP_WIDE", "0" if strict else "auto") != "0"
                or os.environ.get("EXL3_MOE_COOP_KSPLIT", "1" if strict else "auto") != "1"):
            return False  # Never silently change an explicitly different target.
        devices = getattr(self, "_coop512_devices", None)
        if devices is None:
            devices = self._coop512_devices = {}
        key = str(x.device)
        if key not in devices:
            devices[key] = torch.cuda.get_device_capability(x.device) == (12, 0)
        return devices[key]

    def _moe_observer_slot_limit(self):
        observer = getattr(self, "route_observer", None)
        # Capacity counts repeated routed entries, not physical weight slots.
        # Never split a real kernel into fictitious observation calls.
        limit = 384 if observer is None else getattr(observer, "max_routed_slots", 256)
        if type(limit) is not int or not (1 <= limit <= 384 or limit == 1536):
            raise ValueError("invalid actual-call route observer capacity")
        if limit == 1536 and (not getattr(self, "_coop2048_requested", False)
                             or getattr(self, "_native2048_runtime", None) is None):
            raise ValueError("1536-slot observation requires initialized native2048")
        return limit

    def moe_chunk_limits(self, x, topk, k, inter):
        """Explicit pure-planner bounds; actual union is checked again in moe."""
        self._check_placement_health()
        if getattr(self, "_coop3072_requested", False):
            self.validate_native3072_placement()
            if self._coop3072_capable(x, topk, k, inter):
                if verification_max_rows()==1024:
                    return dict(max_rows=1024,max_slots=6144)
                return dict(max_rows=512, max_slots=3072)
            if verification_max_rows()==1024:
                raise ValueError("Cap1024 requires its actual checked3072 joint-layer backend")
        if getattr(getattr(self, "route_observer", None), "max_routed_slots", 256) == 1536:
            self._moe_observer_slot_limit()  # Reject an unattached opt-in before planning.
        if getattr(self, "_coop2048_requested", False):
            self.validate_native2048_placement()
            # Only the explicitly identity-bound1536 tier can observe one
            # large native call. Original256/384 observers retain old chunks.
            if self._coop2048_capable(x, topk, k, inter):
                return dict(max_rows=min(256, verification_max_rows()), max_slots=1536)
        observer_limit = self._moe_observer_slot_limit() if getattr(self, "_coop512_requested", False) else 256
        if self._coop512_capable(x, topk, k, inter) and observer_limit >= 384:
            return dict(max_rows=64, max_slots=384)
        return dict(max_rows=42, max_slots=min(256, observer_limit))

    def _coop2048_capable(self, x, topk, k, inter):
        return (getattr(self, "_coop2048_requested", False)
                and getattr(self, "_native2048_runtime", None) is not None
                and (getattr(self, "route_observer", None) is None
                     or self._moe_observer_slot_limit() == 1536)
                and 6 <= self.stage_limit <= 256
                and self._coop512_capable(x, topk, k, inter))

    def _coop3072_capable(self, x, topk, k, inter):
        return (getattr(self, "_coop3072_requested", False)
                and getattr(self, "_native3072_runtime", None) is not None
                and getattr(self, "route_observer", None) is None
                and not getattr(self, "_coop2048_requested", False)
                and self.stage_limit == 384
                and self._coop512_capable(x, topk, k, inter))

    def _run_coop(self, layer, x, indices, weights, k, limit, inter, *, native512=False, native2048=False,
                  native3072=False, route_snapshot=None):
        rows, hidden = x.reshape(-1, x.shape[-1]).shape
        topk = indices.shape[-1]
        self._flush(layer)
        runner_key = (("native512", hidden, inter, str(x.device)) if native512
                      else (rows, topk, hidden, inter, str(x.device)))
        runner = self._coops.get(runner_key)
        fast = getattr(self, "_fast_pointer_table", None)
        try:
            if native3072:
                if native512 or native2048 or route_snapshot is not None:
                    raise RuntimeError("coop3072 does not accept private route proofs or mixed dispatch")
                runtime = getattr(self, "_native3072_runtime", None)
                if runtime is None:
                    raise RuntimeError("coop3072 has no pre-budget workspace owner")
                self.coop = runtime._runner
                result = runtime.run(self, x.reshape(rows, hidden), indices.reshape(rows, topk),
                                     weights.reshape(rows, topk), tuple(self.packed[layer].unbind(0)), k, limit)
                if fast is not None:
                    fast.record(layer, torch.cuda.current_stream(), reader=True)
                return result
            if native2048:
                if native512 or route_snapshot is not None:
                    raise RuntimeError("coop2048 does not accept private route proofs or mixed dispatch")
                runtime = getattr(self, "_native2048_runtime", None)
                if runtime is None:
                    raise RuntimeError("coop2048 has no pre-budget workspace owner")
                self.coop = runtime._runner
                result = runtime.run(self, x.reshape(rows, hidden), indices.reshape(rows, topk),
                                     weights.reshape(rows, topk), tuple(self.packed[layer].unbind(0)), k, limit)
                if fast is not None:
                    fast.record(layer, torch.cuda.current_stream(), reader=True)
                return result
            if runner is None:
                if native512:
                    # Import/build is explicit opt-in only. One64-row scratch
                    # workspace per device, not one per actual row geometry.
                    from ds41.moe_coop512 import CoopRunner512
                    runner = CoopRunner512(x.device, topk, hidden, inter, max_rows=64)
                else:
                    from ds41.moe_coop import CoopRunner
                    runner = CoopRunner(x.device, topk, hidden, inter, rows=rows)
                self._coops[runner_key] = runner
            self.coop = runner
            packed = self.packed[layer]
            if native512:
                # The original runner accepts one [9,384] table; the isolated
                # native512 ABI takes nine borrowed row views. Unbind retains
                # the same GPU storage and pointer-table update ordering.
                packed = tuple(packed.unbind(0))
            if route_snapshot is None:
                result = runner.run(x, indices, weights, packed, k, limit)
            else:
                runtime = getattr(self, "_route_snapshot_runtime", None)
                if not native512 or runtime is None:
                    raise RuntimeError("private route proof requires this cache's native512 runtime")
                result = runtime.native_run(route_snapshot, self, layer, runner, x, indices, weights,
                                            packed, k, limit, inter)
            if fast is not None:
                fast.record(layer, torch.cuda.current_stream(), reader=True)
            return result
        except BaseException:
            if native3072:
                self._coop3072_failed = True
                self._native3072_runtime.poison()
            if native2048:
                self._coop2048_failed = True
                self._native2048_runtime.poison()
            if native512:
                self._coop512_failed = True
            if fast is not None:
                self._fast_pointer_failed = True
            raise

    def _cpu_offer(self, layer, x, indices, weights, ids, staged, limit):
        """Issue borrowed cold records without staging their weights to VRAM."""
        backend = getattr(self, "cpu_backend", None)
        if backend is None or not self.retain_host or x.reshape(-1, x.shape[-1]).shape[0] != 1:
            return None
        cold = [e for e in ids if (layer, e) not in self.frozen and (layer, e) not in self.vram]
        if not cold:
            return None
        sources = [staged.get(e) or self._ram_get((layer, e)) for e in cold]
        if any(source is None for source in sources):
            return None
        cold_set = set(cold)
        routed = indices.reshape(1, -1)
        columns = [i for i, e in enumerate(routed.detach().reshape(-1).tolist()) if int(e) in cold_set]
        if len(columns) != len(cold):
            return None  # Duplicate selections use the original GPU semantics.
        index = torch.tensor(columns, device=weights.device, dtype=torch.long)
        before = dict(backend.stats)
        ticket = backend.submit(layer, x.reshape(1, -1), cold,
                                weights.reshape(1, -1).index_select(1, index), sources, limit=limit)
        if ticket is None:
            self.stats["cpu_fallback"] += 1
            return None
        for expert, source in zip(cold, sources):
            if expert not in staged:
                self.stats["ram"] += 1
                self.stats["ram_bytes"] += source.nbytes
                key = (layer, expert)
                if key in self._prefetched:
                    self.stats["prefetch_hits"] += 1
                    self._prefetched.discard(key)
        self.stats["cpu_experts"] += len(cold)
        self.stats["cpu_layers"] += 1
        self.stats["cpu_weight_bytes"] += sum(s.nbytes for s in sources)
        hot_columns = [i for i in range(routed.shape[1]) if i not in columns]
        hot_index = torch.tensor(hot_columns, device=indices.device, dtype=torch.long)
        return ticket, before, cold_set, hot_index

    def _collect_cpu(self, offer, gpu_hot):
        ticket, before, _, _ = offer
        backend = self.cpu_backend
        try:
            return backend.collect(ticket, gpu_hot)
        finally:
            mapping = {"cpu_seconds": "cpu_compute_s", "collect_wait_seconds": "cpu_wait_s",
                       "d2h_bytes": "cpu_d2h_bytes", "h2d_bytes": "cpu_result_h2d_bytes"}
            for source, dest in mapping.items():
                self.stats[dest] += backend.stats.get(source, 0) - before.get(source, 0)

    def moe(self, layer, x, indices, weights, k, limit, inter, *, next_ids=None, _route_snapshot=None):
        self._check_placement_health()
        promotion = getattr(self, "_layer_promotion", None)
        if promotion is not None:
            promotion.before_call(layer)
        if getattr(self, "_coop3072_requested", False):
            self.validate_native3072_placement()
            if _route_snapshot is not None or next_ids is not None:
                raise ValueError("coop3072 excludes private route proofs and next-chunk routes")
            if x.ndim > 2 and x.shape[0] != 1:
                raise ValueError("coop3072 target verification requires model batch B=1")
        if getattr(getattr(self, "route_observer", None), "max_routed_slots", 256) == 1536:
            self._moe_observer_slot_limit()  # No new device readback or implicit opt-in.
        if getattr(self, "_coop2048_requested", False):
            self.validate_native2048_placement()
        if next_ids is not None:
            if not self.next_chunk_prefetch_active:
                raise ValueError("next-span routes require an active decode prefetch scope")
            next_ids = validate_next_ids(next_ids)
        rows, hidden = x.reshape(-1, x.shape[-1]).shape
        topk = indices.shape[-1]
        if getattr(self, "adaptive_gpu_enabled", False):
            if rows > verification_max_rows():
                raise ValueError("adaptive GPU target rows exceed the configured verification limit")
            if x.ndim > 2 and x.shape[0] != 1:
                raise ValueError("adaptive GPU target verification requires model batch B=1")
        requested = getattr(self, "_coop512_requested", False)
        joint3072 = (getattr(self,"_coop3072_requested",False) and verification_max_rows()==1024
                     and topk==6 and 513<=rows<=1024)
        expanded3072 = getattr(self, "_coop3072_requested", False) and topk == 6 and (65 <= rows <= 512 or joint3072)
        expanded2048 = getattr(self, "_coop2048_requested", False) and topk == 6 and 65 <= rows <= verification_max_rows()
        expanded = requested and topk == 6 and 1 <= rows <= 64
        if rows * topk > (6144 if joint3072 else 3072 if expanded3072 else 1536 if expanded2048 else 384 if expanded else 256) or indices.numel() != rows * topk:
            raise ValueError("EXL3 MoE expects matching rows and at most 256 routed slots")
        if self.copy_event is not None:
            torch.cuda.current_stream().wait_event(self.copy_event)
            self.copy_event = None
        runtime = getattr(self, "_route_snapshot_runtime", None)
        if _route_snapshot is None:
            routed_ids = [int(v) for v in indices.detach().view(-1).tolist()]
            if runtime is not None:
                runtime.legacy_readback()
        else:
            if runtime is None:
                self._route_snapshot_failed = True
                raise RuntimeError("private route proof supplied to disabled cache")
            routed_ids = list(runtime.validate_span(_route_snapshot, self, layer, x, indices, weights, k, inter))
        ids = list(dict.fromkeys(routed_ids))
        if getattr(self, "_coop3072_requested", False) and any(not 0 <= expert < N_EXPERTS for expert in ids):
            raise ValueError("coop3072 route IDs must be in[0,384) before expert staging")
        needed = sum((layer, expert) not in self.frozen for expert in ids)
        cpu_eligible = rows == 1 and getattr(self, "cpu_backend", None) is not None and self.retain_host
        observer_limit = (3072 if expanded3072 else
                          self._moe_observer_slot_limit() if expanded or expanded2048 else 256)
        native3072 = (expanded3072 and self._coop3072_capable(x, topk, k, inter)
                      and needed <= self.stage_limit)
        if joint3072 and not native3072:
            raise ValueError("Cap1024 joint-layer routes require checked3072 and the complete384-slot union")
        native2048 = (expanded2048 and self._coop2048_capable(x, topk, k, inter)
                      and needed <= self.stage_limit)
        native512 = (expanded and self._coop512_capable(x, topk, k, inter)
                     and rows*topk <= observer_limit and needed <= self.stage_limit)
        if _route_snapshot is not None and not native512:
            runtime.fallback_call("actual_call_requires_checked_fallback")
            return self.moe(layer, x, indices, weights, k, limit, inter, next_ids=next_ids)
        if (expanded or expanded2048 or expanded3072) and not (native512 or native2048 or native3072) and (rows*topk > min(256, observer_limit)
                                          or needed > self.stage_limit and not cpu_eligible):
            from ds41.route_chunks import plan_route_chunks
            # Split BEFORE staging/active ownership. Future-chunk experts must
            # remain evictable. Reuse the existing CPU IDs; no second readback.
            route_rows = [routed_ids[start:start+topk] for start in range(0, len(routed_ids), topk)]
            fallback512 = ((expanded2048 or expanded3072) and observer_limit >= 384
                           and self._coop512_capable(x, topk, k, inter))
            spans = plan_route_chunks(route_rows, self.frozen, layer, self.stage_limit,
                                      max_rows=64 if fallback512 else 42,
                                      max_slots=384 if fallback512 else min(256, observer_limit))
            flat_x, flat_ids = x.reshape(rows, hidden), indices.reshape(rows, topk)
            flat_weights = weights.reshape(rows, topk)
            parts = []
            for number, (start, end) in enumerate(spans):
                future_ids = None
                if self.next_chunk_prefetch_active:
                    future_ids = (routed_ids[spans[number+1][0]*topk:spans[number+1][1]*topk]
                                  if number + 1 < len(spans) else next_ids)
                part = self._moe_single(layer, flat_x[start:end], flat_ids[start:end], flat_weights[start:end],
                                        k, limit, inter, routed_ids[start*topk:end*topk], native512=fallback512,
                                        next_ids=future_ids)
                parts.append(part.clone())  # Own scratch before the next old call.
                if expanded3072:
                    self.stats["moe_coop3072_fallback_calls"] = self.stats.get("moe_coop3072_fallback_calls", 0) + 1
                    self.stats["moe_coop3072_fallback_rows"] = self.stats.get("moe_coop3072_fallback_rows", 0) + end-start
                elif expanded2048:
                    self.stats["moe_coop2048_fallback_calls"] = self.stats.get("moe_coop2048_fallback_calls", 0) + 1
                    self.stats["moe_coop2048_fallback_rows"] = self.stats.get("moe_coop2048_fallback_rows", 0) + end-start
                else:
                    self.stats["moe_coop512_fallback_calls"] = self.stats.get("moe_coop512_fallback_calls", 0) + 1
                    self.stats["moe_coop512_fallback_rows"] = self.stats.get("moe_coop512_fallback_rows", 0) + end-start
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        if _route_snapshot is None:
            if native3072:
                return self._moe_single(layer, x, indices, weights, k, limit, inter, routed_ids,
                                        native3072=True, ids=ids, needed=needed,joint3072=joint3072)
            return self._moe_single(layer, x, indices, weights, k, limit, inter, routed_ids,
                                    native512=native512, native2048=native2048, ids=ids, needed=needed, next_ids=next_ids)
        return self._moe_single(layer, x, indices, weights, k, limit, inter, routed_ids,
                                native512=native512, native2048=native2048, ids=ids, needed=needed, next_ids=next_ids,
                                route_snapshot=_route_snapshot)

    def _moe_single(self, layer, x, indices, weights, k, limit, inter, routed_ids, *, native512=False, native2048=False, native3072=False,
                    ids=None, needed=None, next_ids=None, route_snapshot=None,joint3072=False):
        """One actual kernel geometry and protected union; CPU IDs already read."""
        pipeline = getattr(self, "_demand_h2d_pipeline", None)
        if pipeline is not None:
            pipeline.validate(self, next_ids)
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        if joint3072 and (not native3072 or verification_max_rows()!=1024 or next_ids is not None
                         or route_snapshot is not None or getattr(self,"route_observer",None) is not None
                         or getattr(self,"_layer_promotion",None) is not None):
            raise ValueError("Joint3072 requires exact1024, no observer/promotion/private route or prefetch")
        if ids is None:
            ids = list(dict.fromkeys(routed_ids))
        if needed is None:
            needed = sum((layer, expert) not in self.frozen for expert in ids)
        cpu_eligible = rows == 1 and getattr(self, "cpu_backend", None) is not None and self.retain_host
        if needed > self.stage_limit and not cpu_eligible:
            raise ValueError(f"batch needs {needed} volatile experts, cache has {self.stage_limit}; reduce rows")
        prev = self.last_ids.get(layer)
        if prev:
            prev_set = set(prev)
            self.stats["reuse"] += sum(i in prev_set for i in routed_ids)
            self.stats["reuse_n"] += len(routed_ids)
        # After batched verification, the last row is the next token's useful
        # temporal prediction; retaining the entire batch union wastes prefetch.
        self.last_ids[layer] = routed_ids[-indices.shape[-1] :]
        self.bump(layer, routed_ids)
        if getattr(self, "adaptive_gpu_enabled", False):
            # Placement statistics describe executed target routing work, not
            # accepted output tokens. Reject/restore does not undo this profile.
            self.stats["adaptive_gpu_executed_layer_rows"] += rows
            self.stats["adaptive_gpu_executed_expert_uses"] += len(routed_ids)
            for expert in routed_ids:
                self._adaptive_counts[layer, expert] += 1
        self._active_keys = {(layer, expert) for expert in ids}
        offer = None
        gpu_ids = ids
        staged, pipe_call, pipe_error = {}, None, None
        joint_lease = None
        promotion = getattr(self, "_layer_promotion", None)
        succeeded = False
        try:
            if pipeline is not None:
                demand_stream = torch.cuda.current_stream(x.device)
                pipe_call = pipeline.begin(self, layer, ids, staged,
                                           lambda stream=demand_stream: stream.synchronize())
                pipe_call.install()
            else:
                staged = (self._stage_cold(layer, ids) if next_ids is None
                          else self._stage_cold(layer, ids, next_ids=next_ids))
                offer = self._cpu_offer(layer, x, indices, weights, ids, staged, limit)
                if offer is not None:
                    gpu_ids = [e for e in ids if e not in offer[2]]
                needed = sum((layer, expert) not in self.frozen for expert in gpu_ids)
                if needed > self.stage_limit:
                    raise ValueError(f"batch needs {needed} volatile experts, cache has {self.stage_limit}; reduce rows")
                for expert in gpu_ids:
                    self._install_if_absent(layer, expert, expert not in staged, True, staged.get(expert))
            if offer is None:
                if joint3072:
                    from ds41.native3072_runtime import joint_spans
                    runtime=self._native3072_runtime
                    joint_lease=runtime.begin_joint(self,layer,ids,rows)
                    self.stats['moe_joint3072_stage_calls']=self.stats.get('moe_joint3072_stage_calls',0)+1
                    flat=x.reshape(rows,-1);flat_indices=indices.reshape(rows,-1);flat_weights=weights.reshape(rows,-1)
                    for start,end in joint_spans(rows):
                        part=self._run_coop(layer,flat[start:end],flat_indices[start:end],flat_weights[start:end],
                                            k,limit,inter,native3072=True)
                        # Actual3072 returns owned output; retain it immediately
                        # in the attached transaction before another launch.
                        runtime.retain_joint_output(joint_lease,part)
                        for name,value in (("moe_coop3072_calls",1),("moe_coop3072_rows",end-start),
                                           ("moe_coop3072_routed_slots",(end-start)*indices.shape[-1])):
                            self.stats[name]=self.stats.get(name,0)+value
                    result=torch.cat(joint_lease['outputs'],dim=0)
                elif native3072:
                    result = self._run_coop(layer, x, indices, weights, k, limit, inter, native3072=True)
                    for name, value in (("moe_coop3072_calls", 1), ("moe_coop3072_rows", rows),
                                        ("moe_coop3072_routed_slots", len(routed_ids))):
                        self.stats[name] = self.stats.get(name, 0) + value
                elif native2048:
                    result = self._run_coop(layer, x, indices, weights, k, limit, inter, native2048=True)
                    for name, value in (("moe_coop2048_calls", 1), ("moe_coop2048_rows", rows),
                                        ("moe_coop2048_routed_slots", len(routed_ids))):
                        self.stats[name] = self.stats.get(name, 0) + value
                elif native512:
                    if route_snapshot is None:
                        result = self._run_coop(layer, x, indices, weights, k, limit, inter, native512=True)
                    else:
                        result = self._run_coop(layer, x, indices, weights, k, limit, inter,
                                                native512=True, route_snapshot=route_snapshot)
                    for name, value in (("moe_coop512_calls", 1), ("moe_coop512_rows", rows),
                                        ("moe_coop512_routed_slots", len(routed_ids))):
                        self.stats[name] = self.stats.get(name, 0) + value
                else:
                    result = self._run_coop(layer, x, indices, weights, k, limit, inter)
            else:
                hot_index = offer[3]
                gpu_hot = None
                if gpu_ids:
                    gpu_hot = self._run_coop(layer, x, indices.reshape(1, -1).index_select(1, hot_index),
                                             weights.reshape(1, -1).index_select(1, hot_index), k, limit, inter)
                result = self._collect_cpu(offer, gpu_hot)
            observer = getattr(self, "route_observer", None)
            if observer is not None:
                # Reuse this actual chunk's existing CPU list. No additional
                # GPU readback; diagnostic/draft calls stay outside its scope.
                observer(layer, routed_ids)
            succeeded = True
            return result
        except BaseException as exc:
            if promotion is not None and promotion.active:
                promotion.poison()
            if native3072:
                self._coop3072_failed = True
                self._native3072_runtime.poison()
                if joint_lease is not None:self._native3072_runtime.fail_joint()
            if native2048:
                self._coop2048_failed = True
                self._native2048_runtime.poison()
            if native512:
                self._coop512_failed = True
            if pipe_call is not None:
                pipe_error = exc
                pipe_call.abort(exc)
            raise
        finally:
            # An exceptional GPU launch must still retire the CPU ticket before
            # its borrowed records or fixed pinned buffers can be recycled.
            try:
                if offer is not None and not offer[0].consumed:
                    self._collect_cpu(offer, None)
            finally:
                try:
                    for expert in gpu_ids:
                        slot = self.vram.get((layer, expert))
                        if slot is not None:
                            slot.event.record()
                            slot.seen = True
                except BaseException as exc:
                    if promotion is not None and promotion.active:
                        promotion.poison()
                    if native3072:
                        self._coop3072_failed = True
                        self._native3072_runtime.poison()
                        if joint_lease is not None:self._native3072_runtime.fail_joint()
                    if native2048:
                        self._coop2048_failed = True
                        self._native2048_runtime.poison()
                    if native512:
                        self._coop512_failed = True
                    if pipe_call is not None:
                        pipe_call.abort(exc)
                    if pipe_error is not None:
                        if hasattr(pipe_error, "add_note"):
                            pipe_error.add_note("pipeline slot retirement also failed: " + type(exc).__name__)
                    else:
                        raise
                finally:
                    self._active_keys.clear()
                    if pipe_call is not None:
                        try:
                            pipe_call.close()
                        except BaseException:
                            if promotion is not None and promotion.active:
                                promotion.poison()
                            if joint_lease is not None:
                                self._coop3072_failed=True
                                self._native3072_runtime.fail_joint()
                            raise
            if succeeded and joint_lease is not None:
                try:self._native3072_runtime.finish_joint(joint_lease)
                except BaseException:
                    self._coop3072_failed=True
                    self._native3072_runtime.fail_joint()
                    raise
            if succeeded and promotion is not None:
                promotion.record_call(layer, ids, rows, indices.shape[-1])

    def moe_batch(self, layer, x, indices, weights, k, limit, inter):
        """Exact multi-row verification, sharing the union's weight transfers."""
        return self.moe(layer, x, indices, weights, k, limit, inter)
