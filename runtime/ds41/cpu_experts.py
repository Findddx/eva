"""Optional native-CPU cold experts; never enabled implicitly.

Default weight layout is borrowed, native EXL3: no second RAM expert copy.
The AVX vector backend quantizes activations to int8 and is an APPROXIMATE
compute mode relative to the GPU FP16 path. Callers must expose that choice.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time

import numpy as np
import torch


@dataclass
class CPUExpertTicket:
    owner: object
    slot: object
    sources: tuple
    future: object = None
    consumed: bool = False


class NativeCPUExpertBackend:
    """Bounded decode issue/collect beside a GPU-hot expert launch.

    submit(layer, x_f16_cuda, expert_ids, weights_cuda, pinned_sources) returns
    a ticket or None (unsupported shape, graph capture or all slots busy).
    weights already contain ONLY the cold ids in the same order as sources.
    Collect after launching GPU-hot work; it returns a newly owned FP32 GPU sum.

    Sources match cache.PinnedExpert (.host, .parts, .nbytes and optional
    .allocation_bytes). They are immutable native-layout records. Tickets keep
    them alive through collect even if the RAM LRU evicts their dictionary entry.
    """

    def __init__(self, hidden=5120, threads=16, device="cuda", slots=2,
                 max_cold_experts=6, repack_budget_bytes=0):
        from exllamav3.ext import exllamav3_ext as ext
        self.ext = ext
        self.hidden = int(hidden)
        self.threads = int(threads)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.max_cold_experts = int(max_cold_experts)
        if self.device.type != "cuda" or slots < 1 or self.threads < 1:
            raise ValueError("CPU expert backend requires CUDA, positive slots and threads")
        self.precision = "cpu_activation_int8" if ext.exl3_moe_cpu_has_avx2() else "cpu_scalar_fp32"
        self.repack_budget_bytes = max(0, int(repack_budget_bytes))
        self.repack_bytes = 0
        self._repacked = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ds41-cold-experts")
        self._closed = False
        self._slots = []
        self.stats = dict(jobs=0, cpu_seconds=0.0, collect_wait_seconds=0.0,
                          d2h_bytes=0, h2d_bytes=0, busy_fallback=0, unsupported_fallback=0,
                          repack_hits=0, repack_misses=0, repack_seconds=0.0,
                          repack_peak_bytes_including_scratch=0, native_borrowed_charge_peak=0)
        for _ in range(int(slots)):
            self._slots.append(dict(
                x=torch.empty(1, self.hidden, dtype=torch.float16, pin_memory=True),
                weights=torch.empty(1, self.max_cold_experts, dtype=torch.float16, pin_memory=True),
                out=torch.empty(1, self.hidden, dtype=torch.float32, pin_memory=True),
                gpu=torch.empty(1, self.hidden, dtype=torch.float32, device=self.device),
                ready=torch.cuda.Event(), retired=torch.cuda.Event(), retired_recorded=False,
                busy=False,
            ))

    @property
    def host_buffer_bytes(self):
        return sum(slot[key].numel() * slot[key].element_size()
                   for slot in self._slots for key in ("x", "weights", "out"))

    @property
    def host_buffer_charge_bytes(self):
        # Lazy import avoids a cache/backend import cycle during CLI setup.
        from .cache import pinned_allocation_charge
        return sum(pinned_allocation_charge(slot[key].numel() * slot[key].element_size())
                   for slot in self._slots for key in ("x", "weights", "out"))

    @property
    def gpu_buffer_bytes(self):
        return sum(slot["gpu"].numel() * slot["gpu"].element_size() for slot in self._slots)

    @property
    def total_private_host_budget_bytes(self):
        return self.host_buffer_charge_bytes + self.repack_budget_bytes

    @staticmethod
    def _native_parts(source):
        projections = []
        # Existing cache order is gate/down/up; native CPU ABI is gate/up/down.
        for pi in (0, 2, 1):
            su, sv, tre = source.parts[pi]
            inn, out = su.numel(), sv.numel()
            tiles = (inn // 16) * (out // 16)
            if not tiles or tre.numel() % tiles or inn % 128 or out % 128:
                raise ValueError("CPU EXL3 source has an unsupported matrix layout")
            projections.append((tre.view(inn // 16, out // 16, -1), su, sv))
        return projections

    @staticmethod
    def _private_parts(source, private):
        parts = []
        for projection in source.parts:
            items = []
            for tensor in projection:
                offset = tensor.data_ptr() - source.host.data_ptr()
                nbytes = tensor.numel() * tensor.element_size()
                if offset < 0 or offset + nbytes > private.numel():
                    raise ValueError("CPU expert tensors do not share the declared record")
                items.append(private[offset:offset + nbytes].view(tensor.dtype).view_as(tensor))
            parts.append(tuple(items))
        obj = type("PrivateCPURecord", (), {})()
        obj.host, obj.parts = private, parts
        return NativeCPUExpertBackend._native_parts(obj)

    def _repack_matrix(self, tensor):
        k = tensor.shape[-1] / 16.0
        group = int(self.ext.exl3_moe_cpu_swizzle_group(k))
        if not group:
            return
        tk, tn, ps = tensor.shape
        temp = torch.empty_like(tensor)
        planar = bool(self.ext.exl3_moe_cpu_planar_layout(k))
        if planar:
            wd = ps // 2
            target = temp.view(torch.int32).view(tn // group, tk, group, wd // 8, 8)
            source = tensor.view(torch.int32).view(tk, tn // group, group, 8, wd // 8).permute(1, 0, 2, 4, 3)
        else:
            target = temp.view(tn // group, tk, group, ps)
            source = tensor.view(tk, tn // group, group, ps).permute(1, 0, 2, 3)
        np.copyto(target.numpy(), source.numpy())
        np.copyto(tensor.numpy(), temp.numpy())

    def _job_parts(self, layer, ids, sources):
        """Only the single CPU worker owns this LRU. Budget includes scratch."""
        native = [self._native_parts(s) for s in sources]
        if not self.repack_budget_bytes:
            return native, False
        keys = [(int(layer), int(e)) for e in ids]
        sizes = {key: source.nbytes for key, source in zip(keys, sources)}
        scratch = max(t.numel() * t.element_size() for mats in native for t, _, _ in mats)
        if sum(sizes.values()) + scratch > self.repack_budget_bytes:
            return native, False
        protected = set(keys)
        missing = sum(size for key, size in sizes.items() if key not in self._repacked)
        while self.repack_bytes + missing + scratch > self.repack_budget_bytes:
            victim = next((key for key in self._repacked if key not in protected), None)
            if victim is None:
                return native, False
            _, _, charge = self._repacked.pop(victim)
            self.repack_bytes -= charge
        outputs = []
        started = time.perf_counter()
        for key, source in zip(keys, sources):
            item = self._repacked.get(key)
            if item is not None:
                self.stats["repack_hits"] += 1
                self._repacked.move_to_end(key)
            else:
                private = torch.empty(source.nbytes, dtype=torch.uint8)  # CPU-only, NOT another pinned arena
                np.copyto(private.numpy(), source.host.numpy())
                matrices = self._private_parts(source, private)
                for tensor, _, _ in matrices:
                    self._repack_matrix(tensor)
                charge = private.untyped_storage().nbytes()
                item = (private, matrices, charge)
                self._repacked[key] = item
                self.repack_bytes += charge
                self.stats["repack_misses"] += 1
            outputs.append(item[1])
        self.stats["repack_seconds"] += time.perf_counter() - started
        self.stats["repack_peak_bytes_including_scratch"] = max(
            self.stats["repack_peak_bytes_including_scratch"], self.repack_bytes + scratch)
        return outputs, True

    def _compute(self, ticket, layer, ids, limit):
        slot = ticket.slot
        slot["ready"].synchronize()
        parts, swizzled = self._job_parts(layer, ids, ticket.sources)
        cols = [[m[p][j] for m in parts] for p in range(3) for j in range(3)]
        handle = self.ext.exl3_moe_cpu_make_layer(*cols, [], [], [], 0, float(limit), int(swizzled))
        try:
            local = torch.arange(len(ids), dtype=torch.int32).view(1, -1)
            started = time.perf_counter()
            self.ext.exl3_moe_cpu_forward(handle, slot["x"], local,
                slot["weights"][:, :len(ids)], slot["out"], self.threads)
            elapsed = time.perf_counter() - started
            with self._lock:
                self.stats["cpu_seconds"] += elapsed
        finally:
            # Avoid an unbounded handle cache retaining evicted native records.
            self.ext.exl3_moe_cpu_free_layer(handle)

    def submit(self, layer, x, expert_ids, weights, sources, limit=10.0):
        ids, sources = tuple(map(int, expert_ids)), tuple(sources)
        if (self._closed or x.device != self.device or x.dtype != torch.float16
                or x.shape != (1, self.hidden) or not x.is_contiguous()
                or weights.shape != (1, len(ids)) or weights.device != self.device
                or not 0 < len(ids) <= self.max_cold_experts or len(ids) != len(sources)
                or len(set(ids)) != len(ids) or torch.cuda.is_current_stream_capturing()):
            self.stats["unsupported_fallback"] += 1
            return None
        with self._lock:
            slot = next((s for s in self._slots if not s["busy"] and
                         (not s["retired_recorded"] or s["retired"].query())), None)
            if slot is None:
                self.stats["busy_fallback"] += 1
                return None
            slot["busy"] = True
        try:
            stream = torch.cuda.current_stream(self.device)
            slot["x"].copy_(x, non_blocking=True)
            # One-row slices remain contiguous at any number of cold experts.
            slot["weights"][:, :len(ids)].copy_(weights.to(torch.float16), non_blocking=True)
            slot["ready"].record(stream)
            ticket = CPUExpertTicket(self, slot, sources)
            with self._lock:
                self.stats["jobs"] += 1
                self.stats["d2h_bytes"] += x.numel() * 2 + len(ids) * 2
                self.stats["native_borrowed_charge_peak"] = max(self.stats["native_borrowed_charge_peak"],
                    sum(getattr(s, "allocation_bytes", s.nbytes) for s in sources))
            ticket.future = self._executor.submit(self._compute, ticket, int(layer), ids, limit)
            return ticket
        except Exception:
            with self._lock:
                slot["busy"] = False
            raise

    def collect(self, ticket, gpu_hot=None):
        if ticket.owner is not self or ticket.consumed:
            raise ValueError("CPU ticket is foreign or already collected")
        slot = ticket.slot
        started = time.perf_counter()
        try:
            ticket.future.result()
            self.stats["collect_wait_seconds"] += time.perf_counter() - started
            if gpu_hot is None:
                result = torch.empty_like(slot["gpu"])
                result.copy_(slot["out"], non_blocking=True)
            else:
                if gpu_hot.device != self.device or gpu_hot.shape != (1, self.hidden):
                    raise ValueError("GPU partial must match the CPU output shape/device")
                slot["gpu"].copy_(slot["out"], non_blocking=True)
                result = gpu_hot + slot["gpu"]
            slot["retired"].record(torch.cuda.current_stream(self.device))
            slot["retired_recorded"] = True
            self.stats["h2d_bytes"] += slot["out"].numel() * 4
            return result
        finally:
            ticket.consumed = True
            ticket.sources = ()
            with self._lock:
                slot["busy"] = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)
        for slot in self._slots:
            if slot["retired_recorded"]:
                slot["retired"].synchronize()
        self._repacked.clear()
        self.repack_bytes = 0
