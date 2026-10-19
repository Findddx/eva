"""Opt-in shared NumPy pointer writes with host-DMA/GPU-reader retirement."""
import threading

import numpy as np


def requested(environ):
    value = environ.get("DS41_FAST_POINTER_BIND", "0")
    if value not in ("0", "1"):
        raise ValueError("DS41_FAST_POINTER_BIND must be 0 or 1")
    return value == "1"


class FastPointerTable:
    """One inference CPU thread; CUDA streams may differ but must be ordered.

    Every host write retires the previous copy/use event. A subsequent kernel
    record supersedes its copy event only after the compute stream has waited
    for that copy. Events and the shared pinned allocation live through close.
    """
    def __init__(self, host, layers, event_factory):
        if (str(host.device) != "cpu" or str(host.dtype) != "torch.int64"
                or tuple(host.shape) != (layers, 9, 384) or not host.is_contiguous()
                or not host.is_pinned()):
            raise ValueError("fast pointer table requires contiguous pinned CPU int64 [layers,9,384]")
        self.host = host
        self.view = host.numpy()
        if (self.view.dtype != np.dtype("int64") or not self.view.flags.writeable
                or self.view.__array_interface__["data"][0] != host.data_ptr()):
            raise RuntimeError("NumPy pointer table does not share the original pinned storage")
        self.layers = layers
        self.owner = threading.get_ident()
        self.event_factory = event_factory
        self.events, self.pending = {}, set()
        self.closed = False
        self.counters = dict(binds=0, unbinds=0, dma_records=0, reader_records=0, retire_waits=0)

    def _check(self, layer=None):
        if threading.get_ident() != self.owner:
            raise RuntimeError("fast pointer table requires its inference CPU owner thread")
        if self.closed:
            raise RuntimeError("fast pointer table is closed")
        if layer is not None and not 0 <= layer < self.layers:
            raise IndexError("pointer-table layer is outside its allocation")

    def retire(self, layer):
        self._check(layer)
        if layer in self.pending:
            # Clear only after synchronize succeeds; a failed wait is unsafe.
            self.events[layer].synchronize()
            self.pending.remove(layer)
            self.counters["retire_waits"] += 1

    def bind(self, layer, expert, pointers):
        if not 0 <= expert < 384 or len(pointers) != 9:
            raise ValueError("fast pointer bind requires one expert and nine pointers")
        self.retire(layer)
        self.view[layer, :, expert] = pointers
        self.counters["binds"] += 1

    def unbind(self, layer, expert):
        if not 0 <= expert < 384:
            raise ValueError("pointer expert is outside its allocation")
        self.retire(layer)
        self.view[layer, :, expert] = 0
        self.counters["unbinds"] += 1

    def record(self, layer, stream, *, reader=False):
        self._check(layer)
        event = self.events.get(layer)
        if event is None:
            event = self.event_factory()
            self.events[layer] = event
        event.record(stream)
        self.pending.add(layer)
        self.counters["reader_records" if reader else "dma_records"] += 1

    def close(self):
        if self.closed:
            return
        self._check()
        for layer in tuple(self.pending):
            self.retire(layer)
        self.closed = True

    def status(self):
        return dict(enabled=True, shared_storage=True, shared_pinned_storage=True, owner_thread_only=True,
                    counter_scope="cache_lifetime_including_preload",
                    table_bytes=self.host.numel() * self.host.element_size(),
                    events_created=len(self.events), pending_layers=sorted(self.pending),
                    closed=self.closed, counters=dict(self.counters))
