"""Isolated sealed reader adapter; no inference integration or native CPU use.

The real executor Future carries the inventory's sealed fill result. Cache-like
consumers receive the original (PinnedExpert, elapsed) tuple only after owner
publication. Raw tensors must stay inside counted source leases; Python tensor
escape remains a trusted-caller contract, never a refcount heuristic.
"""
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
import math
import threading


def _duration(value, label):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid nonnegative finite read timing: '+label)
    return value


def _record(result, base, descriptor):
    """Validate the cache tuple ABI without importing its CUDA cache module."""
    from .direct_io import DIRECT_READ_STATS
    from .pinned_buffer_owner import _validate_views
    if type(result) is not tuple or len(result) != 2:
        raise ValueError('Reader must return the original (PinnedExpert, elapsed) tuple')
    pinned, elapsed = result
    if (getattr(pinned, 'host', None) is not base
            or type(getattr(pinned, 'nbytes', None)) is not int
            or pinned.nbytes != descriptor['capacity']
            or type(getattr(pinned, 'allocation_bytes', None)) is not int
            or pinned.allocation_bytes != descriptor['allocation_charge']):
        raise ValueError('Read record must borrow this exact whole base and inherited charge')
    parts = getattr(pinned, 'parts', None)
    if (type(parts) not in (tuple, list) or len(parts) != 3
            or any(type(part) not in (tuple, list) or len(part) != 3 for part in parts)):
        raise ValueError('Read record needs exactly three projections with three borrowed views each')
    views = tuple(view for part in parts for view in part)
    _validate_views(base, views)
    times = tuple(_duration(getattr(pinned, name, None), name) for name in ('alloc_s', 'pread_s', 'views_s'))
    _duration(elapsed, 'elapsed')
    if times[0] != 0:
        raise ValueError('A reused whole base must report zero new allocation time')
    stats = getattr(pinned, 'direct_io_stats', None)
    if (type(stats) is not dict or not set(stats).issubset(DIRECT_READ_STATS)
            or any(type(value) is not int or value < 0 for value in stats.values())):
        raise ValueError('Invalid direct-reader counter sidecar')
    signature = (id(pinned), id(base), pinned.nbytes, pinned.allocation_bytes,
                 tuple(id(view) for view in views), times, elapsed, tuple(sorted(stats.items())))
    return views, signature


class PendingRead:
    """One reservation, one real Future and at-most-once main-thread accounting.

    Use submit only for an already granted exact-capacity demand reservation.
    There is no donor selection, host guard bypass or speculative reclaim here.
    Keep the adapter reachable through interruption, then cancel/discard or
    close the inventory. No cache call site uses this experimental API.
    """
    def __init__(self, inventory, reservation, read_into):
        self._thread = threading.get_ident()
        info = inventory.reservation_info(reservation)
        if info['state'] != 'reserved' or info['started'] or info['future_tracked']:
            raise ValueError('PendingRead needs a fresh unstarted untracked reservation')
        if not callable(read_into):raise TypeError('read_into must be callable')
        self._inventory, self._reservation = inventory, reservation
        self._descriptor = dict(reservation.descriptor)
        self._result_owner = inventory.retain_read_result(reservation)
        self._reader, self._future = read_into, None
        self._record = self._views = self._signature = None
        self._source = self._result_lease = None
        self._state = 'pending'
        self._account_attempted = self._accounted = False
        self._account_error = self._account_origin = None

    @classmethod
    def submit(cls, inventory, reservation, executor, read_into):
        if not isinstance(executor, ThreadPoolExecutor):
            raise TypeError('Require an actual ThreadPoolExecutor, not a proxy future factory')
        value = cls(inventory, reservation, read_into)
        future = None
        try:
            try:
                future = executor.submit(reservation.fill, value._fill)
            finally:
                if future is not None:
                    # Retain and track even if the writer already completed or
                    # the owner is interrupted after submit returned.
                    value._future = future
                    inventory.track_future(reservation, future)
        except BaseException as error:
            value._state = 'submit_failed'
            # A submit interrupted before returning its Future has an unknown
            # enqueue outcome. Never fabricate a future or release its hold.
            # The caller can retain this adapter, close the inventory (which
            # joins any untracked active fill), then explicitly clear it.
            error.pending_read = value
            raise
        return value

    def _owner(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError('PendingRead requires its inventory CPU owner thread')

    def _fill(self, base):
        result = self._reader(base)
        views, signature = _record(result, base, self._descriptor)
        self._record, self._views, self._signature = result, views, signature
        return self._descriptor['capacity']

    @property
    def future(self):
        """The actual strongly held sealed-result Future, never a fake tuple Future."""
        self._owner();return self._future

    def done(self):
        self._owner();return self._future.done()

    def cancelled(self):
        self._owner();return self._future.cancelled()

    def _wait(self, timeout):
        try:return self._future.result(timeout=timeout)
        except BaseException as error:
            if self._future.cancelled():
                if self._state == 'pending':self.cancel()
            elif self._future.done() and self._future.exception(timeout=0) is error:
                # Only a real terminal worker error triggers quarantine. A
                # Timeout/KeyboardInterrupt in the owner leaves a running or
                # already-successful real future and reservation untouched.
                self._state = 'failed'
                try:self._inventory.complete(self._reservation)
                except BaseException:pass  # Preserve the original worker error.
            raise

    def _validate_published(self):
        if self._state != 'published' or self._account_error is not None:
            raise RuntimeError('Read adapter is not a healthy published owner')
        base = self._result_lease.base  # Also checks all owned view metadata.
        _, signature = _record(self._record, base, self._descriptor)
        if signature != self._signature:
            raise RuntimeError('Published record/timing/counter sidecar changed')

    def result(self, timeout=None):
        self._owner()
        if self._state == 'published':
            self._validate_published();return self._record
        if self._state == 'cancelled':raise CancelledError()
        if self._state != 'pending':raise RuntimeError('Read was discarded, handed off or failed')
        self._wait(timeout)
        source = self._inventory.complete(self._reservation, views=self._views)
        self._source = source
        self._result_lease = self._inventory.lease(source)
        self._state = 'published'
        self._validate_published()
        return self._record

    def lease(self):
        """Borrow this published generation; callers must close their own lease."""
        self._owner();self._validate_published()
        return self._inventory.lease(self._source)

    def account_once(self, callback, *, prefetched=False):
        self._owner()
        if type(prefetched) is not bool or not callable(callback):
            raise TypeError('Require a callable accounting sink and bool origin')
        if self._account_attempted:
            if self._account_error is not None:
                raise RuntimeError('Accounting previously failed; do not retry partial counters')
            if prefetched != self._account_origin:
                raise ValueError('Read origin changed after one-time accounting')
            return False
        pinned, elapsed = self.result()
        self._account_attempted, self._account_origin = True, prefetched
        try:callback(pinned, elapsed, prefetched=prefetched)
        except BaseException as error:
            # Retaining the exception/traceback would retain this frame's raw
            # PinnedExpert after discard, outside the counted result hold.
            self._account_error = type(error).__name__
            raise
        self._accounted = True
        return True

    def _forget_bytes(self):
        self._record = self._views = self._signature = self._reader = None
        if self._result_owner is not None:
            self._result_owner.close();self._result_owner = None

    def handoff(self):
        """Transfer the published opaque Source to a trusted RAM/staged owner.

        The receiver must keep publication/leases until every byte consumer is
        done. No adapter/raw tuple may be reused as an uncounted later reader.
        """
        self._owner();self._validate_published()
        if not self._accounted:raise RuntimeError('Account this read before handoff')
        source, lease = self._source, self._result_lease
        self._forget_bytes();self._source = self._result_lease = None
        lease.close();self._state = 'handed_off'
        return source

    def cancel(self):
        self._owner()
        if self._state == 'cancelled':return True
        if self._state != 'pending':return False
        if not self._inventory.cancel(self._reservation):return False
        self._forget_bytes();self._state = 'cancelled'
        return True

    def discard(self, timeout=None):
        """Drop without accounting; running cancellation never frees the base."""
        self._owner()
        if self._state in ('discarded', 'cancelled'):return
        if self._state == 'published':
            poisoned = self._inventory.status()['failed']
            if not poisoned:self._inventory.retire(self._source)
            lease = self._result_lease
            self._forget_bytes();self._source = self._result_lease = None
            lease.close();self._state = 'quarantined' if poisoned else 'discarded'
            return
        if self._state != 'pending':raise RuntimeError('Failed/handoff read needs its owner shutdown protocol')
        if self.cancel():return
        self._wait(timeout)
        self._forget_bytes()  # Drop adapter raw views before permitting idle.
        self._inventory.discard(self._reservation);self._state = 'discarded'

    def clear_after_inventory_close(self):
        """Clear a failed writer's extra references only after the owner drained."""
        self._owner();status = self._inventory.status()
        if (not status['closed'] or status['pending_reads'] or status['running_reads']
                or status['pending_fences'] or status['unfenced_buffers'] or status['live_leases']):
            raise RuntimeError('Inventory must close and drain every reader/writer first')
        self._forget_bytes();self._source = self._result_lease = None
        self._state = 'closed'

    def status(self):
        self._owner();owner = self._inventory.status()
        return dict(state=self._state,descriptor=dict(self._descriptor),
            actual_future_retained=isinstance(self._future, Future),
            future_done=self._future.done() if self._future is not None else None,
            future_cancelled=self._future.cancelled() if self._future is not None else None,
            raw_record_retained=self._record is not None,result_owner_retained=self._result_owner is not None,
            internal_result_lease=self._result_lease is not None,
            account_attempted=self._account_attempted,accounted=self._accounted,
            accounting_failed=self._account_error is not None,prefetched=self._account_origin,
            new_allocation_charge_bytes=0,owner_total_charge_bytes=owner['total_charge_bytes'],
            owner_pending_reads=owner['pending_reads'],owner_failed=owner['failed'],
            production_integrated=False,native_cpu_supported=False)
