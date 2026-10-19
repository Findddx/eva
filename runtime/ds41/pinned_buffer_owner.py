"""Independent exact-capacity pinned storage owner; not wired into inference.

Imports are stdlib-only. Adoption lazily validates real Torch host tensors.
Python callers are trusted to obey lexical leases: raw tensor/view escape or
unmanaged DMA is unsupported. No refcount, weakref or allocator-stat heuristic
establishes overwrite safety. Native CPU expert borrowers are unsupported.
"""
from collections import Counter
from dataclasses import dataclass, field
import threading


_KEY = object()
STATES = ('published', 'retiring', 'idle', 'reserved', 'reading', 'complete', 'quarantined')


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name+' must be a positive integer')
    return value


def _shape(tensor):
    storage = tensor.untyped_storage()
    return (tuple(tensor.shape),tuple(tensor.stride()),str(tensor.dtype),str(tensor.device),
            tensor.storage_offset(),tensor.data_ptr(),storage.data_ptr(),storage.nbytes(),tensor.is_conj(),tensor.is_neg())


def _validate_base(base):
    import torch
    if (not isinstance(base,torch.Tensor) or base.device.type != 'cpu' or base.dtype != torch.uint8
            or base.ndim != 1 or base.numel() <= 0 or not base.is_contiguous() or not base.is_pinned()
            or base.storage_offset() != 0 or base.untyped_storage().nbytes() != base.numel()
            or base.is_conj() or base.is_neg()):
        raise ValueError('Require whole original contiguous pinned CPU uint8 base, zero offset, exact storage capacity')
    return _shape(base)


def _validate_views(base, views):
    import torch
    values = tuple(views)
    for view in values:
        if (not isinstance(view,torch.Tensor) or view.device.type != 'cpu'
                or view.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
                or view.untyped_storage().nbytes() != base.numel() or view.is_conj() or view.is_neg()):
            raise ValueError('Every borrowed view must share the whole original pinned allocation')
    return values,tuple(_shape(view) for view in values)


@dataclass(eq=False)
class _Buffer:
    number: int
    base: object
    capacity: int
    charge: int
    signature: tuple
    key: object
    generation: int = 0
    state: str = 'published'
    views: tuple = ()
    view_signatures: tuple = ()
    leases: set = field(default_factory=set)
    result_owners: set = field(default_factory=set)
    fences: set = field(default_factory=set)
    unfenced: bool = False
    reservation: object = None


@dataclass(eq=False)
class _Fence:
    event: object
    stream: object
    buffers: tuple


class Source:
    """Opaque generation/key identity, not permission to access bytes."""
    def __init__(self, token, pool, buffer, key):
        if token is not _KEY:raise TypeError('Sources are minted by the inventory')
        self._pool,self._buffer,self._generation,self._key = pool,buffer,buffer.generation,key


class SourceLease:
    def __init__(self, token, pool, buffer):
        if token is not _KEY:raise TypeError('Leases are minted by the inventory')
        self._pool,self._buffer,self._generation = pool,buffer,buffer.generation
        self._closed = False

    def _check(self):
        self._pool._owner();self._pool._healthy()
        if self._closed or self not in self._buffer.leases or self._generation != self._buffer.generation:
            raise RuntimeError('Source lease is closed/stale')
        self._pool._metadata(self._buffer)

    @property
    def base(self):
        self._check();return self._buffer.base

    @property
    def views(self):
        self._check();return self._buffer.views

    def close(self):
        self._pool._mutate()
        with self._pool._lock:
            if self._closed:return
            self._buffer.leases.remove(self)
            self._closed = True
            self._pool.counters['source_leases_closed'] += 1
            self._pool._idle(self._buffer)

    def __enter__(self):
        self._check();return self

    def __exit__(self,*_exception):
        self.close();return False


class _ReadResult:
    def __init__(self, token, reservation):
        if token is not _KEY:raise TypeError('Read results are minted by a completed fill')
        self.reservation = reservation


class ReadResultOwner:
    """Count an adapter's raw result ownership, without granting byte access."""
    def __init__(self, token, pool, buffer):
        if token is not _KEY:raise TypeError('Result owners are minted by the inventory')
        self._pool,self._buffer,self._generation = pool,buffer,buffer.generation
        self._closed = False

    def close(self):
        self._pool._mutate()
        with self._pool._lock:
            if self._closed:return
            if (self._generation != self._buffer.generation or self not in self._buffer.result_owners
                    or self._buffer not in self._pool._buffers):
                self._pool._poison(self._buffer)
                raise RuntimeError('Result owner is foreign/stale')
            self._buffer.result_owners.remove(self);self._closed = True
            self._pool.counters['result_owners_closed'] += 1
            self._pool._idle(self._buffer)


class ReadReservation:
    """Strong exclusive writer; fill is the only worker-thread entry point."""
    def __init__(self, token, pool, buffer):
        if token is not _KEY:raise TypeError('Reservations are minted by the inventory')
        self._pool,self._buffer,self._generation = pool,buffer,buffer.generation
        self._active,self._started,self._running = True,False,False
        self._future = None
        self._done = threading.Event()

    @property
    def descriptor(self):
        """Immutable numbers only; never grants access to reserved bytes."""
        return dict(buffer_id=self._buffer.number,generation=self._generation,
                    capacity=self._buffer.capacity,allocation_charge=self._buffer.charge)

    def fill(self, reader):
        pool,buffer = self._pool,self._buffer
        with pool._lock:
            pool._healthy()
            pool._reservation(self)
            if self._started or buffer.state != 'reserved':
                pool._poison(buffer)
                raise RuntimeError('Reservation may start exactly one fill')
            pool._metadata(buffer)
            self._started = self._running = True
            buffer.state = 'reading';pool.counters['reads_started'] += 1
        try:
            count = reader(buffer.base)
            if type(count) is not int or count != buffer.capacity:
                raise IOError('Reader must complete the exact record before publication')
            with pool._lock:
                pool._metadata(buffer)
                if buffer.state != 'quarantined':buffer.state = 'complete'
                pool.counters['reads_completed'] += 1
            return _ReadResult(_KEY,self)
        except BaseException:
            with pool._lock:
                pool.counters['reads_failed'] += 1;pool._poison(buffer)
            raise
        finally:
            with pool._lock:self._running = False
            self._done.set()


class PinnedBufferInventory:
    """One CPU metadata owner, independent writer threads and recorded DMA.

    Budget covers all adopted base allocations in every state. close drains
    users but retains bases/charges until explicit release_idle or inventory
    disposal. Releasing an owner is not proof of allocator/OS physical unmap.
    """
    def __init__(self, budget_bytes, *, native_cpu_backend=None):
        if native_cpu_backend is not None:
            raise ValueError('Native CPU backend borrowing is unsupported by this prototype')
        self.budget_bytes = _positive(budget_bytes,'budget_bytes')
        self._thread = threading.get_ident();self._lock = threading.RLock()
        self._buffers,self._keys,self._fences,self._pending = [],{},set(),set()
        self._next_buffer_id = 0
        self._closed = self._closing = self._failed = False
        self._enqueue_active = False
        self.counters = Counter()

    def _owner(self):
        if threading.get_ident() != self._thread:raise RuntimeError('Inventory requires its CPU owner thread')

    def _mutate(self):
        self._owner()
        if self._enqueue_active:raise RuntimeError('Cannot mutate ownership inside an unfinished DMA enqueue batch')

    def _healthy(self):
        if self._closed or self._closing or self._failed:
            raise RuntimeError('Inventory closed/closing/poisoned; discard after draining')

    def _poison(self, buffer):
        with self._lock:
            self._failed = True
            if buffer.state != 'quarantined':self.counters['quarantines'] += 1
            buffer.state = 'quarantined'

    def _metadata(self, buffer):
        if (_shape(buffer.base) != buffer.signature
                or tuple(_shape(view) for view in buffer.views) != buffer.view_signatures):
            self._poison(buffer)
            raise RuntimeError('Owned base/view metadata changed')

    def _source(self, source):
        if (type(source) is not Source or source._pool is not self
                or source._buffer not in self._buffers
                or source._generation != source._buffer.generation or source._key != source._buffer.key):
            self._failed = True
            raise RuntimeError('Foreign/stale source generation')
        self._metadata(source._buffer)
        return source._buffer

    def _reservation(self, reservation):
        if (type(reservation) is not ReadReservation or reservation._pool is not self
                or not reservation._active or reservation not in self._pending
                or reservation._generation != reservation._buffer.generation
                or reservation._buffer.reservation is not reservation):
            self._failed = True
            raise RuntimeError('Foreign/stale/exhausted read reservation')
        return reservation._buffer

    def _idle(self, buffer):
        if (buffer.state == 'retiring' and not buffer.leases and not buffer.result_owners
                and not buffer.fences and not buffer.unfenced):
            buffer.state = 'idle'

    def adopt_published(self, key, base, allocation_charge, *, views=()):
        self._mutate();self._healthy()
        signature = _validate_base(base)
        charge = _positive(allocation_charge,'allocation_charge')
        if charge < base.numel():raise ValueError('Inherited allocation charge cannot be below payload capacity')
        values,signatures = _validate_views(base,views)
        with self._lock:
            if key in self._keys:raise ValueError('Logical owner already registered')
            if any(b.signature[6] == signature[6] for b in self._buffers):
                raise ValueError('Base allocation already owned, including its other views')
            if sum(b.charge for b in self._buffers)+charge > self.budget_bytes:
                raise MemoryError('Adopted allocation exceeds complete inventory budget')
            buffer = _Buffer(self._next_buffer_id,base,base.numel(),charge,signature,key,
                             views=values,view_signatures=signatures)
            self._next_buffer_id += 1
            self._buffers.append(buffer);self._keys[key] = buffer
            self.counters['adopted_buffers'] += 1
            return Source(_KEY,self,buffer,key)

    def release_idle(self, *, capacity=None, max_buffers=None):
        """Forget safe idle base/view owners; do not promise physical unmap.

        No waiting, eviction, donor selection or new allocation occurs. Raw
        tensors retained outside the trusted lease contract remain unsupported.
        The caller may subsequently ask the host allocator to release cache.
        """
        self._mutate()
        if self._closing or self._failed:raise RuntimeError('Cannot release owners during closing/poisoned inventory')
        if capacity is not None:_positive(capacity,'capacity')
        if max_buffers is not None:_positive(max_buffers,'max_buffers')
        if not self._closed:self.reap()
        released=[]
        with self._lock:
            for buffer in tuple(self._buffers):
                if max_buffers is not None and len(released)>=max_buffers:break
                if buffer.state!='idle' or capacity is not None and buffer.capacity!=capacity:continue
                if (buffer.leases or buffer.result_owners or buffer.fences or buffer.unfenced or buffer.reservation is not None
                        or buffer in self._keys.values() or any(r._buffer is buffer for r in self._pending)
                        or any(buffer in f.buffers for f in self._fences)):
                    self._poison(buffer)
                    raise RuntimeError('Idle owner still has a reader/writer/publication obligation')
                self._metadata(buffer)
                released.append(dict(buffer_id=buffer.number,capacity=buffer.capacity,allocation_charge=buffer.charge))
                self._buffers.remove(buffer)
                # Opaque exhausted handles may outlive the inventory entry.
                # Tombstone every Tensor owner, not only the inventory list.
                buffer.base=None;buffer.views=();buffer.signature=();buffer.view_signatures=()
                buffer.key=None;buffer.state='released'
            payload=sum(row['capacity'] for row in released);charge=sum(row['allocation_charge'] for row in released)
            self.counters['buffers_released']+=len(released)
            self.counters['released_capacity_bytes']+=payload;self.counters['released_charge_bytes']+=charge
            return dict(released_buffers=len(released),released_capacity_bytes=payload,
                released_owner_charge_bytes=charge,records=released,
                remaining_charge_bytes=sum(b.charge for b in self._buffers),
                allocator_unmap_or_MemAvailable_increase_claimed=False)

    def reservation_info(self, reservation):
        """Owner-thread metadata for a sealed reader adapter, no base escape."""
        self._mutate();self._healthy()
        with self._lock:
            buffer=self._reservation(reservation);self._metadata(buffer)
            return dict(reservation.descriptor,state=buffer.state,started=reservation._started,
                        running=reservation._running,future_tracked=reservation._future is not None)

    def retain_read_result(self, reservation):
        """Keep adapter-held bytes charged through shutdown until explicitly dropped."""
        self._mutate();self._healthy()
        with self._lock:
            buffer=self._reservation(reservation);self._metadata(buffer)
            if buffer.result_owners:raise ValueError('Reservation already has its result owner')
            owner=ReadResultOwner(_KEY,self,buffer);buffer.result_owners.add(owner)
            self.counters['result_owners_opened'] += 1
            return owner

    def lease(self, source):
        self._mutate();self._healthy()
        with self._lock:
            buffer = self._source(source)
            if buffer.state != 'published':raise RuntimeError('Cannot acquire a new lease after unpublication')
            lease = SourceLease(_KEY,self,buffer);buffer.leases.add(lease)
            self.counters['source_leases_opened'] += 1
            return lease

    def retire(self, source):
        self._mutate();self._healthy()
        with self._lock:
            buffer = self._source(source)
            if buffer.state != 'published':raise RuntimeError('Source is not published')
            del self._keys[buffer.key];buffer.state = 'retiring'
            self._idle(buffer)

    def copy_batch(self, leases, stream, enqueue, event_factory):
        """One immutable completion fence after this stream's last batch copy.

        Caller must enqueue on the supplied stream and establish producer HB.
        Even a failing enqueue can have submitted a partial copy. Never retry
        inference: record completion if possible, poison, and drain on close.
        """
        self._mutate();self._healthy()
        leases = tuple(leases)
        if not leases:raise ValueError('A DMA batch needs source leases')
        buffers = []
        for lease in leases:
            if type(lease) is not SourceLease or lease._pool is not self:raise ValueError('Foreign source lease')
            lease._check()
            if lease._buffer not in buffers:buffers.append(lease._buffer)
        error = value = None
        self._enqueue_active = True
        try:value = enqueue()
        except BaseException as exc:error = exc
        fence_error = None
        try:
            event = event_factory()
            if (any(f.event is event for f in self._fences)
                    or any(not callable(getattr(event,name,None)) for name in ('record','query','synchronize'))):
                raise RuntimeError('Require an independent completion event, not one still referenced by another batch')
            event.record(stream)
            fence = _Fence(event,stream,tuple(buffers))
            self._fences.add(fence)
            for buffer in buffers:buffer.fences.add(fence)
            self.counters['fence_batches'] += 1
        except BaseException as exc:
            fence_error = exc
            for buffer in buffers:buffer.unfenced = True
        finally:self._enqueue_active = False
        if error is not None or fence_error is not None:
            for buffer in buffers:self._poison(buffer)
            if error is not None:
                if fence_error is not None and hasattr(error,'add_note'):
                    error.add_note('Completion event also failed; explicit device synchronization required on close')
                raise error
            raise fence_error
        return value

    def reap(self, *, wait=False, _shutdown=False):
        self._mutate()
        if not _shutdown:self._healthy()
        # GPU query/wait never holds the metadata lock needed by I/O workers.
        for fence in tuple(self._fences):
            try:
                self.counters['fence_waits' if wait else 'fence_queries'] += 1
                if wait:fence.event.synchronize();done = True
                else:done = fence.event.query()
                if type(done) is not bool:raise RuntimeError('Fence query must return a completion bool')
            except BaseException:
                for buffer in fence.buffers:buffer.unfenced = True;self._poison(buffer)
                raise
            if done:
                self._fences.remove(fence)
                for buffer in fence.buffers:buffer.fences.remove(fence);self._idle(buffer)
                self.counters['fences_completed'] += 1

    def try_reserve(self, key, capacity, *, reason='demand', donor=None,
                    admission_enabled=True, protected_keys=()):
        self._mutate();self._healthy();_positive(capacity,'capacity')
        if reason not in ('demand','speculative') or type(admission_enabled) is not bool:
            raise ValueError('Invalid reservation origin/admission policy')
        self.reap()
        with self._lock:
            if key in self._keys:raise ValueError('Logical owner already published/reserved')
            buffer = next((b for b in self._buffers if b.state == 'idle' and b.capacity == capacity),None)
            if buffer is None and donor is not None and reason == 'demand' and admission_enabled:
                candidate = self._source(donor)
                if (candidate.state == 'published' and candidate.capacity == capacity
                        and candidate.key not in protected_keys and not candidate.leases
                        and not candidate.result_owners and not candidate.fences and not candidate.unfenced):
                    del self._keys[candidate.key];candidate.state = 'idle';buffer = candidate
                    self.counters['demand_donor_transfers'] += 1
            if buffer is None:
                self.counters['reservation_misses'] += 1
                return None
            self._metadata(buffer)
            buffer.generation += 1;buffer.key = key;buffer.state = 'reserved'
            buffer.views = ();buffer.view_signatures = ()
            reservation = ReadReservation(_KEY,self,buffer);buffer.reservation = reservation
            self._pending.add(reservation);self._keys[key] = buffer
            self.counters['reservations'] += 1;self.counters['reuse_bytes'] += capacity
            return reservation

    def track_future(self, reservation, future):
        self._mutate()
        from concurrent.futures import Future
        if not isinstance(future,Future):raise TypeError('Require a concurrent.futures.Future')
        with self._lock:
            self._reservation(reservation)
            if reservation._future is not None:raise ValueError('Reservation already has its strong future owner')
            if any(other._future is future for other in self._pending):
                raise ValueError('One future cannot own two read reservations')
            reservation._future = future

    def _release_reservation(self, reservation):
        buffer = self._reservation(reservation)
        self._pending.remove(reservation);reservation._active = False;buffer.reservation = None
        return buffer

    def complete(self, reservation, *, views=()):
        self._mutate()
        with self._lock:
            buffer = self._reservation(reservation);future = reservation._future
        if future is None:raise RuntimeError('Track the write future before consuming its result')
        try:result = future.result()  # Completion is not publication; wait outside lock.
        except BaseException:
            with self._lock:self._poison(buffer)
            raise
        with self._lock:
            self._healthy();self._reservation(reservation)
            try:
                if (type(result) is not _ReadResult or result.reservation is not reservation
                        or buffer.state != 'complete' or reservation._running):
                    raise RuntimeError('Future did not produce this exact completed generation')
                values,signatures = _validate_views(buffer.base,views)
                self._metadata(buffer)
            except BaseException:self._poison(buffer);raise
            self._release_reservation(reservation)
            buffer.views,buffer.view_signatures,buffer.state = values,signatures,'published'
            self.counters['reads_published'] += 1
            return Source(_KEY,self,buffer,buffer.key)

    def cancel(self, reservation):
        self._mutate()
        with self._lock:
            self._reservation(reservation);future = reservation._future
        if future is None:raise RuntimeError('Queued cancellation requires the actual future')
        if not future.cancel():return False
        with self._lock:
            buffer = self._reservation(reservation)
            if reservation._started:self._poison(buffer);raise RuntimeError('Cancelled future already wrote')
            self._release_reservation(reservation);del self._keys[buffer.key]
            buffer.state = 'retiring';self._idle(buffer);self.counters['reads_cancelled'] += 1
            return True

    def discard(self, reservation):
        self._mutate()
        with self._lock:self._reservation(reservation);future = reservation._future
        if future is None:raise RuntimeError('Discard must join the tracked read future')
        try:
            result = future.result()
            if (type(result) is not _ReadResult or result.reservation is not reservation
                    or reservation._running or reservation._buffer.state != 'complete'):
                raise RuntimeError('Discard must join this exact completed writer generation')
        except BaseException:
            with self._lock:self._poison(reservation._buffer)
            raise
        with self._lock:
            buffer = self._release_reservation(reservation);del self._keys[buffer.key]
            buffer.state = 'retiring' if not self._failed else 'quarantined';self._idle(buffer)
            self.counters['reads_discarded'] += 1

    def close(self, *, device_synchronize=None):
        self._mutate()
        if self._closed:return self.status()
        if any(b.leases for b in self._buffers):raise RuntimeError('Close requires every source lease to be released')
        self._closing = True
        try:
            for reservation in tuple(self._pending):
                future = reservation._future
                if future is not None:
                    if not self.cancel(reservation):
                        try:future.result()
                        except BaseException:self._poison(reservation._buffer)
                if reservation._running:
                    reservation._done.wait()  # No pool lock held; writer can finish.
                if reservation._active:
                    self._release_reservation(reservation)
                    buffer = reservation._buffer
                    self._keys.pop(buffer.key,None)
                    if buffer.state != 'quarantined':buffer.state = 'retiring';self._idle(buffer)
            unknown = any(b.unfenced for b in self._buffers)
            if not unknown:
                try:self.reap(wait=True,_shutdown=True)
                except BaseException:unknown = True
            if unknown:
                if not callable(device_synchronize):
                    raise RuntimeError('Unfenced/failed DMA requires explicit device synchronization before close')
                if device_synchronize() is not None:
                    raise RuntimeError('Device synchronization callback must complete successfully and return None')
                self.counters['explicit_device_drains'] += 1
                self._fences.clear()
                for buffer in self._buffers:buffer.fences.clear();buffer.unfenced = False
            self._closed = True;self.counters['close_drains'] += 1
        finally:self._closing = False
        return self.status()

    def status(self):
        self._owner()
        with self._lock:
            return dict(closed=self._closed,failed=self._failed,budget_bytes=self.budget_bytes,
                buffer_count=len(self._buffers),total_capacity_bytes=sum(b.capacity for b in self._buffers),
                total_charge_bytes=sum(b.charge for b in self._buffers),retained_base_owners=True,
                states={state:sum(b.state == state for b in self._buffers) for state in STATES},
                live_leases=sum(len(b.leases) for b in self._buffers),pending_fences=len(self._fences),
                live_result_owners=sum(len(b.result_owners) for b in self._buffers),
                pending_reads=len(self._pending),running_reads=sum(r._running for r in self._pending),
                unfenced_buffers=sum(b.unfenced for b in self._buffers),counters=dict(self.counters),
                records=[dict(buffer_id=b.number,generation=b.generation,state=b.state,capacity=b.capacity,
                              allocation_charge=b.charge,lease_count=len(b.leases),fence_count=len(b.fences),
                              result_owner_count=len(b.result_owners))
                         for b in self._buffers])
