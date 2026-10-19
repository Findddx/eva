"""Opt-in pinned-backing migration for completed per-layer promotions.

Import/configuration are stdlib-only. One compute owner publishes a borrowed
host allocation under its new expert key only after enqueueing D2H and two
distinct completion events. Ready events and H2D reader events are immutable
generations; slot events may subsequently be recorded again by the cache.
"""
import hashlib
import os
from pathlib import Path
import threading

FLAG = 'DS41_LAYER_PROMOTION_HOST_BACKING'
MAX_ALLOCATIONS, MAX_CHARGE_BYTES = 4, 64*1024**2
COUNTERS = ('attempts','swaps','d2h_calls','d2h_bytes','reader_events','ready_waits','reader_waits',
            'polls','retired_leases','retired_allocations','promotion_event_leases_retired',
            'prefetch_skips','skip_missing','skip_pending','skip_geometry','skip_capacity','failures','close_drains',
            'foreign_reader_drain_attempts','foreign_reader_stream_drains',
            'migration_allocations_peak','migration_charge_bytes_peak','reader_allocations_peak',
            'reader_charge_bytes_peak','all_allocations_peak','all_charge_bytes_peak')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def requested(environ=None):
    env = os.environ if environ is None else environ
    value = env.get(FLAG, '0')
    require(value in ('0','1'), FLAG+' must be0 or1')
    require(value != '1' or env.get('DS41_LAYER_PROMOTION', '0') == '1', FLAG+' requires DS41_LAYER_PROMOTION=1')
    return value == '1'


def configuration(environ=None):
    if not requested(environ):
        return None
    return dict(protocol='per_layer_async_pinned_backing_exchange_v1', enabled=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        maximum_pending_migration_allocations=MAX_ALLOCATIONS, maximum_pending_migration_charge_bytes=MAX_CHARGE_BYTES,
        migration_budget_excludes_ordinary_H2D_reader_leases=True,
        reuse_existing_pinned_allocation=True, new_pinned_allocations=0, new_GPU_slots=0,
        independent_immutable_ready_and_reader_events=True, slot_retirement_covers_D2H=True,
        next_overwrite_waits_actual_H2D_readers=True, poll_at_every_layer_boundary=True,
        active_decode_legacy_prefetch='skip_entire_prefetch_gpu_including_disk_predictions',
        CPU_experts_supported=False, adaptive_supported=False, next_chunk_supported=False,
        lookahead_supported=False, RAM_payload_and_charge_unchanged_on_success=True,
        GPU_validated=False, full_model_validated=False, overlap_measured=False)


class HostBackingMigration:
    """Trusted inference owner; raw host views must not escape to CPU tickets.

    Original untagged pinned sources may be copied only on the compute stream
    while active. Migrated sources additionally carry a ready event and explicit
    H2D reader generations; their H2D may wait on another stream. In either case
    legacy GPU/disk prediction is skipped throughout active decode.
    """
    def __init__(self, promotion):
        self.promotion = promotion
        self.cache = promotion.cache
        self.identity = configuration()
        require(self.identity is not None, 'Construct host backing only for explicit opt-in')
        self.thread = threading.get_ident()
        self.closed = self.failed = False
        self.leases = []
        self.counters = dict.fromkeys(COUNTERS, 0)

    def _inc(self, name, value=1):
        self.counters[name] += value
        key = 'layer_promotion_host_backing_'+name
        self.cache.stats[key] = self.cache.stats.get(key, 0)+value

    def _observe_pending(self):
        for kind, prefix in (('D2H','migration'),('reader','reader'),(None,'all')):
            allocations = self._allocations(kind=kind)
            for suffix, value in (('allocations_peak',len(allocations)),('charge_bytes_peak',sum(allocations.values()))):
                name = prefix+'_'+suffix
                self.counters[name] = max(self.counters[name],value)
                key = 'layer_promotion_host_backing_'+name
                self.cache.stats[key] = max(self.cache.stats.get(key,0),value)

    def poison(self):
        if not self.failed:
            self._inc('failures')
        self.failed = True
        self.promotion.poison()

    def _check(self, *, stream=True):
        if threading.get_ident() != self.thread or self.closed or self.failed:
            raise RuntimeError('Host-backing owner failed/closed or changed thread')
        self.promotion._check(stream=stream)
        cache = self.cache
        if (not self.promotion.active or not cache.retain_host or cache.cpu_backend is not None
                or cache.adaptive_gpu_enabled
                or getattr(cache, '_chunk_prefetch', None) is not None
                or getattr(getattr(cache, '_chunk_prefetch_config', None), 'enabled', False)):
            raise RuntimeError('Host backing requires active retained-host GPU-only promotion without adaptive/prefetch')
        import torch
        with torch.cuda.device(cache.packed.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Host backing does not support CUDA graph capture')

    def _allocations(self, *, kind=None):
        allocations = {}
        for lease in self.leases:
            if kind is None or lease['kind']==kind:
                allocations[lease['allocation']] = lease['charge']
        return allocations

    def _retire(self, lease):
        tag = lease.get('tag')
        if lease['kind'] == 'reader' and tag is not None:
            tag['readers'][:] = [reader for reader in tag['readers'] if reader is not lease]
        self._inc('retired_leases')

    def poll(self, *, stream=True):
        try:
            self._check(stream=stream)
            self._inc('polls')
            before = self._allocations()
            completed = [lease for lease in self.leases if lease['recorded'] and lease['event'].query()]
            for lease in completed:self._retire(lease)
            retired = {id(lease) for lease in completed}
            self.leases = [lease for lease in self.leases if id(lease) not in retired]
            self._inc('retired_allocations', len(set(before)-set(self._allocations())))
        except BaseException:
            self.poison()
            raise

    def poll_promotion_events(self, leases):
        try:
            done = [lease for lease in leases if lease[1].query()]
            self._inc('promotion_event_leases_retired', len(done))
            retired = {id(lease) for lease in done}
            return [lease for lease in leases if id(lease) not in retired]
        except BaseException:
            self.poison()
            raise

    def skip_prefetch(self):
        try:
            self._check()
            self._inc('prefetch_skips')
        except BaseException:
            self.poison()
            raise

    def _views(self, hot, cold, hot_slot, cold_slot):
        """Read metadata only; no host payload is read or cloned."""
        import torch
        cache = self.cache
        source = cache.ram.get(hot)
        if source is None or cold in cache.ram:
            self._inc('skip_missing');return None
        if (hot in cache._pending or cold in cache._pending or cache._active_keys
                or hot in getattr(cache, '_prefetched', ())):
            self._inc('skip_pending');return None
        fields = cache._fields(*cold)
        nbytes, host = int(fields[1]), source.host
        if (cache._sig(*hot) != cache._sig(*cold) or hot_slot.storage.numel() != cold_slot.storage.numel()
                or int(cache._fields(*hot)[1]) != nbytes or source.nbytes != nbytes
                or host.dtype != torch.uint8 or host.device.type != 'cpu' or host.ndim != 1
                or not host.is_contiguous() or not host.is_pinned() or host.numel() != nbytes
                or host.is_conj() or host.is_neg() or host.requires_grad
                or host.storage_offset() != 0 or type(source.allocation_bytes) is not int
                or source.allocation_bytes < nbytes
                or cold_slot.storage.dtype != torch.uint8 or cold_slot.storage.ndim != 1
                or not cold_slot.storage.is_contiguous() or cold_slot.storage.numel() < nbytes
                or cold_slot.storage.is_conj() or cold_slot.storage.is_neg() or cold_slot.storage.requires_grad
                or cold_slot.storage.device != cache.packed.device
                or hot_slot.storage.device != cold_slot.storage.device):
            self._inc('skip_geometry');return None
        for projection in range(3):
            base = 4+projection*11
            sizes, offsets = fields[base+4:base+7], fields[base+8:base+11]
            if any(type(size) is not int or type(offset) is not int or offset < 0 or size < 0
                   or offset+size > nbytes or (host.data_ptr()+offset)%2 or size%2
                   for size, offset in zip(sizes, offsets)):
                self._inc('skip_geometry');return None
        backed = cache._pinned_views(host, fields)
        backed.allocation_bytes = source.allocation_bytes
        allocation = host.untyped_storage().data_ptr()
        if (backed.host is not host or backed.nbytes != nbytes or
                any(part.untyped_storage().data_ptr() != allocation for parts in backed.parts for part in parts)):
            raise RuntimeError('Host-backing views must borrow the complete original allocation')
        return source, backed, allocation

    def exchange(self, hot, cold, hot_slot, cold_slot, slot_retire_event):
        """Called before GPU dictionary ownership changes, after a whole Block.

        A false result skips only host migration. The old GPU promotion still
        proceeds. Any error after unpublication is fatal; corrupted hot bytes
        are never republished or silently restored from an unverified source.
        """
        try:
            self._check();self._inc('attempts')
            values = self._views(hot, cold, hot_slot, cold_slot)
            if values is None:return False
            source, backed, allocation = values
            allocations = self._allocations(kind='D2H')
            if allocation not in allocations and (len(allocations) >= MAX_ALLOCATIONS
                    or sum(allocations.values())+source.allocation_bytes > MAX_CHARGE_BYTES):
                self._inc('skip_capacity');return False
            prior = getattr(source, '_promotion_backing_state', None)
            if prior is not None and (prior['owner'] is not self or prior['host'] is not source.host):
                raise RuntimeError('Foreign/stale host-backing allocation state')
            ready = self.promotion.event_factory()
            prior_readers = () if prior is None else tuple(prior['readers'])
            if any(not reader['recorded'] for reader in prior_readers):
                raise RuntimeError('Cannot overwrite an unfenced H2D reader')
            if (ready is slot_retire_event or prior is not None and
                    (ready is prior['ready'] or slot_retire_event is prior['ready']
                     or any(ready is reader['event'] or slot_retire_event is reader['event'] for reader in prior_readers))):
                raise RuntimeError('Host-ready generation must be distinct from mutable slot event')
            tag = dict(owner=self, host=source.host, ready=ready, readers=[])
            backed._promotion_backing_state = tag
            lease = dict(kind='D2H', allocation=allocation, charge=source.allocation_bytes,
                source=source, backed=backed, hot_slot=hot_slot, cold_slot=cold_slot,
                event=ready, slot_retire_event=slot_retire_event, stream=self.promotion.stream,
                recorded=False, tag=tag)
            self.leases.append(lease)  # Before waits/copy/record can partially enqueue.
            self._observe_pending()
            cache = self.cache
            del cache.ram[hot]
            cache.ram_nbytes -= source.nbytes
            cache.ram_charge_bytes -= source.allocation_bytes
            cache._prefetched.discard(hot)
            if prior is not None:
                self.promotion.stream.wait_event(prior['ready'])
                self._inc('ready_waits')
                for reader in prior_readers:
                    self.promotion.stream.wait_event(reader['event'])
                    self._inc('reader_waits')
            source.host.copy_(cold_slot.storage[:source.nbytes], non_blocking=True)
            self._inc('d2h_calls');self._inc('d2h_bytes', source.nbytes)
            ready.record(self.promotion.stream)
            slot_retire_event.record(self.promotion.stream)
            lease['recorded'] = True
            cache.ram[cold] = backed
            cache.ram_nbytes += backed.nbytes
            cache.ram_charge_bytes += backed.allocation_bytes
            cache._prefetched.discard(cold)
            self._inc('swaps')
            return True
        except BaseException:
            self.poison()
            raise

    def blit(self, slot, pinned, stream, enqueue):
        """Actual H2D enqueue; a migrated source gets a fresh reader fence."""
        try:
            self._check(stream=False)
            if self.promotion.current_stream() != stream:
                raise RuntimeError('Host-backing H2D must report its actual enqueue stream')
            if slot.storage.device != self.cache.packed.device:
                raise RuntimeError('Host-backing H2D requires its original cache device')
            tag = getattr(pinned, '_promotion_backing_state', None)
            if tag is None:
                if stream != self.promotion.stream:
                    raise RuntimeError('Original pinned sources require the active compute stream; prediction is disabled')
                enqueue()
                return
            if tag['owner'] is not self or tag['host'] is not pinned.host:
                raise RuntimeError('Foreign/stale host-backing source')
            self.poll(stream=False)
            event = self.promotion.event_factory()
            if event is tag['ready'] or any(event is reader['event'] for reader in tag['readers']):
                raise RuntimeError('H2D reader event generation must be independent and immutable')
            lease = dict(kind='reader', allocation=pinned.host.untyped_storage().data_ptr(),
                charge=pinned.allocation_bytes, source=pinned, slot=slot, tag=tag,
                stream=stream, event=event, recorded=False)
            self.leases.append(lease);tag['readers'].append(lease)
            self._observe_pending()
            stream.wait_event(tag['ready']);self._inc('ready_waits')
            enqueue()
            event.record(stream);lease['recorded'] = True
            self._inc('reader_events')
        except BaseException:
            self.poison()
            raise

    def close_after_compute_drain(self):
        if self.closed:return
        if threading.get_ident() != self.thread:
            self.poison();raise RuntimeError('Close host backing on its original CPU owner')
        try:
            seen = [self.promotion.stream]
            for lease in self.leases:
                stream = lease['stream']
                if not any(stream == previous for previous in seen):
                    self._inc('foreign_reader_drain_attempts')
                    stream.synchronize();seen.append(stream)
                    self._inc('foreign_reader_stream_drains')
            for lease in self.leases:self._retire(lease)
            self._inc('retired_allocations', len(self._allocations()))
            self.leases.clear()
            self._inc('close_drains');self.closed = True
        except BaseException:
            self.poison()
            raise

    def status(self):
        allocations = self._allocations()
        migrations = self._allocations(kind='D2H')
        readers = self._allocations(kind='reader')
        return dict(configuration=dict(self.identity), failed=self.failed, closed=self.closed,
            counters=dict(self.counters), pending_leases=len(self.leases),
            pending_allocations=len(allocations), pending_allocation_charge_bytes=sum(allocations.values()),
            pending_migration_allocations=len(migrations),pending_migration_charge_bytes=sum(migrations.values()),
            pending_reader_allocations=len(readers),pending_reader_charge_bytes=sum(readers.values()),
            pending_reader_events=sum(lease['kind']=='reader' for lease in self.leases),
            pending_D2H_events=sum(lease['kind']=='D2H' for lease in self.leases),
            source_and_destination_owners_retained=True,
            actual_pending_charge_is_not_extra_RAM_membership=True,
            construction_compute_drain_performed_by_parent_close=True,
            counter_scope='Active decode only; D2H bytes count successful enqueue returns, not physical PCIe or overlap',
            GPU_validated=False, full_model_validated=False)
