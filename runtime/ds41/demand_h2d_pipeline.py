"""Default-off actual-demand scheduling; no tensor allocation or CUDA import.

This changes enqueue timing, not weights/kernels or victim policy. The caller
supplies a drain for exceptional paths and retains staged sources through its
original slot-event finally. A successful prepare does not prove DMA/IO overlap.
"""
import threading
import time


COUNTERS = (
    'calls', 'prepared_calls', 'submitted_demands', 'reused_pending',
    'ready_installs', 'ready_bytes', 'streamed_installs', 'streamed_bytes',
    'failures', 'cleanup_cancels', 'cleanup_joins', 'cleanup_discarded_reads',
    'cleanup_discarded_bytes', 'cleanup_errors', 'drains', 'drain_errors',
    'submit_s', 'ready_install_s', 'join_s', 'account_admit_s',
    'stream_install_s', 'prepare_s', 'cleanup_s',
)


def requested(env):
    flag = env.get('DS41_DEMAND_H2D_PIPELINE', '0')
    if flag not in ('0', '1'):
        raise ValueError('DS41_DEMAND_H2D_PIPELINE must be 0 or 1')
    return flag == '1'


def validate_options(enabled, *, cpu_experts=False, adaptive=False,
                     next_chunk=False, retain_host=True):
    if enabled and (cpu_experts or adaptive or next_chunk):
        raise ValueError('demand H2D pipeline excludes CPU experts, adaptive placement and next-chunk prefetch')
    if enabled and not retain_host:
        raise ValueError('demand H2D pipeline requires retained host staging')


def configuration(enabled):
    return dict(enabled=bool(enabled), policy='actual_demand_RAM_first_ordered_NVMe_v1',
                scope='actual_MoE_calls_including_prefill_and_diagnostics',
                new_copy_streams=0, new_GPU_allocations=0, overlap_measured=False)


class DemandH2DPipeline:
    def __init__(self):
        self.owner = threading.get_ident()
        self.totals = dict.fromkeys(COUNTERS, 0)
        self.current = None
        self.quarantined = []
        self.failed = self.closed = False

    def _owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('demand H2D pipeline requires its inference owner thread')

    def validate(self, cache, next_ids=None):
        self._owner()
        if self.failed or self.closed:
            raise RuntimeError('demand H2D pipeline failed/closed; discard this cache')
        if self.current is not None:
            raise RuntimeError('demand H2D pipeline call is already active')
        config = getattr(cache, '_chunk_prefetch_config', None)
        validate_options(True, cpu_experts=cache.cpu_backend is not None,
                         adaptive=getattr(cache, 'adaptive_gpu_enabled', False),
                         next_chunk=(config is not None and config.enabled
                                     or getattr(cache, '_chunk_prefetch', None) is not None
                                     or next_ids is not None), retain_host=cache.retain_host)

    def begin(self, cache, layer, ids, staged, drain):
        self.validate(cache)
        if self.current is not None:
            raise RuntimeError('demand H2D pipeline call is already active')
        call = DemandH2DCall(self, cache, layer, tuple(ids), staged, drain)
        self.current = call
        call.stat('calls')
        return call

    def close(self):
        self._owner()
        if self.current is not None:
            raise RuntimeError('close demand H2D pipeline after its active call finishes')
        for call in tuple(self.quarantined):
            if not call.cleanup():
                raise RuntimeError('demand H2D pipeline drain failed; retain source owners')
            self.quarantined.remove(call)
            call.release()
        self.closed = True

    def status(self, stats):
        self._owner()
        return dict(configuration(True), failed=self.failed, closed=self.closed, active=self.current is not None,
                    quarantined_calls=len(self.quarantined),
                    counters=dict(self.totals),
                    counters_since_reset={name: stats.get('demand_h2d_' + name, 0) for name in COUNTERS},
                    scoped_counter_scope='since_latest_cache_reset_stats')


class DemandH2DCall:
    def __init__(self, runtime, cache, layer, ids, staged, drain):
        self.runtime, self.cache, self.layer, self.ids = runtime, cache, layer, ids
        self.staged, self.drain = staged, drain
        self.futures, self.accounted, self.sources = {}, set(), []
        self.failed = self.drained = self.closed = False

    def stat(self, name, value=1):
        key = 'demand_h2d_' + name
        self.cache.stats[key] = self.cache.stats.get(key, 0) + value
        self.runtime.totals[name] += value

    def install(self):
        cache, layer = self.cache, self.layer
        if cache._active_keys != {(layer, e) for e in self.ids}:
            raise RuntimeError('pipeline requires the complete current active union')
        missing, ram_ready = [], []
        for expert in self.ids:
            key = layer, expert
            if key in cache.frozen or key in cache.vram:
                continue
            if key in cache.ram:
                ram_ready.append(key)
                self.sources.append(cache.ram[key])
            else:
                missing.append(expert)
        missing_set = set(missing)
        started = time.perf_counter()
        t0 = time.perf_counter()
        try:
            # Preserve every original demand submission/headroom check before
            # installing anything, including reuse of existing legacy futures.
            for expert in missing:
                key = layer, expert
                existing = key in cache._pending
                self.futures[key] = cache._submit_read(key)
                self.stat('reused_pending' if existing else 'submitted_demands')
        finally:
            elapsed = time.perf_counter() - t0
            cache.stats['io_submit'] += elapsed
            self.stat('submit_s', elapsed)
        t0 = time.perf_counter()
        for expert in self.ids:
            if expert in missing_set:
                continue
            key = layer, expert
            needs_copy = key not in cache.frozen and key not in cache.vram
            cache._install_if_absent(layer, expert, True, True)
            if needs_copy:
                self.stat('ready_installs')
                self.stat('ready_bytes', cache._fields(layer, expert)[1])
        self.stat('ready_install_s', time.perf_counter() - t0)
        for expert in missing:
            key = layer, expert
            t0 = time.perf_counter()
            pinned, elapsed = cache._take_pending_read(key)
            joined = time.perf_counter()
            self.stat('join_s', joined - t0)
            # Keep both Future and lexical source until accounting/admission/
            # install complete. Admission failure must never trigger a reread.
            self.sources.append(pinned)
            self.accounted.add(key)
            cache._account_read(pinned, elapsed)
            cache.stats['nvme'] += 1
            cache.stats['nvme_bytes'] += pinned.nbytes
            cache._ram_insert(key, pinned)
            self.staged[expert] = pinned
            admitted = time.perf_counter()
            cache.stats['io_wait'] += admitted - t0
            self.stat('account_admit_s', admitted - joined)
            t1 = time.perf_counter()
            cache._install_if_absent(layer, expert, False, True, pinned)
            self.stat('stream_install_s', time.perf_counter() - t1)
            self.stat('streamed_installs')
            self.stat('streamed_bytes', pinned.nbytes)
            del self.futures[key]
        # Active keys were protected throughout; nonactive victim order has
        # not changed. Restore the successful original call's final LRU order.
        for key in ram_ready:
            cache.ram.move_to_end(key)
        for expert in self.ids:
            key = layer, expert
            if key in cache.vram:
                cache.vram.move_to_end(key)
        self.stat('prepared_calls')
        self.stat('prepare_s', time.perf_counter() - started)

    def cleanup(self, original=None):
        t0, ok = time.perf_counter(), True
        for key, future in tuple(self.futures.items()):
            try:
                if future.cancel():
                    self.stat('cleanup_cancels')
                else:
                    pinned, elapsed = future.result()
                    self.sources.append(pinned)
                    self.stat('cleanup_joins')
                    if key not in self.accounted:
                        self.accounted.add(key)
                        self.cache._account_read(pinned, elapsed)
                        self.stat('cleanup_discarded_reads')
                        self.stat('cleanup_discarded_bytes', pinned.nbytes)
            except BaseException as exc:
                self.stat('cleanup_errors')
                if not future.done():
                    ok = False  # Interrupted running writer remains owned.
                if original is not None and hasattr(original, 'add_note'):
                    original.add_note('pipeline read cleanup: ' + type(exc).__name__)
            finally:
                if future.done():
                    if self.cache._pending.get(key) is future:
                        del self.cache._pending[key]
                    del self.futures[key]
        if not self.drained:
            try:
                self.drain()  # Only exceptional paths, never the successful fast path.
                self.drained = True
                self.stat('drains')
            except BaseException as exc:
                ok = False
                self.stat('drain_errors')
                if original is not None and hasattr(original, 'add_note'):
                    original.add_note('pipeline GPU drain: ' + type(exc).__name__)
        self.stat('cleanup_s', time.perf_counter() - t0)
        return ok and not self.futures and self.drained

    def abort(self, original):
        if not self.failed:
            self.stat('failures')
        self.failed = self.runtime.failed = self.cache._demand_h2d_failed = True
        self.cleanup(original)

    def release(self):
        self.sources.clear()
        self.staged = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.failed and (self.futures or not self.drained):
            self.runtime.quarantined.append(self)
        else:
            self.release()
        self.runtime.current = None
