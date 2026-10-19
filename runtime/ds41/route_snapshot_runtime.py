"""Default-off route snapshot owner/binding protocol; no CUDA on import.

No production numerical certification is implied. Private members are trusted
in-process internals, not a sandbox against malicious Python or raw GPU writes.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import threading
import weakref


_KEY=object()
COUNTERS=('scope_attempts','eligible_windows','snapshot_cpu_readbacks','route_D2H_bytes',
    'private_D2D_bytes','unchecked_attempts','unchecked_calls','fallback_windows',
    'fallback_calls','failures','legacy_cache_cpu_readbacks')


def requested(environ):
    value=environ.get('DS41_ROUTE_SNAPSHOT','0')
    if value not in ('0','1'):raise ValueError('DS41_ROUTE_SNAPSHOT must be 0 or 1')
    return value=='1'


def configuration(enabled):
    directory=Path(__file__).parent
    return dict(requested=bool(enabled),protocol='private_versioned_routes_v1',
        eligible='route-aware strict GPU-only native512 top6 B1/flattened M1..128; actual span1..64',
        source_sha256={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                       for name in ('route_snapshot.py','route_snapshot_runtime.py','pointer_table.py')},
        pointer_table_dependency=dict(required=bool(enabled),flag='DS41_FAST_POINTER_BIND',
            protocol='same_cache_FastPointerTable_host_DMA_and_GPU_reader_retirement_v1',
            actual_owner_checked_at_scope_and_each_private_span=True),
        source_mutation_contract='Consumed routes private/frozen; source flat/weights immutable; versioned writes rejected, untracked/raw writes unsupported',
        numerical_GPU_gate_passed=False,full_model_gate_passed=False)


def _identity(tensor):
    try:version=tensor._version
    except RuntimeError:version=None  # inference tensor: no counter to invent.
    storage=tensor.untyped_storage()
    return (tuple(tensor.shape),tuple(tensor.stride()),str(tensor.dtype),str(tensor.device),
        tensor.storage_offset(),tensor.data_ptr(),storage.data_ptr(),storage.nbytes(),version,
        tensor.is_conj(),tensor.is_neg())


def _slice_identity(tensor,start,end):
    # This creates only an ordinary exact view, never a GPU copy or value read.
    return _identity(tensor[start:end])


class RouteSnapshotRuntime:
    def __init__(self,cache):
        self._cache=weakref.ref(cache)
        self._owner=threading.current_thread()
        self._active=None;self._failed=False;self._pointer_table=None
        self.counters={key:0 for key in COUNTERS};self.reasons=Counter()

    def _bump(self,name,value=1):
        self.counters[name]+=value
        cache=self._cache()
        if cache is not None and hasattr(cache,'stats'):
            key='route_snapshot_'+name;cache.stats[key]=cache.stats.get(key,0)+value

    def _owner_check(self):
        if threading.current_thread() is not self._owner:raise RuntimeError('route runtime requires its CPU owner thread')
        if self._failed:raise RuntimeError('route runtime poisoned; discard cache')
        if not requested(os.environ):raise RuntimeError('route snapshot flag changed after cache construction')
        self.validate_dependency()

    def validate_dependency(self):
        """CPU writes must retire same-layer pointer-table DMA/readers first.

        Removing route readback/bounds barriers makes FAST0 unsafe between
        actual chunks. Requested flags alone cannot prove the table exists.
        """
        try:
            from ds41.pointer_table import FastPointerTable
            cache=self._cache()
            table=None if cache is None else getattr(cache,'_fast_pointer_table',None)
            host=None if cache is None else getattr(cache,'host_ptrs',None)
            if (cache is None or os.environ.get('DS41_FAST_POINTER_BIND','0')!='1'
                    or getattr(cache,'_fast_pointer_requested',False) is not True
                    or type(table) is not FastPointerTable or table.host is not host
                    or table.layers!=cache.n_layers
                    or tuple(host.shape)!=(cache.n_layers,9,384)
                    or table.view.__array_interface__['data'][0]!=host.data_ptr()
                    or self._pointer_table is not None and self._pointer_table is not table):
                raise RuntimeError('route snapshot requires its initialized same-cache FastPointerTable')
            table._check()  # Also reject a closed table or foreign CPU owner.
            self._pointer_table=table
        except BaseException:
            self.poison()
            raise

    def poison(self):
        if not self._failed:self._bump('failures')
        self._failed=True
        cache=self._cache()
        if cache is not None:cache._route_snapshot_failed=True

    def _stamp(self,cache):
        return tuple(os.environ.get(key,default) for key,default in (
            ('DS41_ROUTE_AWARE_CHUNKS','0'),('DS41_STRICT_BATCH','1'),('DS41_STRICT_DENSE','1'),
            ('EXL3_MOE_COOP_WIDE','0'),('EXL3_MOE_COOP_KSPLIT','1')))+(
            getattr(cache,'_coop512_requested',False),getattr(cache,'cpu_backend',None) is None,
            id(getattr(cache,'route_observer',None)),cache._moe_observer_slot_limit(),cache.stage_limit)

    def _eligibility(self,cache,layer,flat,indices,weights,k,inter):
        import torch
        if os.environ.get('DS41_ROUTE_AWARE_CHUNKS','0')!='1':return 'route_aware_disabled'
        if os.environ.get('DS41_STRICT_BATCH','1')!='1' or os.environ.get('DS41_STRICT_DENSE','1')!='1':return 'strict_disabled'
        if not getattr(cache,'_coop512_requested',False):return 'native512_disabled'
        if getattr(cache,'cpu_backend',None) is not None:return 'CPU_experts_enabled'
        if type(layer) is not int or not 0<=layer<cache.n_layers:return 'layer_bounds'
        if (not all(isinstance(t,torch.Tensor) for t in (flat,indices,weights))
            or flat.ndim!=2 or not 1<=flat.shape[0]<=128 or tuple(flat.shape[1:])!=(5120,)
            or flat.dtype!=torch.float16 or flat.device.type!='cuda' or not flat.is_contiguous()
            or indices.shape!=(flat.shape[0],6) or indices.dtype!=torch.int64 or not indices.is_contiguous()
            or weights.shape!=indices.shape or weights.dtype not in (torch.float16,torch.bfloat16,torch.float32)
            or not weights.is_contiguous() or any(t.device!=flat.device or t.is_conj() or t.is_neg() for t in (indices,weights))):
            return 'unsupported_geometry_or_device'
        if not cache._coop512_capable(flat,6,k,inter):return 'native512_incapable'
        if cache._moe_observer_slot_limit()<6:return 'observer_capacity'
        return None

    @contextmanager
    def scope(self,cache,layer,flat,indices,weights,k,inter):
        self._owner_check()
        if cache is not self._cache() or self._active is not None:raise RuntimeError('foreign/nested route snapshot scope')
        self._bump('scope_attempts')
        reason=self._eligibility(cache,layer,flat,indices,weights,k,inter)
        if reason is not None:
            self._bump('fallback_windows');self.reasons[reason]+=1
            yield None;return
        window=None
        try:
            from ds41.route_snapshot import capture_routes
            snapshot=capture_routes(indices)
            window=_Window(_KEY,self,cache,layer,flat,indices,weights,k,inter,snapshot)
            self._active=window
            meta=snapshot.metadata();self._bump('eligible_windows');self._bump('snapshot_cpu_readbacks')
            self._bump('route_D2H_bytes',meta['device_to_host_payload_bytes'])
            self._bump('private_D2D_bytes',meta['private_device_copy_bytes'])
            yield window
        except BaseException:
            self.poison();raise
        finally:
            try:
                if window is not None:window.close()
            except BaseException:self.poison();raise
            finally:self._active=None

    def validate_span(self,proof,cache,layer,x,indices,weights,k,inter):
        try:
            self._owner_check()
            if type(proof) is not _Span or proof._window is not self._active or cache is not self._cache():
                raise RuntimeError('foreign/expired/forged route span')
            proof._validate(cache,layer,x,indices,weights,k,inter)
            return proof._span.flat_ids
        except BaseException:self.poison();raise

    def native_run(self,proof,cache,layer,runner,x,indices,weights,packed,k,limit,inter):
        self.validate_span(proof,cache,layer,x,indices,weights,k,inter)
        self._bump('unchecked_attempts')
        try:
            result=proof._span.run_native512(runner,x,weights,packed,k,limit)
            self._bump('unchecked_calls')  # Successful adapter returns, not failed-launch guesses.
            return result
        except BaseException:self.poison();raise

    def fallback_call(self,reason):
        self._owner_check();self._bump('fallback_calls');self.reasons[reason]+=1

    def legacy_readback(self):self._bump('legacy_cache_cpu_readbacks')

    def status(self):
        return dict(**configuration(True),effective=self.counters['unchecked_calls']>0,
            active=self._active is not None,failed=self._failed,counters=dict(self.counters),fallback_reasons=dict(self.reasons),
            counter_scope='cache lifetime including prefill/checks; flat cache stats provide decode deltas',
            readback_scope='completed snapshots and legacy cache reads only; fallback outer-Block/draft reads excluded',
            unchecked_count_scope='successful private native adapter returns; failures poison without guessing launches')


class _Window:
    def __init__(self,key,runtime,cache,layer,flat,indices,weights,k,inter,snapshot):
        if key is not _KEY:raise TypeError('private runtime window')
        self._runtime,self._cache,self._snapshot=runtime,cache,snapshot
        self._layer,self._k,self._inter=layer,k,inter
        self._inputs=(flat,indices,weights)
        self._identities=tuple(_identity(t) for t in self._inputs)
        self._stamp=runtime._stamp(cache);self._closed=False

    def _check(self):
        self._runtime._owner_check()
        if self._closed or self._runtime._active is not self:raise RuntimeError('expired route window')
        if (self._stamp!=self._runtime._stamp(self._cache)
                or self._identities!=tuple(_identity(t) for t in self._inputs)):
            raise RuntimeError('route configuration or source metadata/version changed')
        self._snapshot.rows  # Private version/owner/current-stream check, no value read.

    @property
    def rows(self):self._check();return self._snapshot.rows

    def span(self,start,end):
        self._check()
        return _Span(_KEY,self,start,end,self._snapshot.span(start,end))

    def close(self):
        if self._closed:return
        self._snapshot.close();self._closed=True


class _Span:
    def __init__(self,key,window,start,end,span):
        if key is not _KEY:raise TypeError('private runtime span')
        self._window,self._span=window,span
        self._start,self._end=start,end
        self._identities=tuple(_slice_identity(t,start,end) for t in window._inputs)

    def _validate(self,cache,layer,x,indices,weights,k,inter):
        self._window._check()
        if (cache is not self._window._cache or layer!=self._window._layer or k!=self._window._k or inter!=self._window._inter
                or not 1<=self._end-self._start<=64
                or self._identities!=tuple(_identity(t) for t in (x,indices,weights))):
            raise RuntimeError('route span does not bind these exact layer/input/route/weight views')


def disabled_status():
    return dict(**configuration(False),effective=False,active=False,failed=False,
        counters={key:0 for key in COUNTERS},fallback_reasons={},counter_scope='disabled')
