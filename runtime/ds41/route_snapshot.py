"""Independent, unused route-readback candidate; no production hooks.

The only factory takes a real tensor, owns a tiny versioned copy and reads that
copy once. No caller-supplied CPU list can authorize unchecked native routing.
CPU tensors support control fixtures, not native GPU execution. Python object
privacy is a trusted in-process contract, not a hostile-code security boundary.
"""
import threading


_CONSTRUCTION_KEY=object()
MAX_ROWS=128
TOPK=6
EXPERTS=384


def _signature(tensor):
    storage=tensor.untyped_storage()
    return (tuple(tensor.shape),tuple(tensor.stride()),tensor.dtype,tensor.device,
            tensor.storage_offset(),tensor.data_ptr(),storage.data_ptr(),storage.nbytes(),tensor._version)


def _cpu_rows(values,rows):
    # Input is only private_tensor.tolist(), never an API argument.
    if (type(values) is not list or len(values)!=rows
            or any(type(row) is not list or len(row)!=TOPK for row in values)
            or any(type(value) is not int or not 0<=value<EXPERTS for row in values for value in row)):
        raise ValueError("actual router IDs must be int64 values in [0,384)")
    return tuple(tuple(row) for row in values)


class RouteSnapshot:
    """One CPU thread and CUDA stream; immutable consumed-route snapshot.

    Do not expose or write private storage via .data, native pointers or another
    stream. Such untracked writes cannot be detected by a CPU version counter.
    The source tensor may be an inference tensor: after capture, only the owned
    normal tensor is consumed. Later source changes cannot alter this snapshot.
    """
    def __init__(self,key,private,values,stream):
        if key is not _CONSTRUCTION_KEY:
            raise TypeError("use capture_routes(actual_tensor), never a CPU-list constructor")
        self._tensor=private
        self._rows=values
        self._signature=_signature(private)
        self._owner=threading.current_thread()
        self._stream=stream
        self._busy=self._failed=self._closed=False
        self._spans_created=self._native_calls=0

    def _check(self):
        if threading.current_thread() is not self._owner:
            raise RuntimeError("route snapshot requires its CPU owner thread")
        if self._closed or self._failed:
            raise RuntimeError("route snapshot is closed or poisoned")
        import torch
        if self._stream is not None:
            if (torch.cuda.current_device()!=self._tensor.device.index
                    or torch.cuda.current_stream(self._tensor.device).cuda_stream!=self._stream.cuda_stream
                    or torch.cuda.is_current_stream_capturing()):
                raise RuntimeError("route snapshot requires its original device/stream outside capture")
        if self._busy:
            raise RuntimeError("route snapshot dispatch is already active")
        if self._tensor.is_inference() or _signature(self._tensor)!=self._signature:
            self._failed=True
            raise RuntimeError("private consumed routes changed; discard snapshot")

    @property
    def rows(self):
        self._check()
        return self._rows

    def span(self,start,end):
        self._check()
        if (type(start) is not int or type(end) is not int
                or not 0<=start<end<=len(self._rows)):
            raise ValueError("route span must be a nonempty exact [start,end) row interval")
        import torch
        with torch.inference_mode(False),torch.no_grad():
            view=self._tensor[start:end]
        self._spans_created+=1
        return RouteSpan(_CONSTRUCTION_KEY,self,start,end,view)

    def close(self):
        if threading.current_thread() is not self._owner or self._busy:
            raise RuntimeError("close requires idle snapshot owner")
        if self._closed:return
        # All enqueues are on the recorded stream. Keep allocations alive through
        # that stream's allocator retirement, without adding a host synchronize.
        if self._stream is not None:self._tensor.record_stream(self._stream)
        self._closed=True
        self._tensor=None

    def metadata(self):
        if threading.current_thread() is not self._owner:
            raise RuntimeError("snapshot metadata requires CPU owner")
        count=len(self._rows)*TOPK*8
        return dict(schema=1,candidate_only=True,production_integration=False,
            row_count=len(self._rows),topk=TOPK,expert_bound=EXPERTS,
            cpu_readbacks=1,route_payload_bytes=count,
            device_to_host_payload_bytes=count if self._stream is not None else 0,
            private_device_copy_bytes=count if self._stream is not None else 0,
            versioned_private_consumed_routes=True,accepts_external_cpu_ids=False,
            source_mutations_after_capture_affect_snapshot=False,
            created_spans=self._spans_created,native_calls=self._native_calls,
            native_bounds_mode="CPU-validated private snapshot only",closed=self._closed,failed=self._failed,
            numerical_GPU_gate_passed=False,full_model_gate_passed=False)


class RouteSpan:
    """Opaque exact view + immutable CPU IDs; no public tensor accessor."""
    def __init__(self,key,snapshot,start,end,view):
        if key is not _CONSTRUCTION_KEY or type(snapshot) is not RouteSnapshot:
            raise TypeError("route spans may only be minted by their snapshot")
        self._snapshot=snapshot
        self._start,self._end,self._view=start,end,view
        self._signature=_signature(view)
        self._ids=tuple(value for row in snapshot._rows[start:end] for value in row)
        self._unique=tuple(dict.fromkeys(self._ids))
        self._minimum,self._maximum=min(self._ids),max(self._ids)

    def _check(self):
        self._snapshot._check()
        if self._view.is_inference() or _signature(self._view)!=self._signature:
            self._snapshot._failed=True
            raise RuntimeError("route span view changed; discard snapshot")

    @property
    def bounds(self):
        self._check();return self._start,self._end

    @property
    def flat_ids(self):
        self._check();return self._ids

    @property
    def unique_ids(self):
        self._check();return self._unique

    def validate_local_bounds(self,n_local):
        self._check()
        if type(n_local) is not int or not 1<=n_local<=EXPERTS:
            raise ValueError("pointer tables require integer lengths in [1,384]")
        if not 0<=self._minimum<=self._maximum<n_local:
            raise ValueError("actual CPU-read route IDs exceed this pointer table")
        return True

    def run_native512(self,runner,x,weights,packed,k,limit):
        """Dispatch unchanged native math with internally authorized bounds.

        Expert/pointer-table leases and producer happens-before are still the
        caller's responsibility. Result ownership remains the runner's borrowed
        output. This API intentionally has no indices/list/check_bounds argument.
        """
        self._check()
        import torch
        from ds41.moe_coop512 import CoopRunner512
        if self._snapshot._stream is None:
            raise RuntimeError("CPU snapshot fixtures cannot launch native CUDA")
        if type(runner) is not CoopRunner512 or runner.device!=self._view.device:
            raise TypeError("require the exact existing CoopRunner512 on the snapshot device")
        if not 1<=self._end-self._start<=64:
            raise ValueError("native512 actual rows must be in [1,64]")
        if (not isinstance(packed,(list,tuple)) or len(packed)!=9
                or any(not isinstance(t,torch.Tensor) or t.dtype!=torch.int64
                       or t.device!=self._view.device or t.ndim!=1 or not t.is_contiguous() for t in packed)
                or len({t.numel() for t in packed})!=1):
            raise ValueError("require nine same-device contiguous int64 pointer tables of equal length")
        self.validate_local_bounds(packed[0].numel())
        snapshot=self._snapshot
        snapshot._busy=True
        try:
            # Shape/dtype/stream/scratch/pointer ABI checks remain in the runner
            # and native binding. Only two redundant GPU value reductions skip.
            result=CoopRunner512.run(runner,x,self._view,weights,packed,k,limit,check_bounds=False)
            if _signature(self._view)!=self._signature or _signature(snapshot._tensor)!=snapshot._signature:
                raise RuntimeError("consumed routes changed during native enqueue")
            snapshot._native_calls+=1
            return result
        except BaseException:
            snapshot._failed=True
            raise
        finally:snapshot._busy=False


def capture_routes(indices):
    """Own a normal route copy and perform exactly one tensor.tolist readback.

    Clone/readback are ordered on the current producer stream. Cross-stream
    producers need an explicit wait before calling this factory; record_stream
    alone does not establish readiness. No arbitrary host IDs are accepted.
    """
    import torch
    if (not isinstance(indices,torch.Tensor) or indices.dtype!=torch.int64
            or indices.ndim!=2 or not 1<=indices.shape[0]<=MAX_ROWS or indices.shape[1]!=TOPK
            or not indices.is_contiguous() or indices.device.type not in ('cpu','cuda')
            or indices.is_conj() or indices.is_neg()):
        raise ValueError("routes require contiguous int64[1..128,6] on CPU/CUDA without logical view flags")
    stream=None
    if indices.device.type=='cuda':
        if torch.cuda.current_device()!=indices.device.index or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("route capture requires current source device outside CUDA graph capture")
        stream=torch.cuda.current_stream(indices.device)
    # Router outputs under inference_mode do not have a version counter.
    # Make the *consumed* route copy normal even under an outer inference scope.
    with torch.inference_mode(False),torch.no_grad():
        private=indices.detach().clone(memory_format=torch.contiguous_format)
    before=_signature(private)
    values=_cpu_rows(private.tolist(),indices.shape[0])
    if private.is_inference() or _signature(private)!=before:
        raise RuntimeError("route snapshot changed during its single readback")
    if stream is not None:private.record_stream(stream)
    return RouteSnapshot(_CONSTRUCTION_KEY,private,values,stream)
