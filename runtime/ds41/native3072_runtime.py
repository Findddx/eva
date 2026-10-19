"""Default-off ownership for one checked M65..512 coop3072 workspace.

This is an unverified production candidate. It changes neither the standalone
primitive nor its certificate fields. Import/disabled configuration use stdlib
only. Source/binary hashing occurs at construction and explicit output checks.
"""
import copy
import hashlib
import os
from pathlib import Path
import threading


FLAG = "DS41_MOE_COOP3072"
COUNTERS = ("calls", "rows", "routed_slots", "failures")
PROTOCOL = "checked_M65_512_actual_union_coop3072_v1"
JOINT_PROTOCOL = "checked_M513_1024_joint_union_two_coop3072_v1"
JOINT_COUNTERS = ("layer_calls", "rows", "routed_slots", "kernel_calls", "failures")


def joint_spans(rows):
    """Balanced halves keep both original production calls in65..512."""
    if type(rows) is not int or not 513 <= rows <= 1024:
        raise ValueError("Joint3072 requires513..1024 actual rows")
    middle = (rows+1)//2
    return ((0,middle),(middle,rows))


def requested(environment=None):
    value = (os.environ if environment is None else environment).get(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(FLAG + " must be 0 or 1")
    return value == "1"


def validate_configuration(environment=None, *, cap, speculation=None, prefill_chunk_size=None,
                           volatile_experts=None, cpu_experts=False, adaptive=False, next_chunk=False):
    env = os.environ if environment is None else environment
    if not requested(env):
        return dict(requested=False)
    from ds41.hc_verify_runtime import requested as arena_requested
    arena_enabled = arena_requested(env)
    if type(cap) is int and cap == 1024:
        required = {"DS41_MOE_COOP512":"1", "DS41_MOE_COOP2048":"0",
            "DS41_FAST_POINTER_BIND":"1", "DS41_ROUTE_SNAPSHOT":"0", "DS41_ROUTE_AWARE_CHUNKS":"1",
            "DS41_STRICT_BATCH":"1", "DS41_STRICT_DENSE":"1", "DS41_FUSED_HC":"0",
            "DS41_DENSE_CANONICAL_TILE":"32", "DS41_NEXT_CHUNK_PREFETCH":"0", "DS41_DEMAND_H2D_PIPELINE":"1",
            "DS41_HC_VERIFY_GRAPHS":"1024", "DS41_HC_VERIFY_SHARED_POOL":"1", "DS41_HC_VERIFY_IO_ARENA":"1" if arena_enabled else "0",
            "DS41_STABLE_LINEAR_GRAPHS":"0", "DS41_CANONICAL_MEAN_SQUARE":"1",
            "DS41_LARGE_KV_PREPARE":"1", "DS41_FULL_COMPRESSED_KV":"1", "DS41_LAYER_PROMOTION":"0",
            "EXL3_MOE_COOP_WIDE":"0", "EXL3_MOE_COOP_KSPLIT":"1"}
        for name,value in required.items():
            if env.get(name) != value:
                raise ValueError("Joint3072 cap1024 requires "+name+"="+value)
        if speculation is not None and speculation != "lookup":
            raise ValueError("Joint3072 cap1024 is lookup-only")
        if prefill_chunk_size is not None and (type(prefill_chunk_size) is not int or prefill_chunk_size != 128):
            raise ValueError("Joint3072 cap1024 retains prefill128")
        if volatile_experts is not None and (type(volatile_experts) is not int or volatile_experts != 384):
            raise ValueError("Joint3072 cap1024 requires actual384 volatile slots")
        if cpu_experts or adaptive or next_chunk:
            raise ValueError("Joint3072 excludes CPU/adaptive/next-chunk combinations")
        return dict(requested=True,protocol=JOINT_PROTOCOL,verification_cap=1024,minimum_native_rows=65,
            maximum_native_rows=512,maximum_layer_rows=1024,maximum_routed_slots=3072,
            maximum_layer_routed_slots=6144,required_volatile_slots=384,observer_supported=False,
            GPU_bounds_checks=True,original512_remains_enabled=True,original2048_workspace_allowed=False,
            shared_512_workspace=True,joint_complete_layer_union_lease=True,balanced_two_kernel_split=True,
            required_environment=required,production_GPU_certificate=False,full_model_GPU_certificate=False)
    if type(cap) is not int or cap != 512:
        raise ValueError("coop3072 production requires explicit target cap512")
    required = {"DS41_MOE_COOP512": "1", "DS41_MOE_COOP2048": "0",
        "DS41_FAST_POINTER_BIND": "1", "DS41_ROUTE_SNAPSHOT": "0",
        "DS41_ROUTE_AWARE_CHUNKS": "1", "DS41_STRICT_BATCH": "1", "DS41_STRICT_DENSE": "1",
        "DS41_FUSED_HC": "0", "DS41_DENSE_CANONICAL_TILE": "32", "DS41_NEXT_CHUNK_PREFETCH": "0",
        "DS41_DEMAND_H2D_PIPELINE": "1", "DS41_HC_VERIFY_GRAPHS": "512",
        "DS41_HC_VERIFY_SHARED_POOL": "1", "DS41_HC_VERIFY_IO_ARENA": "1" if arena_enabled else "0",
        "DS41_STABLE_LINEAR_GRAPHS": "0", "EXL3_MOE_COOP_WIDE": "0", "EXL3_MOE_COOP_KSPLIT": "1"}
    for name, value in required.items():
        if env.get(name) != value:
            raise ValueError("coop3072 requires " + name + "=" + value)
    if speculation is not None and speculation != "lookup":
        raise ValueError("coop3072 production first supports lookup only")
    if prefill_chunk_size is not None and prefill_chunk_size != 128:
        raise ValueError("coop3072 production retains prefill chunk128")
    if volatile_experts is not None and (type(volatile_experts) is not int or volatile_experts != 384):
        raise ValueError("coop3072 requires exactly384 actual volatile slots")
    if cpu_experts or adaptive or next_chunk:
        raise ValueError("coop3072 excludes CPU experts, continuous adaptive and next-chunk prefetch")
    return dict(requested=True, protocol=PROTOCOL, verification_cap=512, minimum_native_rows=65,
        maximum_native_rows=512, maximum_routed_slots=3072, required_volatile_slots=384,
        observer_supported=False, GPU_bounds_checks=True, original512_remains_enabled=True,
        original2048_workspace_allowed=False, required_environment=required,
        production_GPU_certificate=False, full_model_GPU_certificate=False)


def _sources():
    root = Path(__file__).parent
    names = ("native3072_runtime.py", "cache.py", "route_chunks.py", "verification.py", "generate.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _binary(runner):
    path = Path(runner._backend.__file__).resolve(strict=True)
    private = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not private or not path.is_relative_to(Path(private).resolve(strict=True)):
        raise RuntimeError("coop3072 consumer must reside in explicit private TORCH_EXTENSIONS_DIR")
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def disabled_status():
    return dict(requested=False, effective=False, production_integrated=False, closed=False,
        failed=False, identity=dict(requested=False), counters=dict.fromkeys(COUNTERS, 0))


class Native3072Runtime:
    def __init__(self, cache, device, *, volatile_experts):
        # Initialize sentinels before the first import/CUDA call. The cache owns
        # this object before construction, including partially allocated errors.
        self._cache_id = id(cache)
        self._thread = threading.get_ident()
        self._stream = self._runner = None
        self._closed = self._failed = False
        self._identity = self.configuration = None
        self.counters = dict.fromkeys(COUNTERS, 0)
        self._joint_current = None
        self._joint_failed = []
        self._joint_counters = dict.fromkeys(JOINT_COUNTERS,0)
        self._pointer_table = getattr(cache, "_fast_pointer_table", None)
        self._host_ptrs = getattr(cache, "host_ptrs", None)
        self._packed = getattr(cache, "packed", None)
        self._pipeline = getattr(cache, "_demand_h2d_pipeline", None)
        import torch
        from ds41.verification import verification_max_rows
        from ds41.moe_coop3072 import CoopRunner3072, WorkspaceLayout, source_metadata
        self.configuration = validate_configuration(cap=verification_max_rows(), volatile_experts=volatile_experts)
        self._validate_cache(cache, placement=False)
        if (getattr(cache, "vram", None) or getattr(cache, "frozen", None)
                or any(getattr(cache, "pools", {}).values())):
            raise RuntimeError("coop3072 must be initialized before expert GPU budgeting/allocation")
        self._stream = torch.cuda.current_stream(device)
        before = self._memory(torch, device)
        self._runner = CoopRunner3072(device, topk=6, hidden=5120, inter=2304, max_rows=512)
        self._stream.synchronize()
        after = self._memory(torch, device)
        self._identity = dict(self.configuration, source_sha256=_sources(), native_build=source_metadata(),
            extension=_binary(self._runner), nominal_workspace_bytes=WorkspaceLayout().scratch_bytes,
            device=str(self._runner.device), stream_contract="construction_stream_only",
            allocated_before_expert_budget=True, constructor_memory=dict(before=before, after=after,
                allocated_delta_bytes=after["allocated"]-before["allocated"],
                reserved_delta_bytes=after["reserved"]-before["reserved"],
                device_free_delta_bytes=before["free"]-after["free"]),
            original512_workspace_is_additional=True, original2048_workspace_allocated=False,
            new_copy_streams=0, new_weight_slots=0, runtime_GPU_gate_passed=False, full_model_gate_passed=False)
        # The primitive constructor has already checked this actual SM120 device.
        # Avoid another device-properties query in the steady-state capability path.
        cache._coop512_devices[str(self._runner.device)] = True

    @staticmethod
    def _memory(torch, device):
        return dict(allocated=torch.cuda.memory_allocated(device), reserved=torch.cuda.memory_reserved(device),
                    free=torch.cuda.mem_get_info(device)[0])

    def _validate_cache(self, cache, *, placement=True):
        if id(cache) != self._cache_id or threading.get_ident() != self._thread:
            raise RuntimeError("coop3072 requires its original cache/CPU owner")
        if self._closed or self._failed:
            raise RuntimeError("coop3072 runtime is closed or failed")
        from ds41.pointer_table import FastPointerTable
        table = getattr(cache, "_fast_pointer_table", None)
        if (not isinstance(table, FastPointerTable) or table is not self._pointer_table
                or cache.host_ptrs is not self._host_ptrs or table.host is not self._host_ptrs
                or getattr(cache, "packed", None) is not self._packed
                or not getattr(cache, "_fast_pointer_requested", False)
                or getattr(cache, "_route_snapshot_runtime", None) is not None
                or getattr(cache, "route_observer", None) is not None
                or getattr(cache, "_coop2048_requested", False)
                or getattr(cache, "_native2048_runtime", None) is not None
                or getattr(cache, "cpu_backend", None) is not None or not cache.retain_host
                or getattr(cache, "adaptive_gpu_enabled", False)
                or self._pipeline is None or getattr(cache, "_demand_h2d_pipeline", None) is not self._pipeline
                or getattr(cache, "_chunk_prefetch", None) is not None
                or getattr(getattr(cache, "_chunk_prefetch_config", None), "enabled", False)):
            raise RuntimeError("coop3072 cache dependencies changed or are incompatible")
        if placement and (type(cache.stage_limit) is not int or cache.stage_limit != 384):
            raise RuntimeError("coop3072 actual volatile capacity must remain exactly384")

    def run(self, cache, x, indices, weights, packed, k, limit):
        import torch
        self._validate_cache(cache)
        if x.ndim != 2 or not 65 <= x.shape[0] <= 512:
            raise ValueError("production coop3072 is restricted to65..512 B1 rows")
        if torch.cuda.current_stream(x.device).cuda_stream != self._stream.cuda_stream:
            raise RuntimeError("coop3072 requires its actual construction stream")
        try:
            result = self._runner.run(x, indices, weights, packed, k, limit, check_bounds=True)
        except BaseException:
            self.poison()
            cache._coop3072_failed = True
            raise
        self.counters["calls"] += 1
        self.counters["rows"] += x.shape[0]
        self.counters["routed_slots"] += indices.numel()
        return result

    def verify_metadata(self):
        if self._closed or self._failed or threading.get_ident() != self._thread:
            raise RuntimeError("coop3072 metadata requested from closed/failed/foreign runtime")
        from ds41.moe_coop3072 import source_metadata
        from ds41.verification import verification_max_rows
        if (validate_configuration(cap=verification_max_rows()) != self.configuration
                or _sources() != self._identity["source_sha256"]
                or source_metadata() != self._identity["native_build"]
                or _binary(self._runner) != self._identity["extension"]):
            self.poison()
            raise RuntimeError("coop3072 source/build/consumer changed")

    def begin_joint(self, cache, layer, ids, rows):
        """Strong ownership spans all actual kernels and final slot retirement."""
        self._validate_cache(cache)
        joint_spans(rows)
        if self.configuration.get("verification_cap") != 1024 or self._joint_current is not None:
            raise RuntimeError("Joint3072 requires its idle exact1024 owner")
        keys=tuple((layer,expert) for expert in ids)
        if cache._active_keys != set(keys):
            raise RuntimeError("Joint3072 requires the complete current-layer active union")
        owners=[]
        for key in keys:
            slot=cache.frozen.get(key)
            if slot is None:slot=cache.vram.get(key)
            if slot is None:raise RuntimeError("Joint3072 expert was not installed before launch")
            owners.append(slot)
        lease=dict(rows=rows,slots=tuple(owners),packed=self._packed,host=self._host_ptrs,outputs=[])
        self._joint_current=lease
        return lease

    def retain_joint_output(self, lease, output):
        if lease is not self._joint_current:
            raise RuntimeError("Joint3072 output belongs to another transaction")
        lease["outputs"].append(output)

    def finish_joint(self, lease):
        if lease is not self._joint_current or len(lease["outputs"]) != 2:
            raise RuntimeError("Joint3072 retirement lacks two owned kernel outputs")
        for name,value in (("layer_calls",1),("rows",lease["rows"]),("routed_slots",lease["rows"]*6),("kernel_calls",2)):
            self._joint_counters[name]+=value
        self._joint_current=None

    def fail_joint(self):
        if self._joint_current is not None:
            self._joint_failed.append(self._joint_current)
            self._joint_current=None
            self._joint_counters["failures"]+=1
        self.poison()

    def poison(self):
        if not self._failed:
            self.counters["failures"] += 1
        self._failed = True

    def report(self):
        value = dict(requested=True, effective=self._identity is not None, production_integrated=True,
            closed=self._closed, failed=self._failed, identity=copy.deepcopy(self._identity),
            counters=dict(self.counters), primitive_lifecycle=(None if self._runner is None else self._runner.lifecycle()),
            counter_scope="Successful checked65..512 actual target calls; prefill and rejected verifier rows included")
        if self.configuration is not None and self.configuration.get("verification_cap") == 1024:
            value["joint_layer"] = dict(protocol=JOINT_PROTOCOL,counters=dict(self._joint_counters),
                active=self._joint_current is not None,failed_lease_count=len(self._joint_failed),
                failed_owned_outputs=sum(len(lease["outputs"]) for lease in self._joint_failed),
                one_stage_transaction_per_layer=True,actual_native_calls_per_joint=2,
                complete_union_and_outputs_retained_until_retirement=True,
                counter_scope="Joint513..1024 layer stage transactions; native counters above count each actual kernel")
        return value

    def close(self):
        if self._closed:
            return
        if threading.get_ident() != self._thread:
            self.poison()
            raise RuntimeError("coop3072 close requires original CPU owner")
        try:
            if self._runner is not None:
                self._runner.close()  # Drains its recorded stream, retains all leases on failure.
            elif self._stream is not None:
                self._stream.synchronize()
        except BaseException:
            self.poison()
            raise
        self._closed = True  # Keep runner/scratch strongly owned for lifecycle receipts.
        self._joint_current=None
        self._joint_failed.clear()


def initialize(cache, device, *, volatile_experts):
    if not getattr(cache, "_coop3072_requested", False):
        return
    if getattr(cache, "_native3072_runtime", None) is not None:
        raise RuntimeError("coop3072 cannot be initialized twice")
    runtime = Native3072Runtime.__new__(Native3072Runtime)
    cache._native3072_runtime = runtime
    try:
        runtime.__init__(cache, device, volatile_experts=volatile_experts)
    except BaseException:
        cache._coop3072_failed = True
        runtime.poison()
        try:
            runtime.close()
        except BaseException:
            pass  # Preserve original error and the attached poisoned owner.
        raise


def status(cache):
    runtime = getattr(cache, "_native3072_runtime", None)
    if runtime is not None:
        report = runtime.report()
        stats = getattr(cache, "stats", {})
        report["cache_fallback_counts"] = {name: stats.get("moe_coop3072_fallback_"+name, 0)
                                           for name in ("calls", "rows")}
        return report
    if getattr(cache, "_coop3072_requested", False):
        return dict(disabled_status(), requested=True, failed=getattr(cache, "_coop3072_failed", False),
                    initialization_pending=True)
    return disabled_status()
