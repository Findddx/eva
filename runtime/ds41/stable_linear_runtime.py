"""Default-off router/indexer adapter; no Torch/CUDA work at import.

Only the two explicit router/indexer call sites may use it; model.stable_linear
and HC capture remain unchanged. Integration has its own source/evidence scope
(docs/stable-linear-runtime-integration.md); primitive evidence does not certify
a later runtime/full model.
"""
from collections import Counter
import copy
import hashlib
import os
from pathlib import Path
import threading


FLAG = "DS41_STABLE_LINEAR_GRAPHS"
ROWS = (32, 64, 128)
INDEX_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)


def requested(environment=None):
    environment = os.environ if environment is None else environment
    value = environment.get(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(FLAG + " must be 0 or 1")
    return value == "1"


def _sources():
    directory = Path(__file__).parent
    return {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
            for name in ("stable_linear_runtime.py", "stable_linear_graph.py")}


def _torch_build_identity(torch):
    """Hash the actual loaded binding; distinguish dependent-library stat IDs."""
    binding = Path(torch._C.__file__)
    digest = hashlib.sha256()
    with binding.open("rb") as source:
        for part in iter(lambda: source.read(1024**2), b""):
            digest.update(part)
    libraries = {}
    libdir = Path(torch.__file__).parent/"lib"
    for name in ("libc10_cuda.so", "libtorch_cuda.so", "libtorch_cuda_linalg.so",
                 "c10_cuda.dll", "torch_cuda.dll", "torch_cuda_linalg.dll"):
        path = libdir/name
        if path.is_file():
            stat = path.stat()
            libraries[name] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return dict(version=torch.__version__, compiled_cuda=torch.version.cuda,
        git_version=getattr(torch.version, "git_version", None),
        compiled_configuration=torch.__config__.show(),
        binding=dict(name=binding.name, bytes=binding.stat().st_size, sha256=digest.hexdigest()),
        dependent_CUDA_library_stat_identities=libraries,
        all_dependency_content_hashes_claimed=False)


def _key(kind, layer):
    if type(layer) is not int or not 0 <= layer < 40:
        raise ValueError("Require an actual main-model layer0..39")
    if kind == "router":
        return f"layers.{layer}.ffn.gate.weight"
    if kind == "indexer" and layer in INDEX_LAYERS:
        return f"layers.{layer}.attn.indexer.weights_proj.weight"
    raise ValueError("Only the40 routers and8 source indexers may use this adapter")


def _collect_weights(model):
    """Borrow the exact model objects, never invoke a store or load payloads."""
    args = model.args
    if ((args.dim, args.n_layers, args.n_routed_experts, args.index_n_heads)
            != (5120, 40, 384, 32)
            or tuple(args.index_source_layers) != INDEX_LAYERS
            or len(model.layers) != 40):
        raise ValueError("Stable-linear runtime requires the measured40-layer/48-weight layout")
    weights = {}
    for i, layer in enumerate(model.layers):
        if layer.layer_id != i:
            raise ValueError("Main-model layer order changed")
        weights[_key("router", i)] = layer.gate_w
        weight = getattr(layer.attn, "indexer_wn", None)
        if (weight is not None) != (i in INDEX_LAYERS):
            raise ValueError("Source-indexer weight registry changed")
        if weight is not None:
            weights[_key("indexer", i)] = weight
    if len({id(weight) for weight in weights.values()}) != 48:
        raise ValueError("Require48 distinct borrowed weight objects")
    import torch
    for key, weight in weights.items():
        shape = (384, 5120) if ".ffn.gate." in key else (32, 5120)
        if (not isinstance(weight, torch.Tensor) or tuple(weight.shape) != shape
                or weight.dtype != torch.float32 or not weight.is_contiguous()
                or weight.requires_grad):
            raise ValueError("Borrowed router/indexer weight geometry or FP32 dtype changed")
    # The proven runner validates every actual Tensor's device, dtype, shape,
    # contiguity, gradient state and current-device identity before allocation.
    return weights


class StableLinearRuntime:
    """A single, serialized runner shared by explicitly registered call sites.

    Construct after all main weights are ready, before measuring the free VRAM
    expert budget. Inputs/weights require producer happens-before on the runner
    construction stream. Eligible outputs own their storage; M1/other shapes
    call the supplied original helper without substituting any arithmetic.
    """
    def __init__(self, model, *, strict=True):
        if not requested():
            raise ValueError("Explicit stable-linear graph opt-in is required")
        if strict is not True or os.environ.get("DS41_STRICT_DENSE", "1") != "1":
            raise ValueError("Stable-linear graphs require canonical strict-dense arithmetic")
        if os.environ.get("DS41_NO_GRAPH", "0") == "1":
            raise ValueError("Stable-linear graphs conflict with DS41_NO_GRAPH=1")
        from ds41.verification import verification_max_rows
        if verification_max_rows() != 128:
            raise ValueError("Stable-linear runtime requires startup verification cap128")
        from ds41 import stable_linear_graph as core
        self._owner = threading.current_thread()
        self._closed = self._failed = False
        self._runner = None
        self.counters = dict(attempts=0, graph_calls=0, graph_rows=0,
                             fallback_calls=0, failures=0)
        self._hits, self._fallbacks = Counter(), Counter()
        self._weights = _collect_weights(model)
        sources = _sources()
        try:
            self._runner = core.CanonicalStableLinearGraphRunner(
                self._weights, rows=ROWS, strict_dense=True)
            self.device = self._runner.device
            self._buffers = copy.deepcopy(self._runner.buffer_metadata)
            self._startup = copy.deepcopy(self._runner.startup)
            self._identity = dict(requested=True, protocol="canonical_router_indexer_graph_v1",
                rows=list(ROWS), verification_cap=128, weight_registry=list(self._weights),
                source_sha256=sources, candidate_source=core.source_metadata(),
                torch_identity=_torch_build_identity(core.torch),
                precision_workspace=copy.deepcopy(self._runner._config),
                weight_copies=0, owned_output=True,
                integration_scope="Only explicit router/indexer call-site dispatch; global helper unchanged",
                runtime_GPU_gate_passed=False, full_model_gate_passed=False,
                primitive_evidence="docs/stable-linear-and-hc-arena-gpu-v1.json",
                mutation_contract="Borrowed immutable weights; tracked version/storage checked; inference/raw writes unsupported")
            if sources != _sources():
                raise RuntimeError("Stable-linear source changed during construction")
        except BaseException:
            self._failed = True
            if self._runner is not None:
                self._runner.close()
            self._weights.clear()
            raise

    def _check(self):
        if self._closed or self._failed:
            raise RuntimeError("Stable-linear runtime is closed or poisoned")
        if threading.current_thread() is not self._owner:
            raise RuntimeError("Stable-linear runtime requires its original CPU owner")
        from ds41.verification import verification_max_rows
        if (not requested() or verification_max_rows() != 128
                or os.environ.get("DS41_STRICT_DENSE", "1") != "1"
                or os.environ.get("DS41_NO_GRAPH", "0") == "1"):
            raise RuntimeError("Stable-linear execution configuration changed")

    def _poison(self):
        if not self._failed:
            self.counters["failures"] += 1
        self._failed = True

    def linear(self, kind, layer, x, weight, *, fallback):
        """Dispatch one explicit, bias-free original helper invocation."""
        try:
            self._check()
            self.counters["attempts"] += 1
            key = _key(kind, layer)
            if weight is not self._weights[key]:
                raise RuntimeError("Stable-linear bound weight object changed")
            if not callable(fallback):
                raise TypeError("The original stable_linear fallback is required")
            import torch
            reason = None
            if (not isinstance(x, torch.Tensor) or x.ndim not in (2, 3)
                    or x.shape[-1] != 5120 or (x.ndim == 3 and x.shape[0] != 1)):
                reason = "original_geometry"
            elif x.numel()//5120 not in ROWS:
                reason = "original_rows"
            elif (x.dtype != torch.float32 or x.device != self.device
                  or not x.is_contiguous() or x.requires_grad or x.is_conj() or x.is_neg()):
                reason = "original_operand"
            if reason is not None:
                # Even fallback calls on an enabled runtime cannot bypass its
                # bound stream/configuration/immutable-weight owner contract.
                self._runner._check_owner()
                self._runner._check_configuration(key)
                result = fallback(x, weight)
                self.counters["fallback_calls"] += 1
                self._fallbacks[reason] += 1
                return result
            result = self._runner.run(key, x)  # shared IO -> owned clone
            rows = x.numel()//5120
            self.counters["graph_calls"] += 1
            self.counters["graph_rows"] += rows
            self._hits[f"{kind}:{rows}"] += 1
            return result
        except BaseException:
            self._poison()
            raise  # Never retry after a partial capture/replay/fallback failure.

    def identity(self):
        return copy.deepcopy(self._identity)

    def report(self):
        return dict(identity=self.identity(), effective=self.counters["graph_calls"] > 0,
            closed=self._closed, failed=self._failed, counters=dict(self.counters),
            graph_hits=dict(self._hits), fallback_reasons=dict(self._fallbacks),
            graph_buffers=copy.deepcopy(self._buffers), construction=copy.deepcopy(self._startup),
            counter_scope="Successful graph/fallback returns over model lifetime, including prefill/checks; not valid-output throughput")

    def verify_metadata(self):
        try:
            self._check()
            self._runner._check_owner()
            self._runner._check_configuration()
            if _sources() != self._identity["source_sha256"]:
                raise RuntimeError("Stable-linear source identity changed")
            from ds41 import stable_linear_graph as core
            if _torch_build_identity(core.torch) != self._identity["torch_identity"]:
                raise RuntimeError("Stable-linear Torch build identity changed")
        except BaseException:
            self._poison()
            raise

    def close(self):
        if self._closed:
            return
        if threading.current_thread() is not self._owner:
            raise RuntimeError("Stable-linear close requires its original CPU owner")
        try:
            self._runner.close()  # Full device drain before graph/weight release.
        except BaseException:
            self._poison()
            raise
        self._weights.clear()
        self._closed = True


def initialize(model, *, strict=True):
    """Explicit future integration point; flag0 does not import Torch or allocate."""
    if getattr(model, "_stable_linear_initialized", False):
        raise RuntimeError("Stable-linear runtime cannot be initialized/recaptured twice")
    enabled = requested()
    runtime = StableLinearRuntime(model, strict=strict) if enabled else None
    model._stable_linear_runtime = runtime
    model._stable_linear_initialized = True
    return runtime


def dispatch(runtime, kind, layer, x, weight, *, fallback):
    if runtime is None:
        return fallback(x, weight)
    if type(runtime) is not StableLinearRuntime:
        raise TypeError("Require the model's explicit stable-linear runtime")
    return runtime.linear(kind, layer, x, weight, fallback=fallback)


def configuration(model):
    runtime = getattr(model, "_stable_linear_runtime", None)
    return (dict(requested=False, protocol="canonical_router_indexer_graph_v1",
                 source_sha256=_sources()) if runtime is None else runtime.identity())


def status(model):
    runtime = getattr(model, "_stable_linear_runtime", None)
    return (dict(identity=configuration(model), effective=False, counters={})
            if runtime is None else runtime.report())
