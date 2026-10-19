"""Default-off target-only eager adapter; existing graphs keep their M1 math.

No Torch import, CUDA allocation, or scope is created when disabled. Build identities are derived from the current primitive sources and actual
backend. Numerical validation is performed for each supported configuration.
"""
from collections import Counter
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import copy
import hashlib
import os
from pathlib import Path
import threading


FLAG = "DS41_CANONICAL_MEAN_SQUARE"
PROTOCOL = "target_eager_m1_tree_mean_square_v1"
_ACTIVE = ContextVar("ds41_target_mean_square", default=None)
_COUNTERS = ("scopes", "attempts", "calls", "rows", "fallback_calls", "failures")
_FAILED_INITIALIZATIONS = []  # Retain async owners if construction drain fails.


def requested(environ=None):
    value = (os.environ if environ is None else environ).get(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(FLAG + " must be 0 or 1")
    return value == "1"


def validate_options(*, strict=True, environ=None):
    enabled = requested(environ)
    source = os.environ if environ is None else environ
    if enabled and (strict is not True or source.get("DS41_STRICT_DENSE", "1") == "0"):
        raise ValueError("Mean-square runtime requires strict dense arithmetic")
    return enabled


def _sources():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ("mean_square_runtime.py", "model.py")}


def _binary(backend):
    path = Path(backend.__file__).resolve(strict=True)
    directory = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not directory or not path.is_relative_to(Path(directory).resolve(strict=True)):
        raise RuntimeError("Mean-square consumer must be in the explicit private TORCH_EXTENSIONS_DIR")
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class MeanSquareRuntime:
    def __init__(self, model, *, strict=True):
        if not validate_options(strict=strict):
            raise ValueError("Mean-square runtime requires explicit opt-in and strict dense arithmetic")
        from ds41.verification import verification_max_rows
        import torch
        self._owner = threading.current_thread()
        self._model_id = id(model)
        self._methods = {name: getattr(getattr(model, name), "__func__", None)
                         for name in ("forward", "_forward_complete", "run_logged")}
        self.cap = verification_max_rows()
        if self.cap == 1024:
            from ds41 import canonical_mean_square1024 as core
            constructor = core.CanonicalMeanSquare1024Runner
        elif self.cap == 512:
            from ds41 import canonical_mean_square512 as core
            constructor = core.CanonicalMeanSquare512Runner
        else:
            from ds41 import canonical_mean_square as core
            constructor = core.CanonicalMeanSquareRunner
        self._core = core
        self._closed = self._failed = self._entered = False
        self._runner = None
        self.counters = dict.fromkeys(_COUNTERS, 0)
        self._hits, self._width_rows, self._fallbacks = Counter(), Counter(), Counter()
        sources = _sources()
        build_identity = core.source_metadata()
        try:
            self._runner = constructor(model.device)
            if core.source_metadata() != build_identity or self._runner.metadata["build_sha256"] != build_identity["build_sha256"]:
                raise RuntimeError("Mean-square primitive source changed during construction")
            self.device = self._runner.device
            stream = self._runner._stream
            stream.synchronize()
            before = self._memory(torch)
            # Actual M1 config is checked for every supported width before the
            # expert budget. No candidate graph is captured or recaptured.
            for width in core.WIDTHS:
                dummy = torch.zeros((2, width), dtype=torch.float32, device=self.device)
                output = self._runner.run(dummy)
                del output, dummy
            stream.synchronize()
            self._startup = dict(before=before, after=self._memory(torch), warmup_calls=len(core.WIDTHS),
                                 warmup_rows=2, maximum_tensor_warmed=False)
            self._identity = dict(requested=True, protocol=PROTOCOL, verification_cap=self.cap,
                source_sha256=sources, candidate_source=build_identity,
                installed_torch=copy.deepcopy(self._runner.installed),
                extension=_binary(self._runner._backend),
                aten_m1_configs=copy.deepcopy(self._runner.configs),
                support=dict(widths=list(core.WIDTHS), min_rows=2, max_rows=min(self.cap, core.MAX_ROWS),
                             dtypes=["float16", "bfloat16", "float32"], contiguous=True),
                existing_graphs_use_original_helper=True, graph_capture_supported=False,
                independent_engram_hc4_reduction_unchanged=True, draft_scope_enabled=False,
                owned_outputs=True, persistent_tensor_scratch_bytes=0,
                build_verified_at_construction=True)
            self._identity["variant"] = "canonical_M1_tree_mean_square_" + str(self.cap) + "_v1"
            if sources != _sources():
                raise RuntimeError("Mean-square source changed during construction")
        except BaseException:
            self._failed = True
            if self._runner is not None:
                try:
                    self._runner.close()
                except BaseException:
                    _FAILED_INITIALIZATIONS.append(self)
            raise

    def _memory(self, torch):
        return dict(allocated=torch.cuda.memory_allocated(self.device),
                    reserved=torch.cuda.memory_reserved(self.device),
                    free=torch.cuda.mem_get_info(self.device)[0])

    def _health(self):
        if self._closed or self._failed:
            raise RuntimeError("Mean-square runtime is closed or poisoned")
        if threading.current_thread() is not self._owner:
            raise RuntimeError("Mean-square runtime requires its original CPU owner")
        if (getattr(self._runner, "_closed", False) or getattr(self._runner, "_failed", False)):
            self._poison()
            raise RuntimeError("Mean-square primitive runner is closed or poisoned")

    def _configuration(self):
        from ds41.verification import verification_max_rows
        if (not requested() or os.environ.get("DS41_STRICT_DENSE", "1") == "0"
                or verification_max_rows() != self.cap):
            raise RuntimeError("Mean-square execution configuration changed")

    def _poison(self):
        if not self._failed:
            self.counters["failures"] += 1
        self._failed = True

    @contextmanager
    def forward_scope(self, model):
        self._health()
        self._configuration()
        if id(model) != self._model_id or self._entered or _ACTIVE.get() is not None:
            raise RuntimeError("Mean-square requires its non-reentrant target owner scope")
        if any(getattr(getattr(model, name), "__func__", None) is not function
               for name, function in self._methods.items()):
            raise RuntimeError("Mean-square supports the original complete target forward only")
        self._entered = True
        token = _ACTIVE.set(self)
        self.counters["scopes"] += 1
        try:
            yield
        except BaseException:
            self._poison()
            raise
        finally:
            _ACTIVE.reset(token)
            self._entered = False

    def _fallback(self, reason):
        self.counters["fallback_calls"] += 1
        self._fallbacks[reason] += 1
        return None

    def mean_square(self, original, flat):
        self._health()
        if not self._entered or _ACTIVE.get() is not self:
            raise RuntimeError("Mean-square dispatch outside its owned target scope")
        self.counters["attempts"] += 1
        import torch
        WIDTHS, MAX_ROWS = self._core.WIDTHS, self._core.MAX_ROWS
        rows, width = flat.shape
        if rows <= 1:
            return self._fallback("M1")
        if rows > min(self.cap, MAX_ROWS):
            return self._fallback("over_cap")
        if width not in WIDTHS:
            return self._fallback("unsupported_width")
        if original.device != self.device:
            return self._fallback("different_device")
        if original.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return self._fallback("unsupported_dtype")
        if not original.is_contiguous() or original.is_conj() or original.is_neg() or original.requires_grad:
            return self._fallback("unsupported_layout_or_grad")
        if flat.dtype != torch.float32 or not flat.is_contiguous() or flat.data_ptr() % 16:
            return self._fallback("converted_buffer_layout_or_alignment")
        stream = torch.cuda.current_stream(self.device)
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                return self._fallback("capture_original_math")
        if stream.cuda_stream != self._runner._stream.cuda_stream:
            return self._fallback("different_stream_original_math")
        try:
            result = self._runner.run(flat)
        except BaseException:
            self._poison()
            raise
        self.counters["calls"] += 1
        self.counters["rows"] += rows
        self._hits[str(width)] += 1
        self._width_rows[str(width)] += rows
        return result

    def identity(self):
        return copy.deepcopy(self._identity)

    def report(self):
        return dict(identity=self.identity(), effective=True, production_integrated=True,
                    closed=self._closed, failed=self._failed, active_scope=self._entered,
                    counters=dict(self.counters), width_calls=dict(self._hits), width_rows=dict(self._width_rows),
                    fallback_reasons=dict(self._fallbacks), startup=copy.deepcopy(self._startup),
                    counter_scope="Actual eager target returns; includes rejected verification rows; old graph replay excluded")

    def verify_metadata(self):
        try:
            self._health()
            self._configuration()
            core = self._core
            if (self._entered or _sources() != self._identity["source_sha256"]
                    or core.source_metadata() != self._identity["candidate_source"]
                    or core.validate_installed_torch(__import__("torch")) != self._identity["installed_torch"]
                    or _binary(self._runner._backend) != self._identity["extension"]
                    or self._runner.configs != self._identity["aten_m1_configs"]):
                raise RuntimeError("Mean-square runtime metadata/configuration changed")
        except BaseException:
            self._poison()
            raise

    def close(self):
        if self._closed:
            return
        if threading.current_thread() is not self._owner or self._entered:
            raise RuntimeError("Mean-square close requires its idle original owner")
        try:
            self._runner.close()  # Its original stream, even on a changed current stream.
        except BaseException:
            self._poison()
            raise
        self._closed = True  # Keep backend/source owners for status and failed-drain safety.


def initialize(model, *, strict=True):
    if getattr(model, "_mean_square_initialized", False):
        raise RuntimeError("Mean-square runtime cannot be initialized twice")
    model._mean_square_runtime = MeanSquareRuntime(model, strict=strict) if requested() else None
    model._mean_square_initialized = True
    return model._mean_square_runtime


def forward_scope(model):
    runtime = getattr(model, "_mean_square_runtime", None)
    if runtime is not None:
        return runtime.forward_scope(model)
    # A different Transformer invoked inside an owned forward must not inherit
    # that model's adapter merely because ContextVars follow lexical calls.
    return nullcontext() if _ACTIVE.get() is None else _outside_target()


@contextmanager
def _outside_target():
    token = _ACTIVE.set(None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def dispatch(original, flat):
    runtime = _ACTIVE.get()
    return None if runtime is None else runtime.mean_square(original, flat)


def configuration(model):
    runtime = getattr(model, "_mean_square_runtime", None)
    return (dict(requested=requested(), protocol=PROTOCOL, production_integrated=False)
            if runtime is None else runtime.identity())


def status(model):
    runtime = getattr(model, "_mean_square_runtime", None)
    return (dict(identity=configuration(model), effective=False, production_integrated=False,
                 closed=False, failed=False, active_scope=False, counters=dict.fromkeys(_COUNTERS, 0),
                 width_calls={}, width_rows={}, fallback_reasons={}) if runtime is None else runtime.report())
