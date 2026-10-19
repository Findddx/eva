"""Default-off ownership for the complete-forward HC verification IO arena.

Import is stdlib-only. Source hashes describe inputs, not a passed GPU gate.
Graphs retain their tensor owners; close drains but never clears those owners.
"""
from contextlib import contextmanager, nullcontext
import copy
import hashlib
from pathlib import Path
import threading


FLAG = "DS41_HC_VERIFY_IO_ARENA"
SUPPORTED_ROWS = (32, 64, 128)
LARGE_ROWS = (512, 1024)


def requested(environment):
    value = environment.get(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(FLAG + " must be 0 or 1")
    return value == "1"


def validate_configuration(environment, rows, shared_pool, strict, args, device):
    """Reject unsupported opt-in configurations before loading/capturing weights."""
    if not requested(environment):
        return False
    large = (isinstance(rows, (list, tuple)) and len(rows) == 1
             and type(rows[0]) is int and rows[0] in LARGE_ROWS)
    if (not rows or (not large and any(row not in SUPPORTED_ROWS for row in rows))
            or len(set(rows)) != len(rows) or not shared_pool
            or environment.get("DS41_NO_GRAPH", "0") == "1"):
        raise ValueError("HC IO arena requires shared-pool graphs at exact rows32/64/128 or single512/1024")
    if not strict or environment.get("DS41_STRICT_BATCH", "1") != "1":
        raise ValueError("HC IO arena requires strict target execution")
    if (args.dim, args.hc_mult, args.n_layers) != (5120, 4, 40):
        raise ValueError("HC IO arena production geometry requires40 layers, HC4 and dim5120")
    import torch
    target = torch.device(device)
    if target.type != "cuda" or torch.cuda.get_device_capability(target) != (12, 0):
        raise ValueError("HC IO arena production requires CUDA SM120")
    with torch.cuda.device(target):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("HC IO arena must be initialized outside graph capture")
    return True


def source_metadata():
    root = Path(__file__).parent
    names = ("hc_verify_runtime.py", "hc_verify_storage.py", "model.py", "kernels.py", "hc_cuda.py")
    values = {}
    for name in names:
        payload = (root / name).read_bytes()
        if len(payload) > 2 * 1024**2:
            raise ValueError("Unexpected HC source size")
        values[name] = hashlib.sha256(payload).hexdigest()
    return dict(scope="Local Python source inputs; not compiled-binary or GPU certification",
                sha256=values)


class HCVerifyArenaRuntime:
    """Authorize exactly one complete default stack/head per forward.

    The arena may use CPU only in structural fixtures. Production construction
    is separately guarded by validate_configuration and allocates before graphs.
    """
    def __init__(self, arena, *, layers, source=None):
        self.arena = arena
        self.layers = int(layers)
        self._source = source_metadata() if source is None else copy.deepcopy(source)
        self._owner_thread = threading.current_thread()
        self._active = False
        self._stack_entered = False
        self._stack_complete = False
        self._closed = False
        self._failed = False
        self._drain_errors = 0
        self.stats = dict(forward_calls=0, arena_forward_calls=0, original_shape_forward_calls=0,
                          graph_calls=0, graph_rows=0, failures=0, drains=0)

    def _health(self):
        if threading.current_thread() is not self._owner_thread:
            raise RuntimeError("HC runtime requires its construction CPU thread")
        if self._closed or self._failed or self.arena._failed:
            raise RuntimeError("HC runtime is closed/poisoned; discard model and graphs")

    def drain(self):
        # Drain does not release arena or graph references. Keep them consistent
        # even on failure, and let the original forward exception remain primary.
        self.stats["drains"] += 1
        if self.arena.device.type == "cuda":
            self.arena.torch.cuda.synchronize(self.arena.device)

    def _poison_and_drain(self):
        self._failed = True
        self.stats["failures"] += 1
        try:
            self.drain()
        except BaseException:
            self._drain_errors += 1

    def capture_rows(self, block, rows, dtype, ffn_dtype, shared_pool):
        """Capture original HC math into external IO; retain owners on errors."""
        self._health()
        if shared_pool is None or (block.dim, block.hc_mult) != (self.arena.dim, self.arena.hc_mult):
            raise ValueError("HC arena capture requires matching block geometry and shared pool")
        torch, arena = self.arena.torch, self.arena
        stream = shared_pool.capture_stream
        arena.check_capture_owner(stream)
        pair = []
        try:
            for kind, (label, hidden_dtype) in enumerate((("attn", dtype), ("ffn", ffn_dtype))):
                io = arena.buffers(rows, kind, hidden_dtype)
                fn, scale, base = (getattr(block, f"hc_{label}_{suffix}") for suffix in ("fn", "scale", "base"))
                stream.wait_stream(torch.cuda.current_stream(arena.device))
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        block._hc_apply(fn, scale, base, io.pre, io.post, io.comb, io.hidden)
                torch.cuda.current_stream(arena.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=shared_pool.handle, stream=stream):
                    result = block._hc_apply(fn, scale, base, io.pre, io.post, io.comb, io.hidden)
                    if result is not None:
                        raise RuntimeError("HC arena capture must not return captured tensors")
                shared_pool.check_idle()
                pair.append((graph, io.hidden, io.pre, io.post, io.comb))
            return pair
        except BaseException:
            # pair, graph, io, arena and shared pool are all still strongly held
            # while any already enqueued capture/warmup consumers drain.
            self._poison_and_drain()
            raise

    @contextmanager
    def forward_scope(self, input_ids):
        self._health()
        if self._active:
            raise RuntimeError("HC runtime forbids nested/interleaved complete forwards")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("HC arena supports complete B1 forwards only")
        self.arena._bind_replay_stream()  # Also bind original M1/eager forwards.
        if self.arena.device.type == "cuda":
            with self.arena.torch.cuda.device(self.arena.device):
                if self.arena.torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("HC arena complete forward cannot run inside graph capture")
        rows = int(input_ids.shape[1])
        use_arena = rows in self.arena.lengths
        self._active, self._stack_entered, self._stack_complete = True, False, False
        self.stats["forward_calls"] += 1
        self.stats["arena_forward_calls" if use_arena else "original_shape_forward_calls"] += 1
        try:
            with self.arena.replay_scope(rows) if use_arena else nullcontext():
                yield
                if not self._stack_complete:
                    raise RuntimeError("HC complete forward did not consume its final head pre")
        except BaseException:
            self._poison_and_drain()
            raise
        finally:
            self._active = False

    @contextmanager
    def stack_scope(self, layers):
        """Public run_logged cannot independently enter or expose borrowed pre."""
        self.begin_stack(layers)
        yield
        self.finish_stack()

    def begin_stack(self, layers):
        self._health()
        if not self._active or self._stack_entered:
            raise RuntimeError("HC arena run_logged requires one authorized default complete forward")
        if tuple(layers) != tuple(range(self.layers)):
            raise ValueError("HC arena rejects partial/reordered/resumed stacks")
        self._stack_entered = True

    def finish_stack(self):
        self._health()
        if not self._active or not self._stack_entered or self._stack_complete:
            raise RuntimeError("HC head completion requires its authorized stack")
        self._stack_complete = True

    def check_graph(self, rows):
        self._health()
        if not self._active or not self._stack_entered or self._stack_complete:
            raise RuntimeError("HC graph requires its authorized complete stack/head scope")
        self.arena.check_replay_owner(rows)

    def record_graph(self, rows):
        self.stats["graph_calls"] += 1
        self.stats["graph_rows"] += int(rows)

    def identity(self):
        a = self.arena
        result = dict(requested=True, enabled=True, variant="complete_forward_hc_io_arena_v1",
                    lengths=list(a.lengths), max_rows=a.max_rows, hidden_dtypes=[str(x) for x in a.hidden_dtypes],
                    external_bytes=a.external_bytes, backing_allocations=8, M1_storage_shared=False,
                    source=copy.deepcopy(self._source),
                    certification="Production ownership integration requires a new full-model GPU gate")
        if a.lengths in ((512,), (1024,)):
            result.update(variant="complete_forward_hc_io_arena_large_v1",
                          exact_single_large_schedule=True,
                          GPU_math_validated=False, full_model_validated=False)
        return result

    def report(self):
        storage = self.arena.metadata()
        storage["production_integrated"] = True
        return dict(identity=self.identity(), storage=storage, stats=dict(self.stats),
                    production_integrated=True, GPU_math_validated=False, performance_measured=False,
                    closed=self._closed, failed=self._failed, drain_errors=self._drain_errors,
                    graph_storage_retained_until_model_release=True,
                    budget_scope="External arena allocated before HC capture and expert GPU budget; graph scratch is separately reported")

    def verify_metadata(self):
        if source_metadata() != self._source:
            raise RuntimeError("HC arena source identity changed during execution")

    def close(self):
        if self._closed:
            return
        if threading.current_thread() is not self._owner_thread:
            raise RuntimeError("HC runtime close requires its construction CPU thread")
        if self._active:
            raise RuntimeError("Cannot close HC runtime during a live complete forward")
        try:
            self.drain()
        finally:
            self._closed = True


def identity(model):
    runtime = getattr(model, "_hc_verify_io_runtime", None)
    return dict(requested=False, enabled=False) if runtime is None else runtime.identity()


def report(model):
    runtime = getattr(model, "_hc_verify_io_runtime", None)
    return dict(identity=identity(model), stats={}, production_integrated=False) if runtime is None else runtime.report()
