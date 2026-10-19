"""Default-off shared canonical attention dispatch; no CUDA/import work at import.

Source/build snapshots are taken once at construction. They describe provenance,
not a passed GPU numerical gate. Call counters describe successful enqueues.
"""
import copy
import hashlib
import os
from pathlib import Path
from ds41.verification import canonical_runtime_query_capacity


FLAGS = {"sparse": "DS41_CANONICAL_SPARSE_BATCH", "rope": "DS41_CANONICAL_ROPE_BATCH"}
QUERY_TILE = 32
MAX_QUERIES = canonical_runtime_query_capacity()


def requested_flags(environment=None):
    environment = os.environ if environment is None else environment
    result = {}
    for kind, name in FLAGS.items():
        value = environment.get(name, "0")
        if value not in ("0", "1"):
            raise ValueError(name + " must be 0 or 1")
        result[kind] = value == "1"
    return result


def _binary_snapshot(runner):
    path = getattr(runner._backend, "__file__", None)
    if path is None:
        raise RuntimeError("Canonical attention extension has no compiled file identity")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": Path(path).stat().st_size}


class CanonicalAttentionRuntime:
    """One workspace/owner stream shared by all target Attention layers.

    CPU, noncanonical, B!=1 and M1/outside-cap calls keep the original path and
    record skips. Unexpected eligible tensor geometry is an error in the native
    runner, never a reason to silently retry after a partial launch.
    """
    def __init__(self, flags, device, verification_cap, *, strict=True):
        import torch
        if type(verification_cap) is not int or not 1 <= verification_cap <= MAX_QUERIES:
            raise ValueError(f"Canonical attention requires verification capacity1..{MAX_QUERIES}")
        if MAX_QUERIES in (512,1024) and verification_cap != MAX_QUERIES:
            raise ValueError("Explicit large cap requires its own exact dispatch owner")
        self.flags = dict(flags)
        self.verification_cap = verification_cap
        if MAX_QUERIES in (512,1024):
            self._closed = self._failed = False
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.runners = {}
        self.sources = {}
        self._source_readers = {}
        self.binaries = {}
        self.workspaces = {}
        self.stats = {kind: dict(calls=0, rows=0, skipped_calls=0, skip_reasons={}, operations={})
                      for kind in FLAGS}
        self.available = (strict and self.device.type == "cuda"
                          and torch.cuda.get_device_capability(self.device) == (12, 0))
        self.unavailable_reason = ("noncanonical" if not strict else
                                   "non_cuda" if self.device.type != "cuda" else "unsupported_sm")
        self.memory = None
        if self.available:
            before = self._memory(torch)
        for kind, requested in self.flags.items():
            if not requested:
                continue
            if kind == "sparse":
                if MAX_QUERIES == 1024:
                    from ds41 import canonical_sparse1024 as candidate
                    constructor = candidate.CanonicalSparse1024Runner
                elif MAX_QUERIES == 512:
                    from ds41 import canonical_sparse512 as candidate
                    constructor = candidate.CanonicalSparse512Runner
                else:
                    from ds41 import canonical_sparse_batch as candidate
                    constructor = candidate.CanonicalSparseBatchRunner
                options = dict(query_tile=QUERY_TILE, max_queries=MAX_QUERIES)
            else:
                if MAX_QUERIES == 1024:
                    from ds41 import canonical_rope1024 as candidate
                    constructor = candidate.CanonicalRope1024Runner
                elif MAX_QUERIES == 512:
                    from ds41 import canonical_rope512 as candidate
                    constructor = candidate.CanonicalRope512Runner
                else:
                    from ds41 import canonical_rope_batch as candidate
                    constructor = candidate.CanonicalRopeBatchRunner
                options = dict(max_queries=MAX_QUERIES)
            self.sources[kind] = candidate.source_metadata()
            self._source_readers[kind] = candidate.source_metadata
            if self.available:
                runner = constructor(device=self.device, **options)
                if runner.metadata != self.sources[kind]:
                    raise RuntimeError("Canonical attention source changed during construction")
                self.runners[kind] = runner
                self.workspaces[kind] = dict(runner.workspace_metadata)
                self.binaries[kind] = _binary_snapshot(runner)
        if self.available:
            after = self._memory(torch)
            self.memory = dict(before=before, after=after,
                allocated_delta_bytes=after["allocated_bytes"]-before["allocated_bytes"],
                reserved_delta_bytes=after["reserved_bytes"]-before["reserved_bytes"],
                device_free_delta_bytes=before["device_free_bytes"]-after["device_free_bytes"],
                budget_scope="Constructed before generate computes expert GPU budget; one shared workspace per kind")

    def _memory(self, torch):
        return dict(allocated_bytes=torch.cuda.memory_allocated(self.device),
                    reserved_bytes=torch.cuda.memory_reserved(self.device),
                    device_free_bytes=torch.cuda.mem_get_info(self.device)[0])

    def use_batch(self, kind, x, canonical):
        self._check512()
        if not self.flags[kind]:
            return False
        reason = None
        if not canonical:
            reason = "noncanonical"
        elif not x.is_cuda:
            reason = "non_cuda"
        elif x.shape[0] != 1:
            reason = "batch_not_one"
        elif x.shape[1] == 1:
            reason = "single_query_original_path"
        elif not 2 <= x.shape[1] <= self.verification_cap:
            reason = "outside_verification_capacity"
        elif not self.available:
            reason = self.unavailable_reason
        if reason is not None:
            stats = self.stats[kind]
            stats["skipped_calls"] += 1
            stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0)+1
            return False
        if x.device != self.device or kind not in self.runners:
            raise RuntimeError("Eligible canonical attention has no matching shared device runner")
        return True

    def _count(self, kind, rows, operation):
        stats = self.stats[kind]
        stats["calls"] += 1
        stats["rows"] += rows
        stats["operations"][operation] = stats["operations"].get(operation, 0)+1

    def rope_(self, value, freqs, *, inverse=False, operation):
        self._check512()
        try:
            result = self.runners["rope"].run_(value, freqs, inverse)
        except BaseException:
            if self.verification_cap in (512,1024):self._failed = True
            raise
        self._count("rope", value.shape[1], operation)
        return result

    def sparse(self, q, kv, sink, indices, scale):
        self._check512()
        try:
            result = self.runners["sparse"].run(q, kv, sink, indices, scale)
        except BaseException:
            if self.verification_cap in (512,1024):self._failed = True
            raise
        self._count("sparse", q.shape[1], "attention")
        return result

    def identity(self):
        result = dict(flags=dict(self.flags), sparse_query_tile=QUERY_TILE,
                    verification_cap=self.verification_cap, source_build=copy.deepcopy(self.sources),
                    compiled_extensions=copy.deepcopy(self.binaries),
                    compiled_binary_authenticated=False, gpu_numerical_gate_claimed=False)
        if self.verification_cap == 512:
            result.update(variant="independent_fixed512_original_M1_arithmetic_v1",
                          legacy_capacity_or_GPU_certificate_reused=False)
        elif self.verification_cap == 1024:
            result.update(variant="independent_fixed1024_original_M1_arithmetic_v1",
                          legacy_capacity_or_GPU_certificate_reused=False)
        return result

    def report(self):
        result = dict(identity=self.identity(), available=self.available,
                    unavailable_reason=None if self.available else self.unavailable_reason,
                    shared_runner_count=len(self.runners), workspace=copy.deepcopy(self.workspaces),
                    construction_memory=copy.deepcopy(self.memory), stats=copy.deepcopy(self.stats),
                    counter_scope="Successful native runner enqueues across all target layers, including prefill and diagnostics")
        if self.verification_cap == 512:
            result.update(closed=self._closed,failed=self._failed,
                          close_scope="Explicit close drains both independent512 construction streams; owners retained")
        elif self.verification_cap == 1024:
            result.update(closed=self._closed,failed=self._failed,
                          close_scope="Explicit close drains both independent1024 construction streams; owners retained")
        return result

    def _check512(self):
        if self.verification_cap in (512,1024) and (self._closed or self._failed):
            raise RuntimeError("Canonical large attention owner closed or poisoned")

    def close(self):
        """New512 only; leave original <=256 close behavior/metadata intact."""
        if self.verification_cap not in (512,1024) or self._closed:
            return
        first = None
        for kind, runner in self.runners.items():
            try:
                runner.close()
            except BaseException as exc:
                self._failed = True
                if first is None:first = exc
                elif hasattr(first,"add_note"):first.add_note("Additional "+kind+" close failure: "+repr(exc))
        if first is not None:
            raise first
        self._closed = True

    def verify_metadata(self):
        for kind, reader in self._source_readers.items():
            if reader() != self.sources[kind]:
                raise RuntimeError("Canonical attention source/build identity changed during execution")
        for kind, runner in self.runners.items():
            if _binary_snapshot(runner) != self.binaries[kind]:
                raise RuntimeError("Canonical attention compiled file changed during execution")


def initialize(model, *, strict=True):
    """Call after graph capture, before returning model to GPU-budget allocation."""
    flags = requested_flags()
    model.canonical_attention_runtime = None
    if not any(flags.values()):
        return
    if (model.args.n_heads, model.args.head_dim, model.args.rope_head_dim) != (64, 512, 64):
        raise ValueError("Canonical attention batch integration requires64heads/512dim/64RoPE geometry")
    context = getattr(model, "_capture_stream_context", None)
    if context is not None:
        context.check_replay_owner()
    runtime = CanonicalAttentionRuntime(flags, model.device, model.canonical_attention_cap, strict=strict)
    for layer in model.layers:
        layer.attn.canonical_batch_runtime = runtime
    model.canonical_attention_runtime = runtime


def identity(model):
    runtime = getattr(model, "canonical_attention_runtime", None)
    return (dict(flags={kind: False for kind in FLAGS}, sparse_query_tile=QUERY_TILE)
            if runtime is None else runtime.identity())


def report(model):
    runtime = getattr(model, "canonical_attention_runtime", None)
    return dict(identity=identity(model), shared_runner_count=0, stats={}) if runtime is None else runtime.report()


def verify_metadata(model):
    runtime = getattr(model, "canonical_attention_runtime", None)
    expected_flags = {kind: False for kind in FLAGS} if runtime is None else runtime.flags
    if requested_flags() != expected_flags:
        raise RuntimeError("Canonical attention requested flags changed during execution")
    if runtime is not None:
        runtime.verify_metadata()


def close(model):
    """Called by the explicit native3072 CLI owner, including failure cleanup."""
    runtime = getattr(model,"canonical_attention_runtime",None)
    if runtime is not None:
        runtime.close()


def counter_delta(before, after):
    result = {}
    for kind, stats in after.get("stats", {}).items():
        previous = before.get("stats", {}).get(kind, {})
        result[kind] = {name: stats[name]-previous.get(name, 0)
                        for name in ("calls", "rows", "skipped_calls")}
        for name in ("skip_reasons", "operations"):
            result[kind][name] = {key: value-previous.get(name, {}).get(key, 0)
                                 for key, value in stats[name].items()}
    return result
