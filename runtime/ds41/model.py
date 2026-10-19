"""V4.1 forward: single-pass mHC, sqrtsoftplus router, CSA2, conjugate RoPE.

Routed experts are not parameters. They are fetched from ExpertCache and
multiplied on the GPU. Dense EXL3 tensors stay resident.
"""
import json
import os
import threading
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from ds41.args import ModelArgs
from ds41.cache import ExpertCache
from ds41.engram import EngramRuntime
from ds41.embedding import embedding_lookup
from ds41.exl3_linear import build_linear
from ds41.attn_cuda import rope_tail_, sparse_decode_
from ds41.kernels import (
    apply_rotary_emb,
    dequant_fp4_group,
    fake_quant_fp8,
    hc_split_sinkhorn,
    precompute_freqs_cis,
    quant_fp4_group,
    sparse_attn,
    swiglu,
    window_indices,
)
from ds41.store import WeightStore
from ds41.verification import verification_max_rows, extended_verification_contract
from ds41.stable_linear_runtime import dispatch as _stable_linear_dispatch
from ds41.mean_square_runtime import dispatch as _mean_square_dispatch, forward_scope as _mean_square_scope
from ds41.hc_residual import (requested as _hc_residual_requested, bind_model as _bind_hc_residual,
    verify_model as _verify_hc_residual, pre_normalized_half as _hc_pre_bf16_half,
    post_bfloat16 as _hc_post_bf16)


_STRICT_DENSE = os.environ.get("DS41_STRICT_DENSE", "1") != "0"


def _write_window_tail(window, kv, pos0):
    """Write unique physical slots only; retain caller's full causal KV tensor.

    A verification longer than SWA may revisit slots. Last-window selection
    happens before index_copy_ so CUDA never gets duplicate destination IDs.
    Existing <=128/prefill operations retain their original indices/order.
    """
    rows, width = kv.shape[1], window.shape[1]
    first = max(0, rows - width)
    slots = torch.arange(pos0 + first, pos0 + rows, device=kv.device) % width
    window[:kv.shape[0]].index_copy_(1, slots, kv[:, first:])


def _parse_hc_verify_graph_rows(value):
    """Bound optional exact-shape graph capture to four distinct short lengths."""
    if not isinstance(value, str):
        raise ValueError("DS41_HC_VERIFY_GRAPHS must be comma-separated row counts")
    if not value.strip():
        return ()
    parts = [part.strip() for part in value.split(",")]
    try:
        rows = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("DS41_HC_VERIFY_GRAPHS must be comma-separated row counts") from exc
    if (len(rows) > 4 or len(set(rows)) != len(rows)
            or any(str(row) != part or not 2 <= row <= verification_max_rows()
                   for row, part in zip(rows, parts))):
        raise ValueError("DS41_HC_VERIFY_GRAPHS requires at most four distinct lengths in 2..verification_max_rows()")
    return rows


class _HCVerifySharedPool:
    """Scratch-only graph pool: independent outputs and serialized replay."""
    def __init__(self, device):
        if torch.cuda.memory.get_allocator_backend() != "native":
            raise RuntimeError("HC verification shared pool requires the native CUDA allocator")
        self.handle = torch.cuda.graph_pool_handle()
        self.capture_stream = torch.cuda.Stream(device=device)
        self.device = self.capture_stream.device
        self._replay_owner = None
        self._owner_lock = threading.Lock()
        self.memory = dict(reserved_bytes=0, active_bytes=0, segments=0)

    def check_idle(self):
        """No captured allocation may escape into a later independent graph."""
        torch.cuda.synchronize(self.device)
        segments = torch.cuda.memory_snapshot()
        if not isinstance(segments, list) or any(
                not isinstance(segment, dict) or "segment_pool_id" not in segment
                or not isinstance(segment["segment_pool_id"], (tuple, list))
                or len(segment["segment_pool_id"]) != 2 for segment in segments):
            raise RuntimeError("HC shared pool cannot verify allocator snapshot schema")
        own = [segment for segment in segments if tuple(segment["segment_pool_id"]) == tuple(self.handle)]
        if not own:
            raise RuntimeError("HC shared pool is missing from allocator snapshot")
        try:
            idle = all(segment["allocated_size"] == 0 and segment["active_size"] == 0
                       and all(block["state"] == "inactive" for block in segment["blocks"])
                       for segment in own)
            reserved = sum(segment["total_size"] for segment in own)
        except (KeyError, TypeError) as exc:
            raise RuntimeError("HC shared pool cannot verify allocator snapshot schema") from exc
        if not idle:
            raise RuntimeError("HC shared pool has live captured allocations; sharing is unsafe")
        self.memory = dict(reserved_bytes=reserved, active_bytes=0, segments=len(own))

    def check_replay_owner(self):
        # A permanent thread/stream binding makes every shared replay enqueue
        # serial without a CUDA synchronization in the decode path.
        stream = torch.cuda.current_stream(self.device)
        owner = (threading.current_thread(), stream.device, stream.cuda_stream)
        with self._owner_lock:
            if self._replay_owner is None:
                self._replay_owner = owner
            elif self._replay_owner != owner:
                raise RuntimeError("HC shared-pool replay requires its original CPU thread and CUDA stream")


def _capture_workspace_config():
    values = {name: os.environ.get(name) for name in (
        "CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE", "TORCH_CUBLASLT_UNIFIED_WORKSPACE")}
    for name in ("cublas_workspace_size", "cublaslt_workspace_size"):
        getter = getattr(torch.backends.cuda, name, None)
        values[name] = None if getter is None else int(getter())
    return values


class _SharedCaptureStream:
    """Keep stream-keyed library workspaces alive, with serial graph replay."""
    def __init__(self, device):
        self.stream = torch.cuda.Stream(device=device)
        self.device = self.stream.device
        self.workspace_config = _capture_workspace_config()
        self._replay_owner = None
        self._owner_lock = threading.Lock()

    def check_replay_owner(self):
        if _capture_workspace_config() != self.workspace_config:
            raise RuntimeError("Shared-capture workspace configuration changed; captured addresses may be invalid")
        stream = torch.cuda.current_stream(self.device)
        owner = (threading.current_thread(), stream.device, stream.cuda_stream)
        with self._owner_lock:
            if self._replay_owner is None:
                self._replay_owner = owner
            elif self._replay_owner != owner:
                raise RuntimeError("Shared-capture graph replay requires its original CPU thread and CUDA stream")


def _check_capture_stream(capture_stream, replay_guard):
    if (capture_stream is None) != (replay_guard is None) or (
            capture_stream is not None and replay_guard.stream is not capture_stream):
        raise ValueError("An explicit capture stream requires its matching replay guard")


def stable_linear(x, weight, bias=None):
    """Use the canonical M1 FP32 reduction for short causal verification."""
    flat = x.reshape(-1, x.shape[-1])
    if _STRICT_DENSE and flat.shape[0] <= verification_max_rows():
        rows = [F.linear(flat[i:i + 1], weight, bias) for i in range(flat.shape[0])]
        y = rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)
        return y.reshape(*x.shape[:-1], weight.shape[0])
    return F.linear(x, weight, bias)


def mean_square(x):
    """Keep short-row RMS reductions independent of the number of tokens."""
    flat = x.reshape(-1, x.shape[-1]).float()
    if _STRICT_DENSE and flat.shape[0] <= verification_max_rows():
        if flat.shape[0] > 1:
            candidate = _mean_square_dispatch(x, flat)
            if candidate is not None:
                return candidate.reshape(*x.shape[:-1], 1)
        rows = [flat[i:i + 1].square().mean(-1, keepdim=True) for i in range(flat.shape[0])]
        value = rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)
        return value.reshape(*x.shape[:-1], 1)
    return x.float().square().mean(-1, keepdim=True)


def engram_gate(x, key, value, weight, eps, dim):
    """Preserve the M1 HC-group reduction geometry for short verification."""
    def one(h, k, v):
        h = h.float()
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(k.square().mean(-1) + eps)
        dot = (h * weight * k).sum(-1) * rstd * dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (h + gate.unsqueeze(-1) * v.float().unsqueeze(-2)).to(x.dtype)

    # Keep all HC copies together: flattening HC into individual rows would
    # change the existing M1 target. The dense projection remains batched.
    if _STRICT_DENSE and 1 < x.shape[1] <= verification_max_rows():
        return torch.cat([one(x[:, i:i+1], key[:, i:i+1], value[:, i:i+1])
                          for i in range(x.shape[1])], dim=1)
    return one(x, key, value)


class State:
    def __init__(self):
        self.compress_kv = None
        self.index_k = None
        self.topk_idxs = None
        self.candidates = None


class Block(torch.nn.Module):
    def __init__(self, layer_id, args: ModelArgs, store: WeightStore, device):
        super().__init__()
        self.layer_id = layer_id
        if _hc_residual_requested():
            self._hc_residual_bf16 = True
        self.args = args
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.eps = args.norm_eps
        self.fused_hc = os.environ.get("DS41_FUSED_HC", "0") != "0"
        self._hc_verify_hits = {}
        p = f"layers.{layer_id}."
        self.attn_norm = store.get(p + "attn_norm.weight").to(device)
        self.ffn_norm = store.get(p + "ffn_norm.weight").to(device)
        self.hc_attn_fn = store.get(p + "hc_attn_fn").to(device).float()
        self.hc_ffn_fn = store.get(p + "hc_ffn_fn").to(device).float()
        self.hc_attn_base = store.get(p + "hc_attn_base").to(device).float()
        self.hc_ffn_base = store.get(p + "hc_ffn_base").to(device).float()
        self.hc_attn_scale = store.get(p + "hc_attn_scale").to(device).float()
        self.hc_ffn_scale = store.get(p + "hc_ffn_scale").to(device).float()
        self.gate_w = store.get(p + "ffn.gate.weight").to(device).float()
        self.gate_b = store.get(p + "ffn.gate.bias").to(device).float()
        self.gate_b_vl = store.get(p + "ffn.gate.bias_vl").to(device).float() if store.has(p + "ffn.gate.bias_vl") else None
        self.attn = Attention(layer_id, args, store, device)
        self.shared = [exl3(store, p + f"ffn.shared_experts.w{i}", device) for i in (1, 2, 3)]
        self.routed_k = 3.0
        self._sh_graph = None
        self.engram_wkv = exl3(store, p + "engram.wkv", device) if store.has(p + "engram.wkv.trellis") else None
        if self.engram_wkv is not None:
            self.engram_q = store.get(p + "engram.q_weight").to(device).float()
            self.engram_k = store.get(p + "engram.k_weight").to(device).float()

    def _hc_apply(self, fn, scale, base, pre, post, comb, hidden):
        flat = hidden.flatten(2).float()
        rsqrt = torch.rsqrt(mean_square(flat) + self.eps)
        mixes = stable_linear(flat, fn) * rsqrt
        p, o, c = self._hc_split(mixes, scale, base)
        pre.copy_(p)
        post.copy_(o)
        comb.copy_(c)

    def _hc_split(self, mixes, scale, base):
        cpu_rows = mixes.numel() // mixes.shape[-1]
        if _STRICT_DENSE and not mixes.is_cuda and 1 < cpu_rows <= verification_max_rows():
            # CUDA Sinkhorn already computes one row per thread. The torch CPU
            # fallback's vectorized sigmoid can round differently from its M1
            # SIMD tail, so keep the same row geometry in the reference path.
            flat = mixes.reshape(-1, mixes.shape[-1])
            rows = [hc_split_sinkhorn(flat[i:i + 1], scale, base, self.hc_mult,
                                     self.args.hc_sinkhorn_iters, self.args.hc_eps)
                    for i in range(flat.shape[0])]
            output = [torch.cat([row[j] for row in rows], dim=0) for j in range(3)]
            return tuple(value.reshape(*mixes.shape[:-1], *value.shape[1:]) for value in output)
        return hc_split_sinkhorn(mixes, scale, base, self.hc_mult, self.args.hc_sinkhorn_iters, self.args.hc_eps)

    def capture_hc(self, dtype=torch.float16, ffn_dtype=torch.float16, capture_stream=None, replay_guard=None):
        _check_capture_stream(capture_stream, replay_guard)
        self._hc_replay_guard = replay_guard
        if capture_stream is None:
            self._hc_pack = self._capture_hc_rows(1, dtype, ffn_dtype)
        else:
            self._hc_pack = self._capture_hc_rows(1, dtype, ffn_dtype, capture_stream=capture_stream)

    def _capture_hc_rows(self, rows, dtype, ffn_dtype, shared_pool=None, capture_stream=None, io_runtime=None):
        if io_runtime is not None:
            if capture_stream is not None:
                raise ValueError("HC arena uses its shared pool capture stream")
            return io_runtime.capture_rows(self, rows, dtype, ffn_dtype, shared_pool)
        if shared_pool is not None and capture_stream is not None:
            raise ValueError("HC capture cannot combine an explicit stream with a different shared pool")
        device = self.attn_norm.device
        pack = []
        specs = (
            (self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, dtype),
            (self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, ffn_dtype),
        )
        for fn, scale, base, hidden_dtype in specs:
            hidden = torch.empty(1, rows, self.hc_mult, self.dim, device=device, dtype=hidden_dtype)
            pre = torch.empty(1, rows, self.hc_mult, device=device, dtype=torch.float32)
            post = torch.empty(1, rows, self.hc_mult, device=device, dtype=torch.float32)
            comb = torch.empty(1, rows, self.hc_mult, self.hc_mult, device=device, dtype=torch.float32)
            # Shared captures must warm and capture on the same stream, so
            # stream-specific library workspaces are initialized outside it.
            stream = (shared_pool.capture_stream if shared_pool is not None else
                      capture_stream if capture_stream is not None else torch.cuda.Stream())
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._hc_apply(fn, scale, base, pre, post, comb, hidden)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            if shared_pool is None:
                kwargs = {} if capture_stream is None else dict(stream=stream)
                with torch.cuda.graph(graph, **kwargs):
                    self._hc_apply(fn, scale, base, pre, post, comb, hidden)
            else:
                with torch.cuda.graph(graph, pool=shared_pool.handle, stream=stream):
                    result = self._hc_apply(fn, scale, base, pre, post, comb, hidden)
                    if result is not None:
                        raise RuntimeError("HC shared capture must not return captured tensors")
                shared_pool.check_idle()
            pack.append((graph, hidden, pre, post, comb))
        return pack

    def capture_hc_verification(self, lengths, dtype=torch.float16, ffn_dtype=torch.float16, shared_pool=None, io_runtime=None):
        if getattr(self, "_hc_verify_io_runtime", None) is not None:
            raise RuntimeError("HC arena recapture requires discarding the model and graphs")
        if io_runtime is not None:
            if shared_pool is None:
                raise ValueError("HC arena requires its shared capture pool")
            io_runtime.arena.check_capture_owner(shared_pool.capture_stream)
        lengths = tuple(lengths)
        if (len(lengths) > 4 or len(set(lengths)) != len(lengths)
                or any(type(rows) is not int or not 2 <= rows <= verification_max_rows() for rows in lengths)):
            raise ValueError("HC verification graphs require at most four distinct exact lengths in 2..verification_max_rows()")
        self._hc_verify_pack = {}
        self._hc_verify_hits = {}
        self._hc_verify_shared_pool = shared_pool
        self._hc_verify_io_runtime = io_runtime
        for rows in lengths:
            try:
                # Publish only a complete independent attention/FFN pair.
                if io_runtime is not None:
                    pair = self._capture_hc_rows(rows, dtype, ffn_dtype, shared_pool=shared_pool, io_runtime=io_runtime)
                elif shared_pool is None:
                    pair = self._capture_hc_rows(rows, dtype, ffn_dtype)
                else:
                    pair = self._capture_hc_rows(rows, dtype, ffn_dtype, shared_pool=shared_pool)
                self._hc_verify_pack[rows] = pair
                self._hc_verify_hits.update({(rows, 0): 0, (rows, 1): 0})
            except Exception as exc:
                if shared_pool is not None:
                    # A partially captured shared pool is not a safe fallback.
                    raise RuntimeError(f"HC shared capture failed at layer {self.layer_id} rows {rows}") from exc
                print(f"hc verify graph skip layer {self.layer_id} rows {rows}: {type(exc).__name__}: {exc}", flush=True)

    def hc_mixes(self, x, fn, scale, base, kind=0):
        pack = getattr(self, "_hc_pack", None)
        if pack and x.shape[:2] == (1, 1):
            graph, hidden, pre, post, comb = pack[kind]
            if x.dtype == hidden.dtype and x.shape == hidden.shape:
                guard = getattr(self, "_hc_replay_guard", None)
                if guard is not None:
                    guard.check_replay_owner()
                hidden.copy_(x)
                graph.replay()
                return pre, post, comb
        if x.shape[0] == 1 and x.shape[1] > 1:
            pack = getattr(self, "_hc_verify_pack", {}).get(x.shape[1])
            if pack:
                graph, hidden, pre, post, comb = pack[kind]
                io_runtime = getattr(self, "_hc_verify_io_runtime", None)
                if io_runtime is not None:
                    io_runtime.check_graph(x.shape[1])
                    if x.dtype != hidden.dtype or x.device != hidden.device or x.shape != hidden.shape:
                        raise ValueError("HC arena input differs from its exact captured geometry")
                if x.dtype == hidden.dtype and x.device == hidden.device and x.shape == hidden.shape:
                    shared_pool = getattr(self, "_hc_verify_shared_pool", None)
                    if shared_pool is not None:
                        shared_pool.check_replay_owner()
                    hidden.copy_(x)
                    graph.replay()
                    key = (x.shape[1], kind)
                    self._hc_verify_hits[key] = self._hc_verify_hits.get(key, 0) + 1
                    if io_runtime is not None:
                        io_runtime.record_graph(x.shape[1])
                    return pre, post, comb
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(mean_square(flat) + self.eps)
        mixes = stable_linear(flat, fn) * rsqrt
        return self._hc_split(mixes, scale, base)

    def hc_pre(self, x, pre):
        return torch.sum(pre.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)

    def hc_pre_norm(self, x, pre, norm):
        if getattr(self, "_hc_residual_bf16", False):
            return _hc_pre_bf16_half(x, pre, norm, self.eps, rms)
        if getattr(self, "fused_hc", False):
            from ds41.hc_fused import hc_pre_rms
            fast = hc_pre_rms(x, pre, norm, self.eps)
            if fast is not None:
                return fast
        return rms(self.hc_pre(x, pre), norm, self.eps)

    def hc_post(self, x, residual, post, comb):
        if getattr(self, "_hc_residual_bf16", False):
            return _hc_post_bf16(x, residual, post, comb)
        if getattr(self, "fused_hc", False):
            from ds41.hc_fused import hc_post
            fast = hc_post(x, residual, post, comb)
            if fast is not None:
                return fast
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.float().unsqueeze(-2), dim=2)
        return y.to(x.dtype)

    def route(self, x, image_mask):
        runtime = getattr(self, "_stable_linear_runtime", None)
        scores = (stable_linear(x.float(), self.gate_w) if runtime is None else
                  _stable_linear_dispatch(runtime, "router", self.layer_id, x.float(), self.gate_w,
                                          fallback=stable_linear))
        scores = F.softplus(scores).sqrt()
        bias = self.gate_b
        if image_mask is not None and self.gate_b_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.gate_b_vl, bias)
        idx = (scores + bias).topk(self.args.n_activated_experts, dim=-1).indices
        w = scores.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * self.args.route_scale
        return w, idx

    def _shared_run(self):
        gate = self.shared[0].forward(self._sh_x)
        up = self.shared[2].forward(self._sh_x)
        mid = swiglu(gate.float(), up.float(), self.args.swiglu_limit)
        self._sh_y.copy_(self.shared[1].forward(mid.to(torch.float16)).float())

    def capture_shared(self, capture_stream=None, replay_guard=None):
        _check_capture_stream(capture_stream, replay_guard)
        self._sh_replay_guard = replay_guard
        device = self.attn_norm.device
        self._sh_x = torch.empty(1, self.dim, device=device, dtype=torch.float16)
        self._sh_y = torch.empty(1, self.dim, device=device, dtype=torch.float32)
        stream = torch.cuda.Stream() if capture_stream is None else capture_stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._shared_run()
        torch.cuda.current_stream().wait_stream(stream)
        self._sh_graph = torch.cuda.CUDAGraph()
        kwargs = {} if capture_stream is None else dict(stream=stream)
        with torch.cuda.graph(self._sh_graph, **kwargs):
            self._shared_run()

    def _shared(self, flat):
        if self._sh_graph is not None and flat.shape[0] == 1:
            guard = getattr(self, "_sh_replay_guard", None)
            if guard is not None:
                guard.check_replay_owner()
            self._sh_x.copy_(flat)
            self._sh_graph.replay()
            return self._sh_y
        gate = self.shared[0].forward(flat)
        up = self.shared[2].forward(flat)
        mid = swiglu(gate.float(), up.float(), self.args.swiglu_limit)
        return self.shared[1].forward(mid.to(torch.float16)).float()

    def experts(self, x, cache: ExpertCache, image_mask):
        flat = x.reshape(-1, self.dim).to(torch.float16)
        weights, indices = self.route(flat, None if image_mask is None else image_mask.reshape(-1))
        prefetch = getattr(cache, "next_chunk_prefetch_active", False)
        route_scope = (cache.route_snapshot_scope(self.layer_id, flat, indices, weights,
                                                  self.routed_k, self.args.moe_inter_dim)
                       if hasattr(cache, "route_snapshot_scope") else nullcontext(None))
        with route_scope as route_window:
            cpu_routes = None
            if os.environ.get("DS41_ROUTE_AWARE_CHUNKS", "0") == "1":
                from ds41.route_chunks import plan_route_chunks
                # One additional host transfer of actual routes. cache.moe retains
                # its independent active-union/capacity checks and source lifetime.
                bounds = (cache.moe_chunk_limits(flat, indices.shape[-1], self.routed_k, self.args.moe_inter_dim)
                          if hasattr(cache, "moe_chunk_limits") else {})
                cpu_routes = route_window.rows if route_window is not None else indices.detach().tolist()
                spans = plan_route_chunks(cpu_routes, cache.frozen, self.layer_id, cache.stage_limit, **bounds)
                copy_parts = len(spans) > 1
            else:
                if prefetch:
                    raise ValueError("next-chunk prefetch requires DS41_ROUTE_AWARE_CHUNKS=1")
                rows_per_chunk = max(1, min(42, cache.stage_limit // self.args.n_activated_experts))
                spans = ((start, start + rows_per_chunk) for start in range(0, flat.shape[0], rows_per_chunk))
                copy_parts = flat.shape[0] > rows_per_chunk
            parts = []
            try:
                for number, (start, end) in enumerate(spans):
                    options = {}
                    if route_window is not None:
                        options["_route_snapshot"] = route_window.span(start, end)
                    if prefetch and number + 1 < len(spans):
                        lo, hi = spans[number+1]
                        options["next_ids"] = [expert for row in cpu_routes[lo:hi] for expert in row]
                    part = cache.moe(
                        self.layer_id, flat[start:end], indices[start:end], weights[start:end], self.routed_k,
                        self.args.swiglu_limit, self.args.moe_inter_dim, **options,
                    )
                    # Runners reuse output scratch by row count. A later equal-sized
                    # chunk must not overwrite an earlier chunk before concatenation.
                    parts.append(part.clone() if copy_parts else part)
            finally:
                if prefetch:
                    cache.finish_next_chunk_prefetch(self.layer_id)
        y = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        y = y + self._shared(flat)
        return y.to(x.dtype).view_as(x)

    def apply_engram(self, x, rows):
        if self.engram_wkv is None or rows is None:
            return x
        # Real FP8 values multiplied by their per-32-channel scales. Byte values
        # are never valid floating point table entries.
        flat = rows.to(device=x.device, dtype=torch.float16).flatten(-2)
        if flat.shape[-1] != self.engram_wkv.in_features:
            raise ValueError(f"Engram input width {flat.shape[-1]} != {self.engram_wkv.in_features}")
        kv = self.engram_wkv.forward(flat)
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.engram_q * self.engram_k
        return engram_gate(x, key, value, weight, self.eps, self.dim)

    def forward(self, x, pos0, pre_mix, cache, state: State, image_mask=None, engram_rows=None):
        promotion = getattr(cache, "_layer_promotion", None)
        if promotion is not None and promotion.active:
            if x.shape[0] != 1:
                cache.abort_layer_promotion()
                raise ValueError("Layer promotion supports target batch B=1 only")
            cache.begin_layer_promotion(self.layer_id)
        try:
            prof = getattr(cache, "profile", False)

            def tick():
                if not prof:
                    return 0.0
                torch.cuda.synchronize()
                return time.perf_counter()

            t = tick()
            if x.shape[1] == 1:
                cache.prefetch_gpu(self.layer_id)
            if engram_rows is not None:
                x = self.apply_engram(x, engram_rows)
            residual = x
            attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, 0)
            h = self.hc_pre_norm(x, pre_mix, self.attn_norm)
            if prof:
                cache.t_hc += time.perf_counter() - t
            t = tick()
            h = self.attn(h, pos0, state)
            if prof:
                cache.t_attn += time.perf_counter() - t
            t = tick()
            x = self.hc_post(h, residual, attn_post, attn_comb)
            residual = x
            ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, 1)
            h = self.hc_pre_norm(x, attn_pre, self.ffn_norm)
            if prof:
                cache.t_hc += time.perf_counter() - t
            t = tick()
            h = self.experts(h, cache, image_mask)
            if prof:
                cache.t_moe += time.perf_counter() - t
            x = self.hc_post(h, residual, ffn_post, ffn_comb)
            if promotion is not None and promotion.active:
                cache.finish_layer_promotion(self.layer_id)
            return x, ffn_pre
        except BaseException:
            if promotion is not None and promotion.active:
                cache.abort_layer_promotion()
            raise


class Attention(torch.nn.Module):
    def __init__(self, layer_id, args: ModelArgs, store: WeightStore, device):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.device = device
        self.fused_kv_prep = os.environ.get("DS41_FUSED_KV_PREP", "0") != "0"
        p = f"layers.{layer_id}.attn."
        self.wq_a = exl3(store, p + "wq_a", device)
        self.wq_b = exl3(store, p + "wq_b", device)
        self.wkv = exl3(store, p + "wkv", device)
        self.wo_b = exl3(store, p + "wo_b", device)
        self.wo_a = [exl3(store, p + f"wo_a.slice.{i}", device) for i in range(args.o_groups)]
        self.q_norm = store.get(p + "q_norm.weight").to(device)
        self.kv_norm = store.get(p + "kv_norm.weight").to(device)
        self.attn_sink = store.get(p + "attn_sink").to(device).float()
        self.compress_ratio = args.compress_ratios[layer_id]
        self.is_kv_source = layer_id in args.kv_source_layers
        self.is_index_source = layer_id in args.index_source_layers
        self.compressor = None
        self.indexer = None
        if self.is_kv_source:
            self.compressor = exl3(store, p + "compressor.wkv", device)
            self.compressor_norm = store.get(p + "compressor.norm.weight").to(device)
            self.wgate = exl3(store, p + "compressor.wgate", device) if store.has(p + "compressor.wgate.trellis") else None
            ratio = self.compress_ratio
            self.register_buffer("kv_state", torch.zeros(args.max_batch_size, max(ratio, 1), args.head_dim, device=device))
            self.register_buffer("score_state", torch.full((args.max_batch_size, max(ratio, 1), args.head_dim), -torch.inf, device=device))
            slots = args.max_seq_len // max(ratio, 1)
            self.register_buffer("ck_codes", torch.zeros(args.max_batch_size, slots, args.head_dim // 2, dtype=torch.uint8, device=device))
            self.register_buffer("ck_scales", torch.zeros(args.max_batch_size, slots, args.head_dim // 16, dtype=torch.float8_e4m3fn, device=device))
        if self.is_index_source:
            self.indexer_wq = exl3(store, p + "indexer.wq_b", device)
            self.indexer_wn = store.get(p + "indexer.weights_proj.weight").to(device).float()
            self.indexer_wk = None
            self.indexer_kn = None
            if store.has(p + "indexer.wk.trellis"):
                self.indexer_wk = exl3(store, p + "indexer.wk", device)
                self.indexer_kn = store.get(p + "indexer.k_norm.weight").to(device)
                slots = args.max_seq_len // max(self.compress_ratio, 1)
                hd = args.index_head_dim
                self.register_buffer("ik_codes", torch.zeros(args.max_batch_size, slots, hd // 2, dtype=torch.uint8, device=device))
                # Indexer FP4 uses power-of-two E8M0 scales. Float32 stores
                # those powers exactly even on PyTorch builds without E8M0.
                self.register_buffer("ik_scales", torch.zeros(args.max_batch_size, slots, max(hd // 32, 1), dtype=torch.float32, device=device))
        self.freqs = None
        self.qkv_graph = None
        self.wo_graph = None
        self.register_buffer("window", torch.zeros(args.max_batch_size, args.window_size, args.head_dim,
                                                   dtype=torch.float16, device=device))

    def bind_freqs(self, swa, compressed):
        self.freqs = compressed if self.compress_ratio else swa

    def capture_decode_graphs(self, capture_stream=None, replay_guard=None):
        """Graph the position-independent projections. Replay copies a new hidden state in."""
        _check_capture_stream(capture_stream, replay_guard)
        self._projection_replay_guard = replay_guard
        dim = self.args.dim
        device = self.device
        self._sx = torch.empty(1, 1, dim, device=device, dtype=torch.float16)
        self._so = torch.empty(1, 1, self.args.n_heads, self.args.head_dim, device=device, dtype=torch.float16)
        stream = torch.cuda.Stream() if capture_stream is None else capture_stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._qkv_run()
                self._wo_run()
        torch.cuda.current_stream().wait_stream(stream)
        kwargs = {} if capture_stream is None else dict(stream=stream)
        self.qkv_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.qkv_graph, **kwargs):
            self._qr, self._q, self._kv = self._qkv_run()
        self.wo_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.wo_graph, **kwargs):
            self._yo = self._wo_run()

    def _qkv_run(self):
        qr = rms(self.wq_a.forward(self._sx), self.q_norm, self.args.norm_eps)
        q = self.wq_b.forward(qr)
        kv = rms(self.wkv.forward(self._sx), self.kv_norm, self.args.norm_eps)
        return qr, q, kv

    def _wo_run(self):
        o = self._so.reshape(1, 1, self.args.o_groups, -1)
        fake_quant_fp8(o)
        parts = [self.wo_a[g].forward(o[:, :, g, :].contiguous()) for g in range(self.args.o_groups)]
        y = torch.stack(parts, dim=2).flatten(2)
        return self.wo_b.forward(y)

    def project_qkv(self, x):
        if self.qkv_graph is None or x.shape[1] != 1:
            qr = rms(self.wq_a.forward(x), self.q_norm, self.args.norm_eps)
            q = self.wq_b.forward(qr)
            kv = rms(self.wkv.forward(x), self.kv_norm, self.args.norm_eps)
            return qr, q, kv
        guard = getattr(self, "_projection_replay_guard", None)
        if guard is not None:
            guard.check_replay_owner()
        self._sx.copy_(x.reshape(1, 1, -1).to(torch.float16))
        self.qkv_graph.replay()
        return self._qr, self._q, self._kv

    def project_out(self, o):
        if self.wo_graph is None or o.shape[0] != 1 or o.shape[1] != 1:
            og = o.reshape(o.shape[0], o.shape[1], self.args.o_groups, -1)
            fake_quant_fp8(og)
            parts = [self.wo_a[g].forward(og[:, :, g, :].contiguous()) for g in range(self.args.o_groups)]
            y = torch.stack(parts, dim=2).flatten(2)
            return self.wo_b.forward(y)
        guard = getattr(self, "_projection_replay_guard", None)
        if guard is not None:
            guard.check_replay_owner()
        self._so.copy_(o.reshape(1, 1, self.args.n_heads, self.args.head_dim))
        self.wo_graph.replay()
        return self._yo

    def _compress(self, x, pos0):
        bsz, seqlen, _ = x.shape
        ratio = self.compress_ratio
        if self.wgate is None:
            return rms(self.compressor.forward(x), self.compressor_norm, self.args.norm_eps)
        kv = self.compressor.forward(x).float()
        score = self.wgate.forward(x).float()
        if getattr(self, "_capture_verification", False):
            # These fresh FP32 outputs are not mutated below. Retain raw input
            # rows, before prepending the old partial group, for prefix commit.
            self._verification_raw = (kv.detach(), score.detach())
        if pos0 > 0 and seqlen == 1:
            slot = pos0 % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if (pos0 + 1) % ratio != 0:
                return None
            pooled = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(1)).sum(1, keepdim=True)
            # Completed-group slots are no longer model state. Canonicalize
            # them so streaming and chunked verification save identical state.
            self.kv_state[:bsz].zero_()
            self.score_state[:bsz].fill_(-torch.inf)
            return rms(pooled.to(x.dtype), self.compressor_norm, self.args.norm_eps)
        # A verification chunk may start midway through a compression group.
        # Prefix it with the cached partial group, then retain its new tail.
        old = pos0 % ratio
        if old:
            kv = torch.cat([self.kv_state[:bsz, :old], kv], dim=1)
            score = torch.cat([self.score_state[:bsz, :old], score], dim=1)
        self.kv_state.zero_()
        self.score_state.fill_(-torch.inf)
        remainder = kv.shape[1] % ratio
        cutoff = kv.shape[1] - remainder
        if remainder:
            self.kv_state[:bsz, :remainder] = kv[:, cutoff:]
            self.score_state[:bsz, :remainder] = score[:, cutoff:]
        if not cutoff:
            return None
        kv_g = kv[:, :cutoff].unflatten(1, (-1, ratio))
        sc_g = score[:, :cutoff].unflatten(1, (-1, ratio))
        pooled = (kv_g * sc_g.softmax(2)).sum(2)
        return rms(pooled.to(x.dtype), self.compressor_norm, self.args.norm_eps)

    def forward(self, x, pos0, state: State):
        bsz, seqlen, _ = x.shape
        args = self.args
        canonical = _STRICT_DENSE and seqlen <= verification_max_rows()
        batch_runtime = getattr(self, "canonical_batch_runtime", None)
        batch_rope = batch_runtime is not None and batch_runtime.use_batch("rope", x, canonical)
        freqs = self.freqs[pos0 : pos0 + seqlen]
        qr, q, kv = self.project_qkv(x)
        q = q.unflatten(-1, (args.n_heads, args.head_dim))
        if batch_rope:
            batch_runtime.rope_(q, freqs, operation="query_forward")
            batch_runtime.rope_(kv, freqs, operation="kv_forward")
        elif x.is_cuda and canonical and args.rope_head_dim == 64:
            for i in range(seqlen):
                if not rope_tail_(q[:, i:i + 1], freqs[i:i + 1], False):
                    apply_rotary_emb(q[:, i:i + 1, ..., -args.rope_head_dim :], freqs[i:i + 1])
                if not rope_tail_(kv[:, i:i + 1], freqs[i:i + 1], False):
                    apply_rotary_emb(kv[:, i:i + 1, -args.rope_head_dim :], freqs[i:i + 1])
        else:
            apply_rotary_emb(q[..., -args.rope_head_dim :], freqs)
            apply_rotary_emb(kv[..., -args.rope_head_dim :], freqs)
        fake_quant_fp8(kv)
        if getattr(self, "_capture_verification", False) and seqlen > args.window_size:
            # A verification can wrap a small ring more than once.
            # A rejected suffix may otherwise destroy accepted ring entries.
            self._verification_window = kv.detach()
        if seqlen == 1 and pos0 > 0:
            self.window[:bsz, pos0 % args.window_size] = kv.squeeze(1)
            window_kv = self.window[:bsz]
            topk = window_indices(args.window_size, bsz, seqlen, pos0, x.device)
        elif pos0 == 0:
            # The physical ring slot is the absolute token position % window.
            # Simply copying the final window rotates it incorrectly whenever
            # prompt length is not a multiple of the window size.
            _write_window_tail(self.window, kv, pos0)
            window_kv = kv
            topk = window_indices(args.window_size, bsz, seqlen, 0, x.device)
            if canonical:
                positions = torch.arange(seqlen, device=x.device).unsqueeze(-1) - args.window_size + 1
                positions = positions + torch.arange(args.window_size, device=x.device)
                topk = torch.where(positions >= 0, positions, -1).int().unsqueeze(0).expand(bsz, -1, -1).contiguous()
        else:
            # Each verification query sees its previous window and the causal
            # prefix of this chunk. Historical ring rows are put in time order.
            previous_start = max(0, pos0 - args.window_size + 1)
            previous = torch.arange(previous_start, pos0, device=x.device) % args.window_size
            history = self.window[:bsz].index_select(1, previous)
            window_kv = torch.cat([history, kv], dim=1)
            end = previous.numel() + torch.arange(seqlen, device=x.device).unsqueeze(-1)
            width = min(args.window_size, window_kv.shape[1])
            topk = (end - args.window_size + 1).clamp_min(0) + torch.arange(width, device=x.device)
            topk = torch.where(topk <= end, topk, -1).int().unsqueeze(0).expand(bsz, -1, -1).contiguous()
            if canonical:
                positions = torch.arange(pos0, pos0 + seqlen, device=x.device).unsqueeze(-1) - args.window_size + 1
                positions = positions + torch.arange(args.window_size, device=x.device)
                local = positions - previous_start
                valid = (positions >= 0) & (local >= 0) & (local < window_kv.shape[1])
                topk = torch.where(valid, local, -1).int().unsqueeze(0).expand(bsz, -1, -1).contiguous()
            _write_window_tail(self.window, kv, pos0)
        attn_kv = window_kv
        if self.compress_ratio:
            latent = self._compress(x, pos0) if self.is_kv_source else None
            if self.is_kv_source:
                state.compress_kv = self
                if self.is_index_source and self.indexer_wk is not None:
                    state.index_k = self
                if latent is not None:
                    self._append_compressed(latent, pos0, bsz)
            compress_len = (pos0 + seqlen) // max(self.compress_ratio, 1)
            src = state.compress_kv
            if src is not None and compress_len:
                if self.is_index_source:
                    idxs = self._index(x, qr, state, compress_len, window_kv.shape[1], pos0)
                    state.topk_idxs = idxs
                else:
                    idxs = state.topk_idxs
                if idxs is not None:
                    if canonical and seqlen>512 and verification_max_rows()==1024 and getattr(self,"_large_kv_prepare_owner",None) is None:
                        raise RuntimeError("Cap1024 compressed attention requires its full-source KV owner")
                    prepared = None
                    if canonical and seqlen <= 8 and getattr(self, "fused_kv_prep", False):
                        # The optional pack kernel supports at most 8 queries;
                        # longer canonical chunks use the identical torch pack.
                        from ds41.kv_prepare_cuda import prepare_sparse_kv
                        implicit_ring = seqlen == 1 and pos0 > 0
                        prepared = prepare_sparse_kv(
                            window_kv, None if implicit_ring else topk,
                            src.ck_codes[:bsz], src.ck_scales[:bsz], idxs, compress_len,
                            args.window_size, args.index_topk,
                            position=pos0 if implicit_ring else None,
                        )
                    elif canonical and 9 <= seqlen <= (1024 if verification_max_rows()==1024 else 512) and getattr(self, "_large_kv_prepare_owner", None) is not None:
                        # Explicit opt-in owner: unsupported eligible layouts
                        # raise before launch instead of silently using eager.
                        prepared = self._large_kv_prepare_owner.prepare(
                            window_kv, topk, src.ck_codes[:bsz], src.ck_scales[:bsz],
                            idxs, compress_len, args.window_size, args.index_topk)
                    if prepared is not None:
                        attn_kv, topk = prepared
                    else:
                        if canonical and seqlen>512 and verification_max_rows()==1024:
                            raise RuntimeError("Cap1024 compressed attention cannot use eager duplicate-row fallback")
                        # Dequantize selected rows without expanding full context.
                        offset = window_kv.shape[1]
                        selected = idxs - offset
                        valid = (selected >= 0) & (selected < compress_len)
                        selected = selected.clamp(0, compress_len - 1).reshape(bsz, -1).long()
                        codes = src.ck_codes[:bsz].gather(1, selected.unsqueeze(-1).expand(-1, -1, args.head_dim // 2))
                        scale_bytes = src.ck_scales[:bsz].view(torch.uint8)
                        scales = scale_bytes.gather(1, selected.unsqueeze(-1).expand(-1, -1, args.head_dim // 16)).view(torch.float8_e4m3fn)
                        ckv = dequant_fp4_group(codes, scales, 16, dtype=kv.dtype)
                        attn_kv = torch.cat([window_kv, ckv], dim=1)
                        remap = torch.arange(idxs.numel() // bsz, device=x.device).reshape(1, seqlen, -1) + offset
                        remap = torch.where(valid, remap, -1).int()
                        topk = torch.cat([topk, remap], dim=-1)
        if canonical:
            # The same physical reduction width and the same per-query CUDA
            # kernel are essential: masked future slots must not change the
            # softmax summation tree between decode and verification.
            width = args.window_size + (args.index_topk if self.compress_ratio else 0)
            topk = F.pad(topk, (0, width - topk.shape[-1]), value=-1)
            if batch_runtime is not None and batch_runtime.use_batch("sparse", x, canonical):
                o = batch_runtime.sparse(q, attn_kv, self.attn_sink, topk, args.head_dim**-0.5)
            else:
                outputs = []
                for i in range(seqlen):
                    qi, ii = q[:, i:i + 1], topk[:, i:i + 1]
                    fast = sparse_decode_(qi, attn_kv, self.attn_sink, ii, args.head_dim**-0.5) if x.is_cuda else None
                    outputs.append(fast if fast is not None else sparse_attn(qi, attn_kv, self.attn_sink, ii, args.head_dim**-0.5))
                o = outputs[0] if seqlen == 1 else torch.cat(outputs, dim=1)
        else:
            if batch_runtime is not None:
                batch_runtime.use_batch("sparse", x, False)
            o = sparse_attn(q, attn_kv, self.attn_sink, topk, args.head_dim**-0.5)
        if batch_rope:
            batch_runtime.rope_(o, freqs, inverse=True, operation="output_inverse")
        elif x.is_cuda and canonical and args.rope_head_dim == 64:
            for i in range(seqlen):
                if not rope_tail_(o[:, i:i + 1], freqs[i:i + 1], True):
                    apply_rotary_emb(o[:, i:i + 1, ..., -args.rope_head_dim :], freqs[i:i + 1], inverse=True)
        else:
            apply_rotary_emb(o[..., -args.rope_head_dim :], freqs, inverse=True)
        o = o.reshape(bsz, seqlen, args.n_heads, args.head_dim)
        return self.project_out(o)

    def _append_compressed(self, latent, pos0, bsz):
        """Append one completed group; index K is projected before main-KV RoPE."""
        args = self.args
        ratio = self.compress_ratio
        first_pos = pos0 - pos0 % ratio
        positions = torch.arange(latent.shape[1], device=latent.device) * ratio + first_pos
        freqs = self.freqs[positions]
        begin = pos0 // ratio
        end = begin + latent.shape[1]
        if end > self.ck_codes.shape[1]:
            raise ValueError("compressed KV context capacity exceeded")
        if self.is_index_source and self.indexer_wk is not None:
            ik = rms(self.indexer_wk.forward(latent), self.indexer_kn, args.norm_eps)
            apply_rotary_emb(ik[..., -args.rope_head_dim :], freqs)
            ic, isc, _ = quant_fp4_group(ik, 32, scale_fmt="ue8m0")
            self.ik_codes[:bsz, begin:end] = ic
            self.ik_scales[:bsz, begin:end] = isc
        # Avoid mutating a latent which may alias an output graph's storage.
        lat = latent.clone()
        apply_rotary_emb(lat[..., -args.rope_head_dim :], freqs)
        codes, scales, _ = quant_fp4_group(lat, 16)
        self.ck_codes[:bsz, begin:end] = codes
        self.ck_scales[:bsz, begin:end] = scales

    def _index(self, x, qr, state: State, compress_len, offset, pos0=0):
        args = self.args
        bsz, seqlen, _ = x.shape
        q = self.indexer_wq.forward(qr).unflatten(-1, (args.index_n_heads, args.index_head_dim))
        apply_rotary_emb(q[..., -args.rope_head_dim :], self.freqs[pos0 : pos0 + seqlen])
        _, _, q = quant_fp4_group(q, 32, scale_fmt="ue8m0")
        src = state.index_k or self
        ik = dequant_fp4_group(src.ik_codes[:bsz, :compress_len], src.ik_scales[:bsz, :compress_len], 32, dtype=q.dtype)
        ik = ik[:, :, : args.index_head_dim]
        runtime = getattr(self, "_stable_linear_runtime", None)
        weights = (stable_linear(x.float(), self.indexer_wn) if runtime is None else
                   _stable_linear_dispatch(runtime, "indexer", self.layer_id, x.float(), self.indexer_wn,
                                           fallback=stable_linear)) * (args.index_head_dim**-0.5 * args.index_n_heads**-0.5)
        if _STRICT_DENSE and 1 < seqlen <= verification_max_rows():
            scores = []
            for i in range(seqlen):
                visible = (pos0 + i + 1) // self.compress_ratio
                one = torch.einsum("bshd,btd->bsht", q[:, i:i + 1].float(), ik[:, :visible].float())
                one = (one.relu() * weights[:, i:i + 1].unsqueeze(-1).float()).sum(2)
                scores.append(F.pad(one, (0, compress_len - visible), value=-torch.inf))
            score = torch.cat(scores, dim=1)
        else:
            score = torch.einsum("bshd,btd->bsht", q.float(), ik.float())
            score = (score.relu() * weights.unsqueeze(-1).float()).sum(2)
        lens = compress_len
        if seqlen > 1 or pos0 == 0:
            lens = (torch.arange(pos0 + 1, pos0 + seqlen + 1, device=x.device) // self.compress_ratio).unsqueeze(-1)
            score = score.masked_fill(torch.arange(compress_len, device=x.device) >= lens, -torch.inf)
        if self.layer_id == args.candidate_source_layer:
            state.candidates = _candidate_mask(score, lens, args.candidate_topk_blocks, args.candidate_block_size)
        # Earlier indexers must not consume the previous forward's mask when
        # its compressed width happens to match this query's visible prefix.
        elif (0 <= args.candidate_source_layer < self.layer_id and state.candidates is not None
              and score.shape[-1] == state.candidates.shape[-1]):
            score = score.masked_fill(~state.candidates, -torch.inf)
        topk = min(args.index_topk, compress_len)
        if topk <= 0:
            return torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        idxs = _stable_topk_indices(score, topk).sort(-1).values
        idxs = torch.where(idxs < lens, idxs + offset, -1).int()
        if _STRICT_DENSE and seqlen <= verification_max_rows():
            idxs = F.pad(idxs, (0, args.index_topk - idxs.shape[-1]), value=-1)
        return idxs


def _stable_topk_indices(scores, k):
    """Rank equal FP32 scores by their earlier position, independent of width.

    Quantized ReLU index scores often tie at zero. CUDA/CPU topk do not define
    tie order and can otherwise change a causal query when future masked rows
    or verification chunk boundaries change the physical tensor width.
    """
    values = scores.float().contiguous()
    bits = values.view(torch.int32)
    ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    ordered = torch.where(values == 0, 0, ordered).to(torch.int64)
    position = torch.arange(values.shape[-1], device=values.device, dtype=torch.int64)
    keys = ordered * 4294967296 + (4294967295 - position)
    return keys.topk(k, dim=-1, sorted=False).indices


def _candidate_mask(logits, compress_len, topk_blocks, block_size):
    width = logits.size(-1)
    scores = F.pad(logits, (0, (-width) % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(-1)
    num = scores.size(-1)
    last = (compress_len - 1) // block_size
    scores = scores.masked_fill(torch.arange(num, device=logits.device) == last, torch.inf)
    idxs = _stable_topk_indices(scores, min(topk_blocks, num))
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, idxs, scores.gather(-1, idxs) > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def rms(x, weight, eps):
    var = mean_square(x)
    return (weight * x.float() * torch.rsqrt(var + eps)).to(x.dtype)


def exl3(store: WeightStore, prefix: str, device: str):
    tre = store.get(prefix + ".trellis")
    suh = store.get(prefix + ".suh")
    svh = store.get(prefix + ".svh")
    mul = store.get(prefix + ".mul1") if store.has(prefix + ".mul1") else torch.zeros((), dtype=torch.int32)
    k = tre.shape[-1] / 16
    if abs(k - round(k)) > 1e-6:
        tre = tre.view(suh.shape[0] // 16, svh.shape[0] // 16, tre.shape[-1])
    else:
        tre = tre.view(suh.shape[0] // 16, svh.shape[0] // 16, int(round(k)) * 16)
    return build_linear(suh, svh, tre, mul, device)


class Transformer(torch.nn.Module):
    def __init__(self, args: ModelArgs, store: WeightStore, cache: ExpertCache, device="cuda"):
        super().__init__()
        _bind_hc_residual(self)
        self.extended_verification = extended_verification_contract()
        if self.extended_verification["enabled"]:
            if (args.max_batch_size, args.n_heads, args.head_dim, args.rope_head_dim, args.window_size) != (1, 64, 512, 64, 128):
                raise ValueError("Experimental cap256 requires B1/64heads/512dim/64RoPE/SWA128")
            fast = getattr(cache, "_fast_pointer_table", None)
            from ds41.pointer_table import FastPointerTable
            if not isinstance(fast, FastPointerTable) or fast.host is not cache.host_ptrs:
                raise ValueError("Experimental cap256 requires this cache's actual pinned pointer retirement table")
        self.args = args
        self.cache = cache
        self.device = device
        capture_stream_flag = os.environ.get("DS41_CAPTURE_STREAM_SHARED", "0")
        if capture_stream_flag not in ("0", "1"):
            raise ValueError("DS41_CAPTURE_STREAM_SHARED must be 0 or 1")
        self.capture_stream_shared_requested = capture_stream_flag == "1"
        if self.capture_stream_shared_requested and os.environ.get("DS41_NO_GRAPH") == "1":
            raise ValueError("DS41_CAPTURE_STREAM_SHARED requires enabled graph capture")
        self._capture_stream_context = None
        self.capture_stream_shared_workspace = None
        self.capture_stream_shared_captured = {}
        self.hc_verify_graph_requested = _parse_hc_verify_graph_rows(os.environ.get("DS41_HC_VERIFY_GRAPHS", ""))
        shared_pool_flag = os.environ.get("DS41_HC_VERIFY_SHARED_POOL", "0")
        if shared_pool_flag not in ("0", "1"):
            raise ValueError("DS41_HC_VERIFY_SHARED_POOL must be 0 or 1")
        self.hc_verify_graph_shared_pool_requested = shared_pool_flag == "1"
        if self.hc_verify_graph_shared_pool_requested and (
                not self.hc_verify_graph_requested or os.environ.get("DS41_NO_GRAPH") == "1"):
            raise ValueError("HC verification shared pool requires enabled DS41_HC_VERIFY_GRAPHS")
        from ds41.hc_verify_runtime import validate_configuration
        self.hc_verify_io_arena_requested = validate_configuration(
            os.environ, self.hc_verify_graph_requested, self.hc_verify_graph_shared_pool_requested,
            _STRICT_DENSE, args, device)
        self._hc_verify_io_runtime = None
        self._hc_verify_shared_pool = None
        self.hc_verify_graph_shared_pool_memory = None
        self.hc_verify_graph_captured = {}
        self.hc_verify_graph_memory = dict(allocated_bytes=0, reserved_bytes=0, device_free_delta_bytes=0)
        cpu_embedding_flag = os.environ.get("DS41_CPU_EMBEDDING", "0")
        if cpu_embedding_flag not in ("0", "1"):
            raise ValueError("DS41_CPU_EMBEDDING must be 0 or 1")
        self.cpu_embedding_enabled = cpu_embedding_flag == "1"
        self.engram = EngramRuntime.from_store(args, store)
        embedding = store.get("embed.weight")
        self.embed = embedding if self.cpu_embedding_enabled else embedding.to(device)
        self.cpu_embedding_bytes = self.embed.numel()*self.embed.element_size() if self.cpu_embedding_enabled else 0
        self.norm = store.get("norm.weight").to(device)
        self.head = exl3(store, "head", device)
        print("loading dense layers", flush=True)
        names = dense_names(args, store)
        store.preload(names)
        self.layers = torch.nn.ModuleList([Block(i, args, store, device) for i in range(args.n_layers)])
        self.state = State()
        self.position = 0
        self.main_hidden = None
        kmap = cache.kmap
        head_k = self.head.k
        expect = kmap.get("head_bits")
        if expect is not None and int(head_k) != int(expect):
            raise RuntimeError(f"head K {head_k} != kmap head_bits {expect}")
        for layer in self.layers:
            layer.routed_k = float(kmap.get("routed", {}).get(str(layer.layer_id), 3))
        wk = self.layers[2].attn.indexer_wk
        if wk is not None and kmap.get("indexer_wk_bits") is not None:
            if int(wk.k) != int(kmap["indexer_wk_bits"]):
                raise RuntimeError(f"indexer wk K {wk.k} != {kmap['indexer_wk_bits']}")
        swa = precompute_freqs_cis(
            args.rope_head_dim, args.max_seq_len, 0, args.rope_theta, args.rope_factor,
            args.beta_fast, args.beta_slow, device,
        )
        compressed = precompute_freqs_cis(
            args.rope_head_dim, args.max_seq_len, args.original_seq_len, args.compress_rope_theta,
            args.rope_factor, args.beta_fast, args.beta_slow, device,
        )
        for layer in self.layers:
            layer.attn.bind_freqs(swa, compressed)
        if os.environ.get("DS41_NO_GRAPH") != "1":
            print("capturing decode graphs", flush=True)
            self._capture_m1_graphs()
            self._capture_hc_verification_graphs()
            torch.cuda.synchronize()
            t_g = time.perf_counter()
            for _ in range(4):
                for layer in self.layers:
                    if layer.attn.qkv_graph is not None:
                        layer.attn.qkv_graph.replay()
                    if layer.attn.wo_graph is not None:
                        layer.attn.wo_graph.replay()
            torch.cuda.synchronize()
            graph_ms = (time.perf_counter() - t_g) * 1e3 / 4
            print(f"projection graphs {graph_ms:.1f} ms/token", flush=True)
        # Shared workspaces are allocated after capture on the replay stream,
        # before generate measures free VRAM for expert residency.
        from ds41.canonical_attention import initialize
        self.canonical_attention_cap = verification_max_rows()
        initialize(self, strict=_STRICT_DENSE)
        # These48 weights are already resident. Capture one shared adapter only
        # after HC/projection capture, before generate computes expert budgets.
        from ds41.stable_linear_runtime import initialize as initialize_stable_linear
        stable_runtime = initialize_stable_linear(self, strict=_STRICT_DENSE)
        if stable_runtime is not None:
            for layer in self.layers:
                layer._stable_linear_runtime = stable_runtime
                layer.attn._stable_linear_runtime = stable_runtime
        from ds41.mean_square_runtime import initialize as initialize_mean_square
        initialize_mean_square(self, strict=_STRICT_DENSE)
        from ds41.large_kv_prepare_runtime import initialize as initialize_large_kv_prepare
        initialize_large_kv_prepare(self, strict=_STRICT_DENSE)

    def _capture_m1_graphs(self):
        if getattr(self, "_hc_verify_io_runtime", None) is not None:
            raise RuntimeError("HC arena forbids graph recapture on a live model")
        context = (_SharedCaptureStream(self.device)
                   if getattr(self, "capture_stream_shared_requested", False) else None)
        self._capture_stream_context = context
        self.capture_stream_shared_workspace = None if context is None else dict(context.workspace_config)
        kwargs = {} if context is None else dict(capture_stream=context.stream, replay_guard=context)
        for layer in self.layers:
            try:
                layer.attn.capture_decode_graphs(**kwargs)
            except Exception as exc:
                print(f"graph skip layer {layer.layer_id}: {type(exc).__name__}: {exc}", flush=True)
                layer.attn.qkv_graph = None
                layer.attn.wo_graph = None
            try:
                layer.capture_shared(**kwargs)
            except Exception as exc:
                print(f"shared graph skip layer {layer.layer_id}: {type(exc).__name__}: {exc}", flush=True)
                layer._sh_graph = None
            try:
                # ELinear outputs FP16, so only the first attention sees
                # embedding dtype; each FFN and subsequent layer sees FP16.
                if getattr(self, "_hc_residual_bf16", False):
                    layer.capture_hc(dtype=torch.bfloat16, ffn_dtype=torch.bfloat16, **kwargs)
                else:
                    layer.capture_hc(dtype=self.embed.dtype if layer.layer_id == 0 else torch.float16,
                                     ffn_dtype=torch.float16, **kwargs)
            except Exception as exc:
                print(f"hc graph skip layer {layer.layer_id}: {type(exc).__name__}: {exc}", flush=True)
                layer._hc_pack = None
            self.capture_stream_shared_captured[str(layer.layer_id)] = dict(
                qkv=int(layer.attn.qkv_graph is not None), wo=int(layer.attn.wo_graph is not None),
                shared=int(layer._sh_graph is not None), hc=len(layer._hc_pack or ()))

    def _capture_hc_verification_graphs(self):
        if getattr(self, "_hc_verify_io_runtime", None) is not None:
            raise RuntimeError("HC arena recapture requires discarding the model and graphs")
        if not self.hc_verify_graph_requested:
            return
        device = self.device
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        free_before, _ = torch.cuda.mem_get_info(device)
        shared_pool = (_HCVerifySharedPool(device)
                       if getattr(self, "hc_verify_graph_shared_pool_requested", False) else None)
        self._hc_verify_shared_pool = shared_pool
        io_runtime = None
        if getattr(self, "hc_verify_io_arena_requested", False):
            from ds41.hc_verify_storage import HCVerifyIOArena
            from ds41.hc_verify_runtime import HCVerifyArenaRuntime
            arena = HCVerifyIOArena(self.hc_verify_graph_requested, device=device,
                                    hidden_dtypes=(torch.float16, torch.bfloat16))
            io_runtime = HCVerifyArenaRuntime(arena, layers=self.args.n_layers)
            self._hc_verify_io_runtime = io_runtime
            arena.check_capture_owner(shared_pool.capture_stream)
        for layer in self.layers:
            kwargs = dict(dtype=self.embed.dtype if layer.layer_id == 0 else torch.float16,
                          ffn_dtype=torch.float16)
            if getattr(self, "_hc_residual_bf16", False):
                kwargs.update(dtype=torch.bfloat16, ffn_dtype=torch.bfloat16)
            if shared_pool is not None:
                kwargs["shared_pool"] = shared_pool
            if io_runtime is not None:
                kwargs["io_runtime"] = io_runtime
            layer.capture_hc_verification(self.hc_verify_graph_requested, **kwargs)
            self.hc_verify_graph_captured[layer.layer_id] = tuple(layer._hc_verify_pack)
        torch.cuda.synchronize()
        self.hc_verify_graph_shared_pool_memory = None if shared_pool is None else dict(shared_pool.memory)
        free_after, _ = torch.cuda.mem_get_info(device)
        self.hc_verify_graph_memory = dict(
            allocated_bytes=max(0, torch.cuda.memory_allocated(device) - allocated_before),
            reserved_bytes=max(0, torch.cuda.memory_reserved(device) - reserved_before),
            device_free_delta_bytes=max(0, free_before - free_after),
        )
        print("hc verification graphs " + json.dumps(dict(
            requested=self.hc_verify_graph_requested, captured=self.hc_verify_graph_captured,
            memory=self.hc_verify_graph_memory, shared_pool=self.hc_verify_graph_shared_pool_memory)), flush=True)

    def hc_verify_graph_hit_counts(self):
        """Return CPU replay-enqueue counters; no CUDA synchronization/readback."""
        return {str(layer.layer_id): {f"{rows}:{kind}": int(count)
                                    for (rows, kind), count in getattr(layer, "_hc_verify_hits", {}).items()}
                for layer in self.layers}

    def embed_ids(self, input_ids):
        return embedding_lookup(self.embed,input_ids)

    def forward(self, input_ids, pos0=0, ced=False, prefetcher=None, full_logits=False):
        if getattr(self, "_hc_residual_bf16", False):
            _verify_hc_residual(self)
        with _mean_square_scope(self):
            runtime = getattr(self, "_hc_verify_io_runtime", None)
            if runtime is None:
                return self._forward_complete(input_ids, pos0, ced, prefetcher, full_logits)
            if (getattr(self.forward, "__func__", None) is not Transformer.forward
                    or getattr(self.run_logged, "__func__", None) is not Transformer.run_logged
                    or getattr(self._forward_complete, "__func__", None) is not Transformer._forward_complete):
                raise RuntimeError("HC arena supports the default complete Transformer.forward only")
            with runtime.forward_scope(input_ids):
                return self._forward_complete(input_ids, pos0, ced, prefetcher, full_logits)

    def close(self):
        first_error = None
        for name in ("_hc_verify_io_runtime", "_stable_linear_runtime", "_mean_square_runtime", "_large_kv_prepare_runtime"):
            runtime = getattr(self, name, None)
            if runtime is not None:
                try:
                    # HC retains graph/storage owners; stable-linear drains
                    # before releasing its own private graph/IO references.
                    runtime.close()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    else:
                        first_error.add_note(name + " cleanup also failed: " + repr(exc))
        if first_error is not None:
            raise first_error

    def _forward_complete(self, input_ids, pos0=0, ced=False, prefetcher=None, full_logits=False):
        capture_context = getattr(self, "_capture_stream_context", None)
        if capture_context is not None:
            capture_context.check_replay_owner()
        if ced:
            raise NotImplementedError("CED bounded replay requires full encoder global-KV projection; decoder-only replay is not exact")
        if input_ids.shape[1] + pos0 > self.args.max_seq_len:
            raise ValueError("input exceeds reserved context capacity")
        self._clear_verification()
        capture = full_logits and input_ids.shape[1] <= verification_max_rows()
        if capture:
            self._verification = dict(pos0=pos0, length=input_ids.shape[1], batch=input_ids.shape[0])
            for layer in self.layers:
                layer.attn._capture_verification = True
                layer.attn._verification_raw = None
                layer.attn._verification_window = None
        if pos0 == 0:
            self.state = State()
        # Engram hashes use the actual token stream, never token ids as row ids.
        # Legacy prefetcher arguments cannot override the model's table format.
        try:
            if self.engram is not None:
                self.engram.kick(input_ids, pos0)
            logits, _, _ = self.run_logged(input_ids, pos0, range(self.args.n_layers), self.engram, full_logits=full_logits)
        except BaseException:
            self._clear_verification()
            raise
        finally:
            if capture:
                for layer in self.layers:
                    layer.attn._capture_verification = False
        self.position = pos0 + input_ids.shape[1]
        return logits

    def forward_logits_all(self, input_ids, pos0=0):
        """One causal target verification pass, returning [batch, tokens, vocab]."""
        return self.forward(input_ids, pos0, full_logits=True)

    def snapshot_state(self, pos0=None, token_count=None):
        """Save only ring slots and append ranges a short verification may alter.

        No full-context KV copy is made. A snapshot can be restored repeatedly,
        but must cover every subsequently evaluated token before restoration.
        """
        max_tokens = verification_max_rows() if token_count is None else token_count
        pos0 = self.position if pos0 is None else pos0
        if pos0 < 0 or max_tokens < 1 or pos0 + max_tokens > self.args.max_seq_len:
            raise ValueError("invalid state snapshot range")
        blocks = []
        for layer in self.layers:
            attn = layer.attn
            slots = torch.arange(pos0, pos0 + max_tokens, device=attn.window.device) % self.args.window_size
            slots = slots.unique()
            saved = dict(attn=attn, slots=slots, window=attn.window.index_select(1, slots).clone())
            if attn.is_kv_source:
                begin, end = pos0 // attn.compress_ratio, (pos0 + max_tokens) // attn.compress_ratio
                saved.update(begin=begin, end=end, ck_codes=attn.ck_codes[:, begin:end].clone(),
                             ck_scales=attn.ck_scales[:, begin:end].clone(),
                             kv_state=attn.kv_state.clone(), score_state=attn.score_state.clone())
                if attn.is_index_source and attn.indexer_wk is not None:
                    saved.update(ik_codes=attn.ik_codes[:, begin:end].clone(), ik_scales=attn.ik_scales[:, begin:end].clone())
            blocks.append(saved)
        shared = dict(self.state.__dict__)
        result = dict(owner=id(self), pos0=pos0, length=max_tokens, position=self.position,
                      layers=blocks, shared=shared, main_hidden=getattr(self, "main_hidden", None))
        if self.engram is not None:
            result["engram"] = self.engram.hash.cache[:, pos0:pos0 + max_tokens].clone()
        return result

    def restore_state(self, snapshot):
        if snapshot["owner"] != id(self):
            raise ValueError("snapshot belongs to a different model")
        if self.position > snapshot["pos0"] + snapshot["length"]:
            raise ValueError("verification exceeded its snapshot range")
        for saved in snapshot["layers"]:
            attn = saved["attn"]
            attn.window.index_copy_(1, saved["slots"], saved["window"])
            if "ck_codes" in saved:
                begin, end = saved["begin"], saved["end"]
                attn.ck_codes[:, begin:end].copy_(saved["ck_codes"])
                attn.ck_scales[:, begin:end].copy_(saved["ck_scales"])
                attn.kv_state.copy_(saved["kv_state"])
                attn.score_state.copy_(saved["score_state"])
                if "ik_codes" in saved:
                    attn.ik_codes[:, begin:end].copy_(saved["ik_codes"])
                    attn.ik_scales[:, begin:end].copy_(saved["ik_scales"])
        self.state = State()
        self.state.__dict__.update(snapshot["shared"])
        self.position = snapshot["position"]
        self.main_hidden = snapshot["main_hidden"]
        if "engram" in snapshot:
            begin = snapshot["pos0"]
            self.engram.hash.cache[:, begin:begin + snapshot["length"]].copy_(snapshot["engram"])
        self._clear_verification()

    def _clear_verification(self):
        if getattr(self, "_verification", None) is not None:
            for layer in self.layers:
                layer.attn._capture_verification = False
                layer.attn._verification_raw = None
                layer.attn._verification_window = None
        self._verification = None

    @torch.inference_mode()
    def commit_prefix(self, snapshot, committed_inputs):
        """Commit a verified causal input prefix without replaying the backbone.

        Requires a forward_logits_all within verification_max_rows() from this snapshot's
        position. Completed accepted groups are retained. Only rejected ring
        slots, compressed append tails and hash history are restored, while
        partial compression groups are rebuilt from saved raw projection rows.
        """
        verify = getattr(self, "_verification", None)
        if snapshot["owner"] != id(self):
            raise ValueError("snapshot belongs to a different model")
        if verify is None or verify["pos0"] != snapshot["pos0"] or snapshot["position"] != snapshot["pos0"]:
            raise ValueError("prefix commit requires a short verification from the snapshot position")
        pos, count, batch = verify["pos0"], verify["length"], verify["batch"]
        k = int(committed_inputs)
        if k != committed_inputs or not 0 <= k <= count or count > snapshot["length"] or self.position != pos + count:
            raise ValueError("invalid committed input count or stale verification")
        # Validate retained raw state before making any mutation.
        for saved in snapshot["layers"]:
            attn = saved["attn"]
            if "kv_state" in saved and attn.wgate is not None:
                raw = getattr(attn, "_verification_raw", None)
                if raw is None or raw[0].shape[:2] != (batch, count) or raw[1].shape != raw[0].shape:
                    raise ValueError("missing raw compressor rows for prefix commit")
            if count > self.args.window_size and getattr(attn, "_verification_window", None) is None:
                raise ValueError("missing verified KV rows for a multiply wrapped ring")
        if k == 0:
            self.restore_state(snapshot)
            return
        if k == count:
            self._clear_verification()
            return

        for saved in snapshot["layers"]:
            attn = saved["attn"]
            device, window = attn.window.device, self.args.window_size
            if count <= window:
                slots = torch.arange(pos + k, pos + count, device=device) % window
                original = saved["window"].index_select(1, torch.searchsorted(saved["slots"], slots))
                attn.window.index_copy_(1, slots, original)
            else:
                attn.window.index_copy_(1, saved["slots"], saved["window"])
                start = max(0, k - window)
                slots = torch.arange(pos + start, pos + k, device=device) % window
                attn.window[:batch].index_copy_(1, slots, attn._verification_window[:, start:k])
            if "ck_codes" not in saved:
                continue
            ratio = attn.compress_ratio
            keep, end = (pos + k) // ratio, saved["end"]
            offset = keep - saved["begin"]
            attn.ck_codes[:, keep:end].copy_(saved["ck_codes"][:, offset:])
            attn.ck_scales[:, keep:end].copy_(saved["ck_scales"][:, offset:])
            if "ik_codes" in saved:
                attn.ik_codes[:, keep:end].copy_(saved["ik_codes"][:, offset:])
                attn.ik_scales[:, keep:end].copy_(saved["ik_scales"][:, offset:])
            attn.kv_state.copy_(saved["kv_state"])
            attn.score_state.copy_(saved["score_state"])
            if attn.wgate is None:
                continue
            raw_kv, raw_score = attn._verification_raw
            if k == 1 and pos > 0:
                # Match streaming's untouched inactive batch/unused slots.
                if (pos + 1) % ratio:
                    attn.kv_state[:batch, pos % ratio] = raw_kv[:, 0]
                    attn.score_state[:batch, pos % ratio] = raw_score[:, 0]
                else:
                    attn.kv_state[:batch].zero_()
                    attn.score_state[:batch].fill_(-torch.inf)
            else:
                attn.kv_state.zero_()
                attn.score_state.fill_(-torch.inf)
                remainder = (pos + k) % ratio
                old_needed, raw_needed = max(0, remainder - k), min(remainder, k)
                if old_needed:
                    old = pos % ratio
                    attn.kv_state[:batch, :old_needed] = saved["kv_state"][:batch, old - old_needed:old]
                    attn.score_state[:batch, :old_needed] = saved["score_state"][:batch, old - old_needed:old]
                if raw_needed:
                    attn.kv_state[:batch, old_needed:remainder] = raw_kv[:, k - raw_needed:k]
                    attn.score_state[:batch, old_needed:remainder] = raw_score[:, k - raw_needed:k]

        def window_width(tokens):
            return self.args.window_size if pos > 0 and tokens == 1 else tokens + min(pos, self.args.window_size - 1)

        # These per-forward routing caches are recomputed by the next pass,
        # but retain exactly the shape/offset a prefix replay would produce.
        sources = [layer.attn for layer in self.layers if layer.attn.is_index_source and layer.attn.compress_ratio]
        if sources:
            topk = self.state.topk_idxs
            if (pos + k) // sources[-1].compress_ratio == 0:
                self.state.topk_idxs = snapshot["shared"]["topk_idxs"]
            elif topk is not None:
                topk = topk[:, :k]
                delta = window_width(k) - window_width(count)
                self.state.topk_idxs = torch.where(topk >= 0, topk + delta, topk) if delta else topk
        candidates = self.state.candidates
        candidate_source = next((layer.attn for layer in self.layers
                                 if layer.layer_id == self.args.candidate_source_layer), None)
        if candidate_source is not None and candidate_source.compress_ratio:
            visible = (pos + k) // candidate_source.compress_ratio
            self.state.candidates = (candidates[:, :k, :visible] if visible and candidates is not None
                                     else snapshot["shared"]["candidates"])
        if "engram" in snapshot:
            self.engram.hash.cache[:, pos + k:pos + snapshot["length"]].copy_(snapshot["engram"][:, k:])
        self.main_hidden = self.main_hidden[:, :k] if self.main_hidden is not None else None
        self.position = pos + k
        self._clear_verification()

    def _forward_ced(self, input_ids, prefetcher):
        raise NotImplementedError("CED bounded replay has not passed reference parity")

    def _stack(self, input_ids, pos0, layers, prefetcher, hidden):
        if getattr(self, "_hc_verify_io_runtime", None) is not None:
            raise RuntimeError("HC arena rejects partial/resumed stacks; use default complete forward")
        if hidden is None:
            h = self.embed_ids(input_ids)
            if getattr(self, "_hc_residual_bf16", False):
                h = h.to(torch.bfloat16)
            h = h.unsqueeze(2).expand(-1, -1, self.args.hc_mult, -1).contiguous()
            pre = torch.zeros(h.shape[0], h.shape[1], self.args.hc_mult, device=h.device, dtype=torch.float32)
            pre[:, :, 0] = 1.0
        else:
            h, pre = hidden
        engram_slots = {lid: i for i, lid in enumerate(self.args.engram_layer_ids)}
        for layer_id in layers:
            rows = None
            if prefetcher is not None and layer_id in engram_slots:
                rows = prefetcher.wait(engram_slots[layer_id])
            h, pre = self.layers[layer_id](h, pos0, pre, self.cache, self.state, None, rows)
        return h, pre

    def run_logged(self, input_ids, pos0, layers, prefetcher, full_logits=False):
        runtime = getattr(self, "_hc_verify_io_runtime", None)
        if runtime is not None:
            runtime.begin_stack(layers)
            layers = range(self.args.n_layers)
        h = self.embed_ids(input_ids)
        if getattr(self, "_hc_residual_bf16", False):
            h = h.to(torch.bfloat16)
        h = h.unsqueeze(2).expand(-1, -1, self.args.hc_mult, -1).contiguous()
        pre = torch.zeros(h.shape[0], h.shape[1], self.args.hc_mult, device=h.device, dtype=torch.float32)
        pre[:, :, 0] = 1.0
        engram_slots = {lid: i for i, lid in enumerate(self.args.engram_layer_ids)}
        main_hiddens = []
        for layer_id in layers:
            rows = None
            if prefetcher is not None and layer_id in engram_slots:
                rows = prefetcher.wait(engram_slots[layer_id])
            if layer_id in self.args.dspark_target_layer_ids:
                # DSpark consumes the pre-attention residual stream at its
                # target layers, as in the official forward_spec reference.
                main_hiddens.append(h.mean(dim=2))
            h, pre = self.layers[layer_id](h, pos0, pre, self.cache, self.state, None, rows)
        self.main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        collapsed = torch.sum(pre.unsqueeze(-1) * h.float(), dim=2)
        collapsed = rms(collapsed, self.norm, self.args.norm_eps)
        logits = self.head.forward((collapsed if full_logits else collapsed[:, -1]).to(torch.float16))
        if runtime is not None:
            # The default forward discards borrowed pre only after head work is
            # enqueued. No extra clone or standalone run_logged API is enabled.
            runtime.finish_stack()
        return logits, h, pre


def dense_names(args: ModelArgs, store: WeightStore):
    names = ["embed.weight", "norm.weight", "head.trellis", "head.suh", "head.svh", "head.mul1"]
    for i in range(args.n_layers):
        p = f"layers.{i}."
        for prefix in (
            "attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b",
            "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3",
        ):
            for suffix in ("trellis", "suh", "svh", "mul1"):
                names.append(p + prefix + "." + suffix)
        for s in range(args.o_groups):
            for suffix in ("trellis", "suh", "svh", "mul1"):
                names.append(p + f"attn.wo_a.slice.{s}.{suffix}")
        names += [
            p + "attn_norm.weight", p + "ffn_norm.weight",
            p + "attn.q_norm.weight", p + "attn.kv_norm.weight", p + "attn.attn_sink",
            p + "hc_attn_fn", p + "hc_ffn_fn", p + "hc_attn_base", p + "hc_ffn_base",
            p + "hc_attn_scale", p + "hc_ffn_scale",
            p + "ffn.gate.weight", p + "ffn.gate.bias",
        ]
        if store.has(p + "ffn.gate.bias_vl"):
            names.append(p + "ffn.gate.bias_vl")
        if i in args.kv_source_layers:
            for prefix in ("attn.compressor.wkv", "attn.compressor.wgate"):
                for suffix in ("trellis", "suh", "svh", "mul1"):
                    n = p + prefix + "." + suffix
                    if store.has(n):
                        names.append(n)
            names.append(p + "attn.compressor.norm.weight")
        if i in args.index_source_layers:
            for prefix in ("attn.indexer.wq_b", "attn.indexer.wk"):
                for suffix in ("trellis", "suh", "svh", "mul1"):
                    n = p + prefix + "." + suffix
                    if store.has(n):
                        names.append(n)
            names += [p + "attn.indexer.weights_proj.weight", p + "attn.indexer.k_norm.weight"]
        if i in args.engram_layer_ids:
            for suffix in ("trellis", "suh", "svh", "mul1"):
                names.append(p + "engram.wkv." + suffix)
            names += [p + "engram.q_weight", p + "engram.k_weight"]
    return [n for n in names if store.has(n)]
