"""The checkpoint's real three-stage DSpark draft head.

This module implements the reference semi-AR window attention, mHC blocks,
128-way/top-3 routed EXL3 experts, sequential Markov bias, and raw confidence.
Draft tokens are proposals only: acceptance must come from backbone verification.
"""
import json
import os
from collections import OrderedDict
from dataclasses import replace

import torch
import torch.nn.functional as F

from ds41.kernels import apply_rotary_emb, fake_quant_fp8, precompute_freqs_cis, sparse_attn, swiglu
from ds41.model import Attention, Block, exl3, rms
from ds41.moe_coop import CoopRunner
from ds41.store import WeightStore
from ds41.cache import pinned_allocation_charge
from ds41.exl3_linear import ELinear
from ds41.embedding import embedding_lookup


def _native_draft_linears(root):
    """Use native batch dispatch in proposal-only wrappers, including lists.

    ELinear is not an nn.Module, so modules() misses attention projections,
    grouped wo_a and shared-expert lists. Never walk the underlying lin object
    or an external weight store/cache, whose ownership may include the target.
    """
    seen, pending, count = set(), [root], 0
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, ELinear):
            value.strict_small_rows = False
            count += 1
        elif isinstance(value, torch.nn.Module):
            pending.extend(vars(value).values())
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    return count


class MTPManifestStore(WeightStore):
    """WeightStore's tensor loader, bound to a metadata-only independent MTP map."""
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("format") != "ds41-dspark-tensors-v1":
            raise ValueError("not a DSpark tensor manifest")
        self.entries = manifest["tensors"]
        if any(not name.startswith("mtp.") for name in self.entries):
            raise ValueError("DSpark manifest contains a non-MTP tensor")
        self.dir = os.path.dirname(os.path.abspath(path))
        self.wm = {name: entry["path"] for name, entry in self.entries.items()}
        self.cache, self.headers = {}, {}

    def meta(self, name):
        entry = self.entries[name]
        path = entry["path"]
        if not os.path.isabs(path):
            path = os.path.join(self.dir, path)
        return path, entry["offset"], entry["nbytes"], entry["dtype"], entry["shape"]


class _MTPNamespace:
    """Reuse backbone linear/attention helpers while keeping all names in mtp.*."""
    def __init__(self, store):
        self.store = store

    @staticmethod
    def _name(name):
        if not name.startswith("layers."):
            raise ValueError("MTP adapter expects a layer tensor name")
        return "mtp." + name[len("layers."):]

    def get(self, name):
        return self.store.get(self._name(name))

    def has(self, name):
        return self.store.has(self._name(name))


class DSparkExperts:
    """Independent 3x128 expert cache: bounded GPU slots backed by pinned RAM.

    RAM mode lazily retains up to all 6.344 GiB of original EXL3 expert bytes;
    GPU mode can preload them all for a 6.344 GiB residency experiment. This
    budget must be subtracted from backbone hot-expert space by the caller.
    """
    def __init__(self, args, store, device="cuda", mode="ram", gpu_slots=24,
                 ram_budget_bytes=None):
        if mode not in ("ram", "resident"):
            raise ValueError("DSpark expert mode must be ram or resident")
        self.args, self.store, self.device, self.mode = args, store, device, mode
        self.metadata, self.fds = {}, {}
        self.ram, self.gpu = OrderedDict(), OrderedDict()
        self.ram_nbytes = 0
        self.ram_charge_bytes = 0
        self.runners = {}
        self.stats = dict(ram=0, gpu=0, disk=0, disk_bytes=0, h2d_bytes=0)
        total = 0
        for stage in range(args.n_mtp_layers):
            for expert in range(args.dspark_n_routed_experts):
                parts = []
                cursor = 0
                for proj in (1, 2, 3):
                    p = f"mtp.{stage}.ffn.experts.{expert}.w{proj}."
                    specs = [store.meta(p + kind) for kind in ("suh", "svh", "trellis")]
                    marker = store.meta(p + "mul1")
                    if marker[2:4] != (4, "I32") or marker[4] != []:
                        raise ValueError("MTP experts require the EXL3 mul1 codebook marker")
                    k = specs[2][4][-1] / 16
                    if k != 4:
                        raise ValueError(f"expected checkpoint MTP K=4, got {k}")
                    parts.append((cursor, specs))
                    cursor += sum(spec[2] for spec in specs)
                self.metadata[(stage, expert)] = (cursor, parts)
                total += cursor
        self.record_bytes = max(entry[0] for entry in self.metadata.values())
        self.ram_budget_bytes = (0 if mode == "resident" else total) if ram_budget_bytes is None else int(ram_budget_bytes)
        self.gpu_slots = len(self.metadata) if mode == "resident" else max(int(gpu_slots), args.dspark_block_size * args.dspark_n_activated_experts)
        self.host_ptrs = torch.zeros(args.n_mtp_layers, 9, args.dspark_n_routed_experts, dtype=torch.long, pin_memory=True, device="cpu")
        self.packed = torch.zeros_like(self.host_ptrs, device=device)

    def close(self):
        if self.device != "cpu":
            torch.cuda.synchronize()
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()
        self.ram.clear()
        self.gpu.clear()
        self.runners.clear()
        self.ram_nbytes = 0
        self.ram_charge_bytes = 0

    def _load(self, key):
        cached = self.ram.get(key)
        if cached is not None:
            self.ram.move_to_end(key)
            self.stats["ram"] += 1
            return cached
        nbytes, parts = self.metadata[key]
        host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True, device="cpu")
        target = memoryview(host.numpy())
        cursor = 0
        for _, specs in parts:
            for path, offset, size, dtype, shape in specs:
                path = os.path.join(self.store.dir, path)
                if path not in self.fds:
                    self.fds[path] = os.open(path, os.O_RDONLY)
                done = 0
                while done < size:
                    got = os.preadv(self.fds[path], [target[cursor + done:cursor + size]], offset + done)
                    if not got:
                        raise IOError(f"short DSpark expert read {key}")
                    done += got
                cursor += size
        self.stats["disk"] += 1
        self.stats["disk_bytes"] += nbytes
        charge = pinned_allocation_charge(nbytes)
        if charge <= self.ram_budget_bytes:
            while self.ram and self.ram_charge_bytes + charge > self.ram_budget_bytes:
                _, victim = self.ram.popitem(last=False)
                self.ram_nbytes -= victim.numel()
                self.ram_charge_bytes -= pinned_allocation_charge(victim.numel())
            self.ram[key] = host
            self.ram_nbytes += nbytes
            self.ram_charge_bytes += charge
        return host

    def _bind(self, key, storage):
        stage, expert = key
        _, parts = self.metadata[key]
        pointers = []
        for offset, specs in parts:
            p = []
            for spec in specs:
                p.append(storage.data_ptr() + offset)
                offset += spec[2]
            pointers.append(p)  # suh, svh, trellis
        gate, down, up = pointers
        self.host_ptrs[stage, :, expert] = torch.tensor(
            [gate[2], gate[0], gate[1], up[2], up[0], up[1], down[2], down[0], down[1]], dtype=torch.long, device="cpu")

    def _install(self, key, active):
        if key in self.gpu:
            self.gpu.move_to_end(key)
            self.stats["gpu"] += 1
            return
        if len(self.gpu) >= self.gpu_slots:
            victim = next((k for k in self.gpu if k not in active), None)
            if victim is None:
                raise RuntimeError("DSpark GPU cache cannot hold the active expert union")
            storage, event = self.gpu.pop(victim)
            event.synchronize()
            self.host_ptrs[victim[0], :, victim[1]] = 0
        else:
            storage = torch.empty(self.record_bytes, dtype=torch.uint8, device=self.device)
        host = self._load(key)
        storage[:host.numel()].copy_(host, non_blocking=True)
        self.stats["h2d_bytes"] += host.numel()
        event = torch.cuda.Event()
        event.record()
        self.gpu[key] = (storage, event)
        self._bind(key, storage)

    def preload(self):
        """Optional long cold start. Resident mode allocates all expert GPU bytes."""
        for key in self.metadata:
            if self.mode == "resident":
                self._install(key, set())
            else:
                self._load(key)
        torch.cuda.synchronize()

    def moe(self, stage, x, indices, weights):
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        topk = self.args.dspark_n_activated_experts
        ids = list(dict.fromkeys(indices.detach().reshape(-1).tolist()))
        active = {(stage, int(expert)) for expert in ids}
        if len(active) > self.gpu_slots:
            raise ValueError("DSpark active expert union exceeds its independent cache")
        for key in active:
            self._install(key, active)
        self.packed[stage].copy_(self.host_ptrs[stage], non_blocking=True)
        runner_key = (rows, topk)
        if runner_key not in self.runners:
            self.runners[runner_key] = CoopRunner(self.device, topk, self.args.dim, self.args.moe_inter_dim, rows=rows)
        output = self.runners[runner_key].run(x, indices, weights, self.packed[stage], 4., self.args.swiglu_limit)
        for key in active:
            self.gpu[key][1].record()
        return output


class DSparkAttention(Attention):
    def _main_kv(self, main_x, start_pos):
        kv = rms(self.wkv.forward(main_x), self.kv_norm, self.args.norm_eps)
        freqs = self.freqs[start_pos:start_pos + main_x.shape[1]]
        apply_rotary_emb(kv[..., -self.args.rope_head_dim:], freqs)
        return fake_quant_fp8(kv)

    def ingest(self, main_x, start_pos):
        kv = self._main_kv(main_x, start_pos)
        win = self.args.window_size
        if start_pos == 0:
            self.window.zero_()
        # A ring assignment handles prefill and accepted multi-token prefixes.
        # If the chunk exceeds the window, only its last window can remain live.
        first = max(0, kv.shape[1] - win)
        positions = torch.arange(start_pos + first, start_pos + kv.shape[1], device=kv.device) % win
        self.window[:kv.shape[0], positions] = kv[:, first:].to(self.window.dtype)

    def forward(self, x, start_pos, main_x):
        if main_x.shape[1] != 1:
            raise ValueError("draft attention expects exactly one anchor hidden state")
        self.ingest(main_x, start_pos)
        bsz, block_size, _ = x.shape
        args = self.args
        freqs = self.freqs[start_pos + 1:start_pos + 1 + block_size]
        qr = rms(self.wq_a.forward(x), self.q_norm, args.norm_eps)
        q = self.wq_b.forward(qr).unflatten(-1, (args.n_heads, args.head_dim))
        kv = rms(self.wkv.forward(x), self.kv_norm, args.norm_eps)
        apply_rotary_emb(q[..., -args.rope_head_dim:], freqs)
        apply_rotary_emb(kv[..., -args.rope_head_dim:], freqs)
        fake_quant_fp8(kv)
        past = torch.arange(min(args.window_size, start_pos + 1), device=x.device)
        draft = args.window_size + torch.arange(block_size, device=x.device)
        # All draft positions see the entire block, as in get_dspark_topk_idxs.
        idx = torch.cat((past, draft)).int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()
        keys = torch.cat((self.window[:bsz], kv), dim=1)
        out = sparse_attn(q, keys, self.attn_sink, idx, args.head_dim ** -0.5)
        apply_rotary_emb(out[..., -args.rope_head_dim:], freqs, True)
        grouped = out.reshape(bsz, block_size, args.o_groups, -1)
        fake_quant_fp8(grouped)
        parts = [self.wo_a[group].forward(grouped[:, :, group].contiguous()) for group in range(args.o_groups)]
        return self.wo_b.forward(torch.stack(parts, dim=2).flatten(2))


class DSparkBlock(Block):
    def __init__(self, stage, args, store, experts, device):
        torch.nn.Module.__init__(self)
        self.layer_id, self.args = stage, args
        self.dim, self.hc_mult, self.eps = args.dim, args.hc_mult, args.norm_eps
        p = f"layers.{stage}."
        for name in ("attn_norm", "ffn_norm"):
            setattr(self, name, store.get(p + name + ".weight").to(device))
        for name in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale"):
            setattr(self, name, store.get(p + name).to(device).float())
        self.gate_w = store.get(p + "ffn.gate.weight").to(device).float()
        self.gate_b = store.get(p + "ffn.gate.bias").to(device).float()
        self.gate_b_vl = None
        self.attn = DSparkAttention(stage, args, store, device)
        self.shared = [exl3(store, p + f"ffn.shared_experts.w{i}", device) for i in (1, 2, 3)]
        self._sh_graph = None
        self.expert_cache = experts

    def forward(self, x, start_pos, pre_mix, main_x):
        residual = x
        attn_pre, post, comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, 0)
        hidden = rms(self.hc_pre(x, pre_mix), self.attn_norm, self.eps)
        hidden = self.attn(hidden, start_pos, main_x).to(residual.dtype)
        x = self.hc_post(hidden, residual, post, comb)
        residual = x
        ffn_pre, post, comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, 1)
        hidden = rms(self.hc_pre(x, attn_pre), self.ffn_norm, self.eps)
        flat = hidden.reshape(-1, self.dim).to(torch.float16)
        weights, indices = self.route(flat, None)
        routed = self.expert_cache.moe(self.layer_id, flat, indices, weights)
        hidden = (routed + self._shared(flat)).to(x.dtype).reshape_as(hidden)
        return self.hc_post(hidden, residual, post, comb), ffn_pre


class DSparkModel(torch.nn.Module):
    def __init__(self, args, store, embed, head, device="cuda", expert_mode="ram", gpu_slots=24,
                 ram_budget_bytes=None, expert_cache=None):
        super().__init__()
        if args.n_mtp_layers != 3 or len(args.dspark_target_layer_ids) != 3:
            raise ValueError("this DSpark checkpoint requires three stages and three target inputs")
        self.args = args
        self.embed = embed
        # Share immutable quantized head tensors, not the target wrapper's
        # canonical small-row dispatch policy.
        self.head = ELinear(head.lin) if isinstance(head, ELinear) else head
        self.temperature = args.temperature
        draft_args = replace(args, n_routed_experts=args.dspark_n_routed_experts,
                             n_activated_experts=args.dspark_n_activated_experts,
                             compress_ratios=(0,) * args.n_mtp_layers,
                             kv_source_layers=(), index_source_layers=())
        namespace = _MTPNamespace(store)
        self.expert_cache = expert_cache or DSparkExperts(args, store, device, expert_mode, gpu_slots, ram_budget_bytes)
        self.blocks = torch.nn.ModuleList(DSparkBlock(stage, draft_args, namespace, self.expert_cache, device)
                                          for stage in range(args.n_mtp_layers))
        self.main_proj = exl3(store, "mtp.0.main_proj", device)
        self.main_norm = store.get("mtp.0.main_norm.weight").to(device)
        self.norm = store.get("mtp.2.norm.weight").to(device)
        self.markov_embed = store.get("mtp.2.markov_head.embed.weight").to(device)
        self.markov_weight = store.get("mtp.2.markov_head.head.weight").to(device).float()
        self.confidence_weight = store.get("mtp.2.confidence_head.proj.weight").to(device).float()
        # Reference DSpark attention is pure SWA and explicitly disables YaRN.
        freqs = precompute_freqs_cis(args.rope_head_dim, args.max_seq_len + args.dspark_block_size + 1,
                                    0, args.rope_theta, args.rope_factor,
                                    args.beta_fast, args.beta_slow, device)
        for block in self.blocks:
            block.attn.freqs = freqs
        self.native_batch_linears = _native_draft_linears(self)
        self._seeded = False
        self._history_end = 0
        self._tail_seed_pending = False
        self._tail_seed_failed = False
        self._tail_seed_metadata = None

    @property
    def tail_seed_metadata(self):
        """Identity of the explicit tail seed, not a legacy warm parity claim."""
        value = getattr(self, "_tail_seed_metadata", None)
        return None if value is None else dict(value)

    def _check_tail_seed_health(self):
        if getattr(self, "_tail_seed_failed", False):
            raise RuntimeError("DSpark tail seed failed; discard this draft model")
        if getattr(self, "_tail_seed_pending", False):
            raise RuntimeError("DSpark tail seed is still pending")

    def _synchronize_tail_seed(self, device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @torch.inference_mode()
    def seed_tail(self, main_hidden_tail, *, prompt_end):
        """Reset all three rings from accepted target rows [N-min(N,W), N).

        N is the absolute exclusive prompt endpoint. Native EXL3 warm geometry
        changes from M=N to M=min(N,W): this is a separate proposal variant,
        not a promise of bitwise equality with full warm. A failure after reset
        poisons this instance, including its normal warm/ingest entry points.
        """
        self._check_tail_seed_health()
        args = self.args
        if type(prompt_end) is not int or not 1 <= prompt_end <= args.max_seq_len:
            raise ValueError("invalid absolute DSpark prompt endpoint")
        window = args.window_size
        if type(window) is not int or window < 1:
            raise ValueError("invalid DSpark window size")
        rows = min(prompt_end, window)
        width = args.dim * len(args.dspark_target_layer_ids)
        tail = main_hidden_tail
        if (not isinstance(tail, torch.Tensor) or tail.ndim != 3
                or not 1 <= tail.shape[0] <= args.max_batch_size
                or tail.shape[1:] != (rows, width)
                or tail.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            raise ValueError("DSpark tail must contain every retained absolute prompt row")
        if len(self.blocks) != 3 or len(args.dspark_target_layer_ids) != 3:
            raise ValueError("DSpark tail seed requires exactly three stages and target inputs")
        if self.main_norm.device != tail.device:
            raise ValueError("DSpark tail device differs from main projection norm")
        # Validate every allocation before changing flags or any ring bytes.
        for block in self.blocks:
            attn = block.attn
            ring, freqs = attn.window, attn.freqs
            if (attn.args.window_size != window or not isinstance(ring, torch.Tensor)
                    or ring.shape != (args.max_batch_size, window, args.head_dim)
                    or ring.device != tail.device or not isinstance(freqs, torch.Tensor)
                    or freqs.device != tail.device or len(freqs) < prompt_end):
                raise ValueError("DSpark tail stage window/device/frequency range is invalid")
        start = prompt_end - rows
        self._tail_seed_pending = True
        self._seeded, self._history_end = False, 0
        self._tail_seed_metadata = None
        try:
            # Clear the entire max_batch allocation, including inactive slots.
            for block in self.blocks:
                block.attn.window.zero_()
            main_x = self._project_main(tail)
            if (main_x.shape != (tail.shape[0], rows, args.dim)
                    or main_x.device != tail.device):
                raise RuntimeError("DSpark tail main projection changed shape/device")
            for block in self.blocks:
                block.attn.ingest(main_x, start)
            # Retain tail/main_x through completion before publishing success.
            self._synchronize_tail_seed(tail.device)
            metadata = {
                "seed_mode": "absolute_tail_v1", "proposal_variant": "absolute_tail_v1",
                "hidden_start_pos": start, "hidden_end_pos": prompt_end,
                "retained_rows": rows, "window_size": window, "hidden_width": width,
                "hidden_dtype": str(tail.dtype),
                "retained_hidden_bytes": tail.numel() * tail.element_size(),
                "stages": 3, "seed_reset": True,
                "legacy_full_warm_bitwise_validated": False,
            }
            self._history_end = prompt_end
            self._tail_seed_metadata = metadata
            self._seeded = True
        except BaseException:
            self._tail_seed_failed = True
            self._seeded, self._history_end = False, 0
            self._tail_seed_metadata = None
            raise
        finally:
            # The public health guard forbids observing a partially published seed.
            self._tail_seed_pending = False
        return self.tail_seed_metadata

    def _project_main(self, main_hidden):
        width = self.args.dim * len(self.args.dspark_target_layer_ids)
        if main_hidden.ndim != 3 or main_hidden.shape[-1] != width:
            raise ValueError(f"DSpark requires main_hidden[B,T,{width}] from target attention inputs")
        return rms(self.main_proj.forward(main_hidden), self.main_norm, self.args.norm_eps)

    @torch.inference_mode()
    def ingest(self, main_hidden, start_pos=0):
        self._check_tail_seed_health()
        if main_hidden.shape[1] < 1 or start_pos < 0:
            raise ValueError("DSpark ingest requires a nonempty accepted prefix")
        if start_pos and (not self._seeded or start_pos > self._history_end):
            raise ValueError("DSpark accepted history has a gap; warm the prompt first")
        main_x = self._project_main(main_hidden)
        for block in self.blocks:
            block.attn.ingest(main_x, start_pos)
        self._seeded = True
        self._history_end = start_pos + main_hidden.shape[1]
        if start_pos == 0:
            self._tail_seed_metadata = None

    warm = ingest

    def _forward_head(self, x, pre_mix, input_ids):
        hidden = self.blocks[-1].hc_pre(x, pre_mix)
        logits = self.head.forward(rms(hidden, self.norm, self.args.norm_eps)).float()
        batch, block_size = logits.shape[:2]
        output = input_ids.new_empty((batch, block_size + 1))
        output[:, 0] = input_ids.reshape(batch)
        markov_embeds = []
        for position in range(block_size):
            markov = F.embedding(output[:, position], self.markov_embed)
            logits[:, position].add_(F.linear(markov.float(), self.markov_weight))
            markov_embeds.append(markov)
            if self.temperature == 0:
                output[:, position + 1] = logits[:, position].argmax(-1)
            else:
                probabilities = torch.softmax(logits[:, position] / max(self.temperature, 1e-5), -1)
                output[:, position + 1] = probabilities.div_(torch.empty_like(probabilities).exponential_(1)).argmax(-1)
        markov = torch.stack(markov_embeds, dim=1)
        confidence = F.linear(torch.cat((hidden.float(), markov.float()), dim=-1), self.confidence_weight).squeeze(-1)
        return output, logits, confidence

    @torch.inference_mode()
    def draft(self, input_ids, main_hidden, start_pos):
        self._check_tail_seed_health()
        if not self._seeded or start_pos <= 0:
            raise RuntimeError("seed DSpark windows with warm(main_hidden, 0) before drafting")
        if main_hidden.shape[1] != 1:
            raise ValueError("draft takes exactly the last accepted main hidden state")
        if start_pos > self._history_end:
            raise ValueError("DSpark draft anchor skips uncommitted main history")
        if start_pos < self._history_end - 1:
            raise ValueError("DSpark cannot draft a historical anchor without replaying its window")
        main_x = self._project_main(main_hidden)
        input_ids = input_ids.reshape(-1)
        ids = input_ids.new_full((input_ids.shape[0], self.args.dspark_block_size), self.args.dspark_noise_token_id)
        ids[:, 0] = input_ids
        # Preserve the checkpoint's BF16 embedding/mHC residual stream; the
        # EXL3 linear adapter performs its required F16 conversion internally.
        x = embedding_lookup(self.embed, ids)
        x = x.unsqueeze(2).expand(-1, -1, self.args.hc_mult, -1).clone()
        pre_mix = x.new_zeros((*x.shape[:2], self.args.hc_mult), dtype=torch.float32)
        pre_mix[..., 0] = 1.
        for block in self.blocks:
            x, pre_mix = block(x, start_pos, pre_mix, main_x)
        self._history_end = max(self._history_end, start_pos + 1)
        return self._forward_head(x, pre_mix, input_ids)

    def close(self):
        self.expert_cache.close()
