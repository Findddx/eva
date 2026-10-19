"""Experimental, backbone-verified greedy decoding using real DSpark proposals.

Correct rollback/acceptance cannot repair batch-vs-sequential target numeric
drift. Callers must pass their target's parity gate before enabling this path.
"""
import math
import time

from ds41.speculate import lookup_draft, verify_greedy
from ds41.dspark_config import validate_decode_configuration, validate_override_anchor, validate_override_output
from ds41.verification import verification_max_rows


def _sync(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def _stats(owner):
    return dict(getattr(owner, "stats", {}))


def _delta(after, before):
    return {key: value - before.get(key, 0) for key, value in after.items()
            if isinstance(value, (int, float)) and isinstance(before.get(key, 0), (int, float))}


def _validate_tail_prefill(model, draft, hidden, prompt_tokens):
    """Require an explicit collector receipt; never infer a valid tail by length."""
    import torch

    window = getattr(getattr(draft, "args", None), "window_size", None)
    if type(window) is not int or window < 1:
        raise ValueError("DSpark tail seed requires a configured draft window")
    rows = min(prompt_tokens, window)
    metadata = getattr(model, "prefill_metadata", None)
    if (int(model.position) != prompt_tokens or not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != rows
            or not isinstance(metadata, dict)):
        raise ValueError("DSpark tail seed requires complete absolute prefill tail metadata")
    required = {"retention_mode": "tail", "retain_tail_hidden": window,
                "hidden_start_pos": prompt_tokens-rows, "hidden_end_pos": prompt_tokens,
                "retained_rows": rows, "prompt_tokens": prompt_tokens,
                "processed_tokens": prompt_tokens, "retain_full_hidden": False,
                "retained_hidden_bytes": hidden.numel()*hidden.element_size(),
                "hidden_width": hidden.shape[-1], "hidden_dtype": str(hidden.dtype)}
    if (any(type(metadata.get(key)) is not type(value) or metadata[key] != value
            for key, value in required.items())
            or type(metadata.get("chunk_size")) is not int or metadata["chunk_size"] <= 0):
        raise ValueError("DSpark tail seed prefill receipt does not match the accepted prompt")


def decode_dspark(model, draft, prompt_ids, first, max_new, device="cuda", depth=5,
                  raw_conf_min=None, eos=1, ignore_eos=False, verbose=True, use_lookup=False,
                  commit_strategy="replay", seed_mode="full", block_size_override=None):
    """Return generated IDs, amortized token records, and full cycle metrics.

    The model starts after prompt prefill, with `first` still pending as input.
    Only true accepted input hiddens are ingested into the DSpark window. Every
    final emitted token remains pending, matching ordinary autoregressive state.
    """
    import torch

    if getattr(model.cache, "cpu_backend", None) is not None:
        raise ValueError("CPU cold experts are disabled for speculative target verification")
    proposal_config = validate_decode_configuration(draft, depth, verification_max_rows(), block_size_override)
    if commit_strategy not in ("replay","truncate"):
        raise ValueError("Unknown prefix commit strategy")
    if seed_mode not in ("full", "absolute_tail_v1"):
        raise ValueError("Unknown DSpark seed mode")
    if max_new <= 0:
        return [], [], []
    generated, records, cycles = [int(first)], [], []
    if max_new == 1 or (not ignore_eos and generated[-1] == eos):
        return generated, records, cycles
    hidden = getattr(model, "main_hidden", None)
    if seed_mode == "absolute_tail_v1":
        _validate_tail_prefill(model, draft, hidden, len(prompt_ids))
    elif hidden is None or hidden.shape[1] != len(prompt_ids):
        raise ValueError("DSpark decoding requires full-prompt target-layer hidden states from prefill")
    _sync(device)
    warm_start = time.perf_counter()
    if seed_mode == "absolute_tail_v1":
        seed_metadata = draft.seed_tail(hidden, prompt_end=len(prompt_ids))
    else:
        draft.warm(hidden, start_pos=0)
        seed_metadata = None
    _sync(device)
    warm_ms = (time.perf_counter() - warm_start) * 1000
    while len(generated) < max_new and (ignore_eos or generated[-1] != eos):
        pos = int(model.position)
        remaining = max_new - len(generated)
        capacity = int(getattr(getattr(model, "args", None), "max_seq_len", pos + proposal_config.effective_block_size + 1)) - pos
        if capacity < 1:
            raise ValueError("DSpark verification exceeds the reserved target context")
        proposed_depth = min(depth, remaining - 1, capacity - 1)
        cache = model.cache
        draft_cache = getattr(draft, "expert_cache", None)
        before, draft_before = _stats(cache), _stats(draft_cache)
        _sync(device)
        cycle_start = time.perf_counter()
        proposals, raw_scores = [], []
        draft_ms = lookup_ms = mtp_draft_ms = 0.
        proposal_source = "none"
        if proposed_depth:
            draft_start = time.perf_counter()
            if use_lookup:
                lookup_start = time.perf_counter()
                proposals = [int(token) for token in lookup_draft(list(prompt_ids) + generated, proposed_depth)]
                lookup_ms = (time.perf_counter() - lookup_start) * 1000
                if proposals:
                    proposal_source = "lookup"
            if not proposals:
                proposal_source = "dspark"
                mtp_start = time.perf_counter()
                seed = torch.tensor([generated[-1]], device=device, dtype=torch.long)
                if block_size_override is not None:
                    validate_override_anchor(draft, model, pos, proposal_config.effective_block_size)
                ids, proposal_logits, confidence = draft.draft(seed, model.main_hidden[:, -1:], start_pos=pos - 1)
                _sync(device)
                if block_size_override is not None:
                    validate_override_output(ids, proposal_logits, confidence, seed, draft.args)
                ids = ids.detach().reshape(-1).tolist()
                full_scores = confidence.detach().float().reshape(-1).tolist()
                raw_scores = full_scores[:proposed_depth]
                if len(ids) < proposed_depth + 1 or int(ids[0]) != generated[-1]:
                    raise ValueError("DSpark returned an invalid seed/proposal block")
                if block_size_override is not None and (any(not math.isfinite(value) for value in full_scores)
                        or any(not 0 <= token < draft.args.vocab_size for token in ids)):
                    raise ValueError("DSpark override returned nonfinite confidence or invalid full-block token IDs")
                if block_size_override is not None:
                    # The independently owned Python lists survive draft cache,
                    # tensor and borrowed expert-output reuse in later cycles.
                    del proposal_logits, confidence
                proposals = [int(token) for token in ids[1:1 + proposed_depth]]
                if raw_conf_min is not None:
                    kept = 0
                    for raw in raw_scores:
                        if not math.isfinite(raw) or raw < raw_conf_min:
                            break
                        kept += 1
                    proposals = proposals[:kept]
                mtp_draft_ms = (time.perf_counter() - mtp_start) * 1000
            draft_ms = (time.perf_counter() - draft_start) * 1000
        inputs = [generated[-1]] + proposals
        batch = torch.tensor([inputs], device=device, dtype=torch.long)
        state_start = time.perf_counter()
        saved = model.snapshot_state(pos0=pos, token_count=len(inputs)) if proposals else None
        _sync(device)
        state_ms = (time.perf_counter() - state_start) * 1000
        verify_start = time.perf_counter()
        logits = model.forward_logits_all(batch, pos)
        predictions = logits.argmax(-1).reshape(-1).tolist()
        _sync(device)
        verify_ms = (time.perf_counter() - verify_start) * 1000
        emitted, matched = verify_greedy(proposals, predictions)
        if not ignore_eos and eos in emitted:
            emitted = emitted[:emitted.index(eos) + 1]
        commit_count = len(emitted)
        replay_ms = 0.
        commit_ms = 0.
        if commit_count < len(inputs):
            replay_start = time.perf_counter()
            if commit_strategy == "truncate":
                model.commit_prefix(saved,commit_count)
            else:
                model.restore_state(saved)
                model.forward(batch[:, :commit_count].contiguous(), pos)
            _sync(device)
            commit_ms = (time.perf_counter() - replay_start) * 1000
            if commit_strategy == "replay":
                replay_ms = commit_ms
        committed_hidden = model.main_hidden
        if committed_hidden is None or committed_hidden.shape[1] != commit_count:
            raise RuntimeError("target did not expose exactly the committed prefix's main hidden states")
        ingest_start = time.perf_counter()
        draft.ingest(committed_hidden, start_pos=pos)
        _sync(device)
        ingest_ms = (time.perf_counter() - ingest_start) * 1000
        if getattr(cache,"adaptive_gpu_enabled",False):
            _sync(device)
            cache.rebalance_gpu(len(generated)+len(emitted)-1)
        elapsed_ms = (time.perf_counter() - cycle_start) * 1000
        if not cycles:
            elapsed_ms += warm_ms
        generated.extend(emitted)
        reserved = getattr(cache, "_host_reserved_bytes", lambda: 0)()
        cycle = dict(position=pos, drafted=len(proposals), requested_depth=proposed_depth,
                     proposal_source=proposal_source, lookup_ngram_length=None,
                     accepted=min(matched, len(emitted)), matched=matched, emitted=len(emitted),
                     committed_inputs=commit_count, elapsed_ms=elapsed_ms,
                     warm_ms=warm_ms if not cycles else 0., draft_ms=draft_ms,
                     seed_mode=seed_mode, seed_metadata=seed_metadata,
                     proposal_config=proposal_config.as_dict(),
                     lookup_ms=lookup_ms, mtp_draft_ms=mtp_draft_ms, state_ms=state_ms,
                     verify_ms=verify_ms, replay_ms=replay_ms, ingest_ms=ingest_ms,
                     commit_strategy=commit_strategy,commit_ms=commit_ms,
                     rollback_s=commit_ms / 1000,
                     raw_confidence=raw_scores, cache=_delta(_stats(cache), before),
                     draft_cache=_delta(_stats(draft_cache), draft_before),
                     ram_payload_bytes=int(getattr(cache, "ram_nbytes", 0)),
                     ram_charge_bytes=int(getattr(cache, "ram_charge_bytes", 0)),
                     draft_ram_payload_bytes=int(getattr(draft_cache, "ram_nbytes", 0)),
                     draft_ram_charge_bytes=int(getattr(draft_cache, "ram_charge_bytes", 0)),
                     host_reserved_bytes=reserved)
        cycles.append(cycle)
        records.extend(dict(token=token, latency_ms=elapsed_ms / len(emitted),
                            latency_kind="amortized_cycle") for token in emitted)
        if verbose:
            print(f"{proposal_source} pos {pos}: draft {len(proposals)} accept {cycle['accepted']} "
                  f"emit {len(emitted)} {elapsed_ms:.2f} ms "
                  f"(draft {draft_ms:.2f}, verify {verify_ms:.2f}, replay {replay_ms:.2f})", flush=True)
    return generated, records, cycles
