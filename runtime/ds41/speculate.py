"""Exact greedy prompt-lookup speculation.

Lookup only proposes tokens. Every emitted token is checked against a causal
target-model logit. Rejected speculative KV is rolled back before continuing.
This module makes no claim about stochastic distribution-preserving sampling.
"""
import time


def lookup_draft(history, max_tokens=5, max_ngram=16, min_ngram=3):
    """Reuse the continuation of the latest earlier matching suffix."""
    if max_tokens < 1:
        return []
    for n in range(min(max_ngram, len(history)-1), min_ngram-1, -1):
        suffix = history[-n:]
        for end in range(len(history)-1, n-1, -1):
            if history[end-n:end] == suffix:
                return list(history[end:end+max_tokens])
    return []


def verify_greedy(draft, predictions):
    """Return accepted draft prefix plus one target token, and match count."""
    if len(predictions) != len(draft)+1:
        raise ValueError("Need one target prediction per pending/draft input")
    matched = 0
    while matched < len(draft) and draft[matched] == predictions[matched]:
        matched += 1
    return list(draft[:matched])+[int(predictions[matched])], matched


def decode_lookup(model, prompt_ids, first, max_new, device="cuda", depth=5,
                  eos=1, ignore_eos=False, commit_strategy="replay", edit_candidate_session=None):
    import torch
    from ds41.lookup_policy import LookupProposalPolicy

    if getattr(model.cache, "cpu_backend", None) is not None:
        raise ValueError("CPU expert activation quantization has not passed speculative target parity")
    if commit_strategy not in ("replay","truncate"):
        raise ValueError("Unknown prefix commit strategy")
    proposal_policy = LookupProposalPolicy.from_environment()
    generated = [int(first)]
    records = []
    cycles = []
    while len(generated) < max_new and (ignore_eos or generated[-1] != eos):
        history = list(prompt_ids)+generated
        pos = len(history)-1
        capacity = min(depth, max_new-len(generated)-1)
        edit_decision = None
        draft = None
        if edit_candidate_session is not None:
            draft, edit_decision = edit_candidate_session.select(generated, capacity)
        if draft is None:
            draft = (proposal_policy.select(history, capacity, prompt_tokens=len(prompt_ids))
                     if proposal_policy.enabled else lookup_draft(history, capacity))
            proposal_decision = proposal_policy.decision() if proposal_policy.enabled else None
        else:
            proposal_decision = None
        inputs = [generated[-1]]+draft
        t0 = time.perf_counter()
        before = dict(model.cache.stats)
        saved = model.snapshot_state(pos0=pos, token_count=len(inputs)) if draft else None
        batch = torch.tensor([inputs],device=device,dtype=torch.long)
        logits = model.forward_logits_all(batch,pos)
        predictions = logits.argmax(-1).reshape(-1).tolist()
        emitted, matched = verify_greedy(draft,predictions)
        if not ignore_eos and eos in emitted:
            emitted = emitted[:emitted.index(eos)+1]
        rollback_s = 0.
        commit_count = len(emitted)
        if commit_count < len(inputs):
            tr = time.perf_counter()
            if commit_strategy == "truncate":
                model.commit_prefix(saved,commit_count)
            else:
                model.restore_state(saved)
                model.forward(batch[:,:commit_count].contiguous(),pos)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            rollback_s = time.perf_counter()-tr
        if getattr(model.cache,"adaptive_gpu_enabled",False):
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            model.cache.rebalance_gpu(len(generated)+len(emitted)-1)
        elapsed = (time.perf_counter()-t0)*1000
        delta = {k:model.cache.stats[k]-before.get(k,0) for k in model.cache.stats}
        generated.extend(emitted)
        cycles.append({"position":pos,"drafted":len(draft),"accepted":matched,
                       "emitted":len(emitted),"elapsed_ms":elapsed,"rollback_s":rollback_s,
                       "commit_strategy":commit_strategy,
                       "cache":delta})
        if proposal_decision is not None:
            cycles[-1]["proposal_policy"] = proposal_decision
        if edit_decision is not None:
            cycles[-1]["edit_candidate"] = edit_decision
        # Per-token latency is amortized cycle service time, not inter-token
        # arrival latency. Preserve the actual cycle time separately above.
        for token in emitted:
            records.append({"token":token,"latency_ms":elapsed/len(emitted),
                            "latency_kind":"amortized_cycle"})
        print(f"lookup pos {pos}: draft {len(draft)} accept {matched} "
              f"emit {len(emitted)} {elapsed:.2f} ms",flush=True)
    return generated,records,cycles
