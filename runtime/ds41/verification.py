"""Process-wide bound for canonical causal target verification.

Set DS41_VERIFY_MAX_ROWS before importing the inference modules. The bound is
shared by dense arithmetic, attention and transaction capture; changing only
one of those bounds can silently change a speculative target's numerics.

The construction ceiling of512 is an experimental candidate, not a numerical
certificate. Default8 and explicitly selected64 retain their old branches.
Selecting128 also changes a128-token prefill from native to canonical math;
compare targets with the same process cap and prefill schedule.
"""
import os

MAX_VERIFICATION_ROWS = 1024
LEGACY_MAX_VERIFICATION_ROWS = 128


def _parse_max_rows(value):
    if not isinstance(value, str):
        raise ValueError("DS41_VERIFY_MAX_ROWS must be an integer from 1 to 128 or explicit256/512/1024")
    try:
        rows = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("DS41_VERIFY_MAX_ROWS must be an integer from 1 to 128 or explicit256/512/1024") from exc
    if str(rows) != value.strip() or not (1 <= rows <= LEGACY_MAX_VERIFICATION_ROWS or rows in (256,512,1024)):
        raise ValueError("DS41_VERIFY_MAX_ROWS must be an integer from 1 to 128 or explicit256/512/1024")
    return rows


_MAX_ROWS = _parse_max_rows(os.environ.get("DS41_VERIFY_MAX_ROWS", "8"))


def verification_max_rows():
    """Return the startup-validated target input limit (pending + proposals)."""
    return _MAX_ROWS


def canonical_query_capacity():
    """Legacy wrappers retain128/256; never advertise an uncompiled512 tier."""
    return 256 if verification_max_rows() == 256 else 128


def canonical_runtime_query_capacity():
    """Only the dispatch owner selects the new independent fixed512 modules."""
    if verification_max_rows() == 1024:
        return 1024
    return 512 if verification_max_rows() == 512 else canonical_query_capacity()


def _with_coop2048(contract, environment, speculation, prefill_chunk_size):
    from ds41.native2048_runtime import validate_configuration
    candidate = validate_configuration(environment, cap=verification_max_rows(),
        speculation=speculation, prefill_chunk_size=prefill_chunk_size)
    if not candidate["requested"]:
        return _with_edit_candidate(contract, environment, speculation)
    result = dict(contract)
    result.update(enabled=True, variant=result.get("variant", "lookup128")+"_coop2048_v1",
                  native_rows_max=min(256, verification_max_rows()), native_coop2048=candidate,
                  prefill_chunk_size=128, speculation="lookup", GPU_numerically_verified=False)
    return _with_edit_candidate(result, environment, speculation)


def _with_edit_candidate(contract, environment, speculation):
    value = environment.get("DS41_EDIT_CANDIDATE", "0")
    if value == "0":
        return contract
    if value != "1":
        raise ValueError("DS41_EDIT_CANDIDATE must be 0 or 1")
    if verification_max_rows() not in (128,256,512,1024) or (speculation is not None and speculation != "lookup"):
        raise ValueError("Prompt edit candidates require cap128/256/512/1024 and lookup target verification")
    result = dict(contract)
    # `enabled` belongs to the extended numeric/owner contract. A prompt-only
    # proposal must not enable cap256's pointer/geometry guard under cap128.
    result.update(variant=result.get("variant", "lookup128")+"_prompt_edit_candidate_v1",
        edit_candidate=dict(requested=True, protocol="prompt_only_literal_default_candidate_v1",
            target_verification_required=True, GPU_numerically_verified=False),
        GPU_numerically_verified=False)
    return result


def _contract512(environment, speculation, prefill_chunk_size):
    """Distinct experimental owner/capacity; no old primitive proof transfer."""
    from ds41.lookup_policy import configuration as lookup_configuration
    from ds41.native3072_runtime import validate_configuration
    lookup_policy = lookup_configuration(environment)
    from ds41.hc_verify_runtime import requested as arena_requested
    arena_enabled = arena_requested(environment)
    required = {"DS41_FAST_POINTER_BIND":"1", "DS41_ROUTE_SNAPSHOT":"0",
                "DS41_ROUTE_AWARE_CHUNKS":"1", "DS41_HC_VERIFY_IO_ARENA":"1" if arena_enabled else "0",
                "DS41_STABLE_LINEAR_GRAPHS":"0", "DS41_LOOKUP_POLICY":lookup_policy["mode"],
                "DS41_NEXT_CHUNK_PREFETCH":"0", "DS41_STRICT_BATCH":"1", "DS41_STRICT_DENSE":"1",
                "DS41_DENSE_CANONICAL_TILE":"32", "DS41_FUSED_HC":"0", "DS41_MOE_COOP512":"1",
                "DS41_MOE_COOP2048":"0", "DS41_MOE_COOP3072":"1", "DS41_DEMAND_H2D_PIPELINE":"1",
                "DS41_CANONICAL_SPARSE_BATCH":"1", "DS41_CANONICAL_ROPE_BATCH":"1",
                "DS41_HC_VERIFY_GRAPHS":"512", "DS41_HC_VERIFY_SHARED_POOL":"1", "DS41_NO_GRAPH":"0",
                "EXL3_MOE_COOP_WIDE":"0", "EXL3_MOE_COOP_KSPLIT":"1"}
    defaults = {"DS41_STRICT_BATCH":"1", "DS41_STRICT_DENSE":"1", "DS41_LOOKUP_POLICY":"off",
                "EXL3_MOE_COOP_KSPLIT":"1"}
    for name, value in required.items():
        if environment.get(name, defaults.get(name,"0")) != value:
            raise ValueError("Experimental cap512 requires " + name + "=" + value)
    if speculation is not None and speculation != "lookup":
        raise ValueError("Experimental cap512 supports lookup-only target verification")
    if prefill_chunk_size is not None and (type(prefill_chunk_size) is not int or prefill_chunk_size != 128):
        raise ValueError("Experimental cap512 retains prefill_chunk_size128")
    native = validate_configuration(environment, cap=512, speculation=speculation,
                                    prefill_chunk_size=prefill_chunk_size)
    if not native.get("requested"):
        raise ValueError("Experimental cap512 requires its actual native3072 owner")
    result = dict(enabled=True, variant=("lookup512_hc512_only_coop3072_v1" if lookup_policy["mode"] == "off"
                                        else "lookup512_history_policy_hc512_only_coop3072_v1"),
        verification_max_rows=512, canonical_attention_query_capacity=512,
        canonical_attention_variant="independent_fixed512_original_M1_arithmetic_v1",
        gathered_KV_capacity=262783, native_rows_max=512, native_coop3072=native,
        required_volatile_experts=384, maximum_routed_slots=3072, dense_tile=32,
        prefill_chunk_size=128, speculation="lookup", lookup_policy=lookup_policy,
        hc_verify_graph_rows=[512], hc_capture_variant="existing_HC0_exact512_shared_scratch_v1",
        required_environment=required, inherited_256_GPU_certificate=False,
        GPU_numerically_verified=False)
    if arena_enabled:
        result.update(variant=result["variant"]+"_hc_io_arena_large_v1",
                      hc_io_arena_variant="complete_forward_hc_io_arena_large_v1",
                      hc_io_arena_GPU_validated=False)
    return _with_edit_candidate(result, environment, speculation)


def _contract1024(environment, speculation, prefill_chunk_size):
    """Separate opt-in tier;512-row kernels share one complete-layer lease."""
    from ds41.lookup_policy import configuration as lookup_configuration
    from ds41.native3072_runtime import validate_configuration
    lookup = lookup_configuration(environment)
    from ds41.hc_verify_runtime import requested as arena_requested
    arena_enabled = arena_requested(environment)
    required = {"DS41_FAST_POINTER_BIND":"1", "DS41_ROUTE_SNAPSHOT":"0",
        "DS41_ROUTE_AWARE_CHUNKS":"1", "DS41_HC_VERIFY_IO_ARENA":"1" if arena_enabled else "0",
        "DS41_STABLE_LINEAR_GRAPHS":"0", "DS41_LOOKUP_POLICY":lookup["mode"],
        "DS41_NEXT_CHUNK_PREFETCH":"0", "DS41_STRICT_BATCH":"1", "DS41_STRICT_DENSE":"1",
        "DS41_DENSE_CANONICAL_TILE":"32", "DS41_FUSED_HC":"0", "DS41_MOE_COOP512":"1",
        "DS41_MOE_COOP2048":"0", "DS41_MOE_COOP3072":"1", "DS41_DEMAND_H2D_PIPELINE":"1",
        "DS41_CANONICAL_SPARSE_BATCH":"1", "DS41_CANONICAL_ROPE_BATCH":"1",
        "DS41_CANONICAL_MEAN_SQUARE":"1", "DS41_LARGE_KV_PREPARE":"1", "DS41_FULL_COMPRESSED_KV":"1",
        "DS41_HC_VERIFY_GRAPHS":"1024", "DS41_HC_VERIFY_SHARED_POOL":"1", "DS41_NO_GRAPH":"0",
        "DS41_LAYER_PROMOTION":"0", "EXL3_MOE_COOP_WIDE":"0", "EXL3_MOE_COOP_KSPLIT":"1"}
    defaults = {"DS41_STRICT_BATCH":"1", "DS41_STRICT_DENSE":"1", "DS41_LOOKUP_POLICY":"off", "EXL3_MOE_COOP_KSPLIT":"1"}
    for name,value in required.items():
        if environment.get(name,defaults.get(name,"0")) != value:
            raise ValueError("Experimental cap1024 requires "+name+"="+value)
    if speculation is not None and speculation != "lookup":
        raise ValueError("Experimental cap1024 supports lookup-only target verification")
    if prefill_chunk_size is not None and (type(prefill_chunk_size) is not int or prefill_chunk_size != 128):
        raise ValueError("Experimental cap1024 retains prefill_chunk_size128")
    native = validate_configuration(environment,cap=1024,speculation=speculation,prefill_chunk_size=prefill_chunk_size)
    result = dict(enabled=True,variant="lookup1024_hc1024_joint3072_v1",verification_max_rows=1024,
        canonical_attention_query_capacity=1024,canonical_attention_variant="independent_fixed1024_original_M1_arithmetic_v1",
        gathered_KV_capacity=525439,full_compressed_source_required_above512=True,
        native_rows_max=512,joint_layer_rows_max=1024,native_coop3072=native,
        required_volatile_experts=384,maximum_routed_slots=3072,maximum_layer_routed_slots=6144,
        dense_tile=32,prefill_chunk_size=128,speculation="lookup",lookup_policy=lookup,
        hc_verify_graph_rows=[1024],hc_capture_variant="existing_HC0_exact1024_shared_scratch_v1",
        required_environment=required,inherited_512_GPU_certificate=False,GPU_numerically_verified=False)
    if arena_enabled:
        result.update(variant=result["variant"]+"_hc_io_arena_large_v1",
                      hc_io_arena_variant="complete_forward_hc_io_arena_large_v1",
                      hc_io_arena_GPU_validated=False)
    return _with_edit_candidate(result,environment,speculation)


def extended_verification_contract(environment=None, *, speculation=None, prefill_chunk_size=None):
    """Fail closed on unimplemented extended-cap combinations; metadata only.

    The model calls the environment-only portion before loading weights. A CLI
    owner must also pass its speculation/prefill values before construction.
    CPU/state fixtures may bypass Transformer.__init__, never this GPU claim.
    """
    environment = os.environ if environment is None else environment
    cap = verification_max_rows()
    if cap == 1024:
        return _contract1024(environment,speculation,prefill_chunk_size)
    if cap == 512:
        return _contract512(environment, speculation, prefill_chunk_size)
    if cap != 256:
        return _with_coop2048(dict(enabled=False, verification_max_rows=cap),
                             environment, speculation, prefill_chunk_size)
    # Import lazily: lookup_policy itself reads the already-fixed process cap.
    # Bounded lookup verification retains its target/commit math. Optional
    # prompt-derived proposals receive a separate startup identity below.
    from ds41.lookup_policy import configuration as lookup_configuration
    lookup_policy = lookup_configuration(environment)
    required = {"DS41_FAST_POINTER_BIND":"1", "DS41_ROUTE_SNAPSHOT":"0",
                "DS41_HC_VERIFY_IO_ARENA":"0", "DS41_STABLE_LINEAR_GRAPHS":"0",
                "DS41_LOOKUP_POLICY":lookup_policy["mode"], "DS41_NEXT_CHUNK_PREFETCH":"0",
                "DS41_STRICT_BATCH":"1", "DS41_STRICT_DENSE":"1",
                "DS41_DENSE_CANONICAL_TILE":"32", "DS41_FUSED_HC":"0",
                "DS41_MOE_COOP512":"1",
                "DS41_CANONICAL_SPARSE_BATCH":"1", "DS41_CANONICAL_ROPE_BATCH":"1"}
    defaults = {"DS41_STRICT_BATCH":"1", "DS41_STRICT_DENSE":"1", "DS41_LOOKUP_POLICY":"off"}
    for name,value in required.items():
        if environment.get(name,defaults.get(name,"0")) != value:
            raise ValueError("Experimental cap256 requires " + name + "=" + value)
    if speculation is not None and speculation != "lookup":
        raise ValueError("Experimental cap256 supports lookup-only target verification")
    if prefill_chunk_size is not None and (type(prefill_chunk_size) is not int or prefill_chunk_size != 128):
        raise ValueError("Experimental cap256 retains prefill_chunk_size128")
    graphs = environment.get("DS41_HC_VERIFY_GRAPHS","")
    graph_schedule = graphs.replace(" ","")
    if graph_schedule not in ("32,64,128", "256"):
        raise ValueError("Experimental cap256 requires HC graphs32,64,128 or exact256-only; arena remains unsupported")
    if graph_schedule == "256" and (environment.get("DS41_HC_VERIFY_SHARED_POOL", "0") != "1"
                                    or environment.get("DS41_NO_GRAPH", "0") != "0"):
        raise ValueError("HC256-only requires enabled verification graphs and their shared scratch pool")
    contract = dict(enabled=True, variant=("lookup256_minimal_v1" if lookup_policy["mode"] == "off"
                                           else "lookup256_history_policy_v1"), verification_max_rows=256,
                canonical_attention_query_capacity=256, gathered_KV_capacity=131455,
                native_rows_max=64, dense_tile=32, prefill_chunk_size=128, speculation="lookup",
                lookup_policy=lookup_policy, required_environment=required, GPU_numerically_verified=False)
    if graph_schedule == "256":
        contract.update(variant=("lookup256_hc256_only_v1" if lookup_policy["mode"] == "off"
                                 else "lookup256_history_policy_hc256_only_v1"),
                        hc_verify_graph_rows=[256], hc_capture_variant="existing_HC0_exact256_shared_scratch_v1")
        contract["required_environment"] = dict(required, DS41_HC_VERIFY_GRAPHS="256",
            DS41_HC_VERIFY_SHARED_POOL="1", DS41_NO_GRAPH="0")
    return _with_coop2048(contract, environment, speculation, prefill_chunk_size)
