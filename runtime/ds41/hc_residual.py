"""Explicit BF16 HC residual numerical target; no CUDA work on import/OFF."""
import hashlib
import os
from pathlib import Path

FLAG = 'DS41_HC_RESIDUAL_BF16'
VARIANT = 'bf16_hc_residual_fp32_pre_rms_half_branch_v1'
TARGET = 'fixed_native_dense_m32_hc_bf16_residual_v1'


def requested(environment=None):
    env = os.environ if environment is None else environment
    value = env.get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError(FLAG+' must be exactly0 or1')
    return value == '1'


def configuration(environment=None):
    env = os.environ if environment is None else environment
    if not requested(env):
        return None
    if (env.get('DS41_STRICT_DENSE','1') != '1' or env.get('DS41_STRICT_BATCH','1') != '1'
            or env.get('DS41_FUSED_HC','0') != '0' or env.get('DS41_DENSE_CANONICAL_TILE','0') != '32'
            or env.get('DS41_VERIFY_MAX_ROWS','8') not in ('128','256','512','1024')):
        raise ValueError(FLAG+' requires strict dense/batch, denseM32, fusedHC0 and explicit cap128/256/512/1024')
    for name in ('DS41_LAYER_PROMOTION','DS41_LAYER_PROMOTION_HOST_BACKING','DS41_ROUTE_SNAPSHOT',
                 'DS41_NEXT_CHUNK_PREFETCH','DS41_STABLE_LINEAR_GRAPHS'):
        if env.get(name,'0') != '0':
            raise ValueError(FLAG+' initially excludes '+name)
    return dict(schema=1,variant=VARIANT,numerical_target=TARGET,
        verification_max_rows=int(env['DS41_VERIFY_MAX_ROWS']),
        residual_dtype='torch.bfloat16',hc_post_accumulator_dtype='torch.float32',
        hc_pre_normalization_dtype='torch.float32',normalized_branch_dtype='torch.float16',
        main_hidden_dtype='torch.bfloat16',engram_residual_dtype='torch.bfloat16',
        hc_graph_hidden_dtype='torch.bfloat16',dense_and_routed_CUDA_unchanged=True,
        old_Half_target_certificate_inherited=False,GPU_numerically_verified=False,
        full_model_gate_passed=False)


def validate_cli(args, environment=None):
    value = configuration(environment)
    if value is None:
        return None
    forbidden = ('cpu_experts','adaptive_gpu','adaptive_host_backing','personal_profile_state',
        'personal_profile_apply','personal_prompt_retain_gib','router_lookahead','lookahead_prefetch',
        'next_chunk_prefetch','ignore_eos','allow_missing_engram','allow_unverified_speculation')
    if (args.speculation != 'lookup' or args.prefill_chunk_size != 128
            or any(bool(getattr(args,name,False)) for name in forbidden)
            or getattr(args,'personal_profile_max_routed_slots',256) != 256
            or not args.check_parity or not args.check_commit_prefix
            or args.parity_tokens != int((os.environ if environment is None else environment).get('DS41_VERIFY_MAX_ROWS','8'))
            or args.speculation_commit != 'truncate'):
        raise ValueError(FLAG+' initially supports only full-model lookup/prefill128, normalEOS, no personal/draft/CPU/adaptive/promotion/unsafe override')
    return value


def bind_model(model):
    value = configuration()
    if value is None:
        return
    model._hc_residual_bf16 = True
    model._hc_residual_configuration = value
    model._hc_residual_source = {
        'helper':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model':hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest()}


def verify_model(model):
    if configuration() != model._hc_residual_configuration:
        raise RuntimeError('HC residual numerical target configuration changed')


def metadata(model):
    if not getattr(model,'_hc_residual_bf16',False):
        return None
    verify_model(model)
    hidden = getattr(model,'main_hidden',None)
    if hidden is not None and str(hidden.dtype) != 'torch.bfloat16':
        raise RuntimeError('HC residual target main_hidden has the wrong dtype')
    sources = {'helper':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'model':hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest()}
    if sources != model._hc_residual_source:
        raise RuntimeError('HC residual numerical target source changed')
    return dict(configuration=model._hc_residual_configuration,source_sha256=sources,
        no_new_CUDA_module=True,existing_model_stream_graph_and_close_owners=True,
        distinct_from_legacy_numerical_target=True)


def pre_normalized_half(x,pre,norm,eps,rms):
    import torch
    if x.dtype != torch.bfloat16 or pre.dtype != torch.float32:
        raise ValueError('BF16 HC residual pre requires BFloat16 residual and Float32 coefficients')
    # Normalize before the Half branch cast: a finite raw residual sum may
    # exceed65504 even though its normalized activation is safe for EXL3.
    combined = torch.sum(pre.unsqueeze(-1) * x.float(), dim=2)
    return rms(combined,norm,eps).to(torch.float16)


def post_bfloat16(x,residual,post,comb):
    import torch
    if (x.dtype != torch.float16 or residual.dtype != torch.bfloat16
            or post.dtype != torch.float32 or comb.dtype != torch.float32):
        raise ValueError('BF16 HC post requires Half branch, BFloat16 residual and Float32 coefficients')
    # Keep the legacy FP32 expression/order; only the destination cast changes.
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.float().unsqueeze(-2), dim=2)
    return y.to(torch.bfloat16)
