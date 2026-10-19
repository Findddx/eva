"""Stateless, opt-in full-source KV packing for the existing large-KV owner.

No CUDA is compiled here. The caller retains operands, uses its original
stream/lease/close, and passes the unchanged FP4 dequantization entry.
"""
import os

FLAG='DS41_FULL_COMPRESSED_KV'
PROTOCOL='full_compressed_source_KV_original_quant_v1'
QUANT_SOURCE_SHA256='1bdb8cbe21cdd51161de99a392b49141129d0183f68ae9ac8534336d932bf98f'
COUNTERS=('full_calls','full_queries','full_output_requested_bytes','full_dequant_requested_bytes',
          'fallback_calls','fallback_queries','fallback_output_requested_bytes')


def _require(value,message):
    if not value:raise ValueError(message)


def requested(environ=None):
    value=(os.environ if environ is None else environ).get(FLAG,'0')
    _require(value in ('0','1'),FLAG+' must be0 or1')
    return value=='1'


def layout(queries,nw,supplied,*,capacity,length):
    _require(all(type(v) is int for v in (queries,nw,supplied,capacity,length)), 'Full-source KV dimensions must be plain integers')
    _require(9<=queries<=1024 and 1<=nw<=queries+127 and 0<=supplied<=512
        and 0<=length<=capacity<=262144,'Unsupported full-source KV geometry')
    return dict(output_shape=(1,nw+length,512),indices_shape=(1,queries,640),
        output_bytes=(nw+length)*1024,indices_bytes=queries*640*4,dequant_bytes=length*1024,
        eligible=0<length<queries*supplied)


def configuration(cap):
    _require(type(cap) is int and (9<=cap<=128 or cap in (256,512,1024)),'Unsupported full-source canonical cap')
    value = dict(protocol=PROTOCOL,requested=True,verification_cap=cap,
        cost_policy='Use only0<compress_len<queries*supplied; otherwise original large-KV CUDA pack',
        max_source_capacity=262144,single_original_owner=True,original_quant_cuda=True,new_JIT_modules=0,
        graph_capture_supported=False,cross_layer_cache=False,legacy_small_queries_unchanged=True,
        owned_outputs=True,all_invalid_indices_masked_before_Int32=True,ordered_K640_attention_unchanged=True,
        production_GPU_verified=False,full_model_gate_passed=False)
    if cap==1024:
        value.update(protocol='full_compressed_source_KV_original_quant_1024_v1',
            mandatory_above512=True,maximum_queries=1024,required_supplied_above512=512,
            above512_cost_rejection='Fail before any output allocation; no eager fallback')
    return value


def prepare(window,window_idx,codes,scales,selected,length,*,dequant,charge):
    """Caller validated tensors/recorded input uses on its fixed stream.

    charge records CPU allocation requests only. It retains no tensor or state.
    On failure the caller must retain this exception/traceback until drain.
    """
    import torch
    import torch.nn.functional as F
    meta=layout(selected.shape[1],window.shape[1],selected.shape[2],capacity=codes.shape[1],length=length)
    _require(meta['eligible'],'Full-source helper called for cost-rejected geometry')
    charge('dequant',meta['dequant_bytes'])
    ckv=dequant(codes[:,:length],scales[:,:length],16,dtype=window.dtype)
    output=torch.cat((window,ckv),dim=1)
    charge('output',meta['output_bytes'])
    nw=window.shape[1]
    valid=(selected>=nw)&(selected<nw+length)
    compressed=torch.where(valid,selected,-1).to(torch.int32)
    indices=torch.cat((window_idx.to(torch.int32),compressed),dim=-1)
    indices=F.pad(indices,(0,640-indices.shape[-1]),value=-1)
    charge('indices',meta['indices_bytes'])
    return output,indices
