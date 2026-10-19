"""Single-launch sparse-KV preparation, preserving the existing attention math.

Decode can build ring indices internally. Short verification supplies the same
canonical window indices as the reference; compressed rows keep per-query
segments and fixed padded key order. Unsupported inputs return None.
"""
from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline

_CPP = r"""
#include <torch/extension.h>
void prepare_kv_cuda(torch::Tensor window, torch::Tensor window_idx,
                     torch::Tensor codes, torch::Tensor scales, torch::Tensor selected,
                     torch::Tensor output, torch::Tensor indices, int64_t compress_len,
                     int64_t window_size, int64_t index_topk, int64_t position);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__device__ __forceinline__ int64_t read_index(const void* pointer, bool wide, int i) {
    return wide ? static_cast<const int64_t*>(pointer)[i] : static_cast<const int*>(pointer)[i];
}

__device__ __forceinline__ float fp4_mag(int nib) {
    int c = nib & 7;
    float value = c == 0 ? 0.f : c == 1 ? .5f : c == 2 ? 1.f : c == 3 ? 1.5f
                : c == 4 ? 2.f : c == 5 ? 3.f : c == 6 ? 4.f : 6.f;
    return (nib & 8) ? -value : value;
}

template <typename scalar_t>
__global__ void prepare_kv_kernel(
    const scalar_t* window, const void* window_idx, bool window_wide,
    const unsigned char* codes, const unsigned char* scales,
    const void* selected, bool selected_wide, scalar_t* output, int* indices,
    int nw, int capacity, int dim, int queries, int supplied, int compress_len,
    int ws, int topk, int position
) {
    int batch = blockIdx.y, row = blockIdx.x, t = threadIdx.x;
    int out_rows = nw + queries * topk;
    if (row < out_rows) {
        scalar_t* dst = output + (static_cast<int64_t>(batch) * out_rows + row) * dim;
        if (row < nw) {
            const scalar_t* src = window + (static_cast<int64_t>(batch) * nw + row) * dim;
            for (int d = t; d < dim; d += blockDim.x) dst[d] = src[d];
        } else {
            int r = row - nw, query = r / topk, rank = r % topk;
            int selected_row = 0;
            bool present = rank < supplied && compress_len > 0;
            if (present) {
                int64_t raw = read_index(selected, selected_wide, (batch * queries + query) * supplied + rank);
                selected_row = raw < nw ? 0 : raw >= static_cast<int64_t>(nw) + compress_len
                             ? compress_len - 1 : static_cast<int>(raw - nw);
            }
            for (int pair = t; pair < dim / 2; pair += blockDim.x) {
                float lo = 0.f, hi = 0.f;
                if (present) {
                    unsigned char packed = codes[(static_cast<int64_t>(batch) * capacity + selected_row) * (dim / 2) + pair];
                    __nv_fp8_e4m3 scale;
                    scale.__x = scales[(static_cast<int64_t>(batch) * capacity + selected_row) * (dim / 16) + pair / 8];
                    float s = static_cast<float>(scale);
                    hi = fp4_mag(packed >> 4) * s;
                    lo = fp4_mag(packed & 15) * s;
                }
                dst[pair * 2] = static_cast<scalar_t>(hi);
                dst[pair * 2 + 1] = static_cast<scalar_t>(lo);
            }
        }
    }
    int keys = ws + topk;
    if (t == 0 && row < queries * keys) {
        int query = row / keys, key = row % keys, value = -1;
        if (key < ws) {
            if (window_idx) value = static_cast<int>(read_index(window_idx, window_wide, (batch * queries + query) * ws + key));
            else {
                value = ((position % ws) + 1 + key) % ws;
                if (value > position) value = -1;
            }
        } else if (key - ws < supplied && compress_len > 0) {
            int rank = key - ws;
            int64_t raw = read_index(selected, selected_wide, (batch * queries + query) * supplied + rank);
            if (raw >= nw && raw < static_cast<int64_t>(nw) + compress_len)
                value = nw + query * topk + rank;
        }
        indices[(batch * queries + query) * keys + key] = value;
    }
}

void prepare_kv_cuda(torch::Tensor window, torch::Tensor window_idx,
                     torch::Tensor codes, torch::Tensor scales, torch::Tensor selected,
                     torch::Tensor output, torch::Tensor indices, int64_t compress_len,
                     int64_t window_size, int64_t index_topk, int64_t position) {
    const c10::cuda::CUDAGuard guard(window.device());
    int b = window.size(0), nw = window.size(1), dim = window.size(2);
    int queries = selected.size(1), supplied = selected.size(2);
    int rows = std::max(nw + queries * static_cast<int>(index_topk), queries * static_cast<int>(window_size + index_topk));
    dim3 grid(rows, b);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto wi = window_idx.numel() ? window_idx.data_ptr() : nullptr;
    auto cp = reinterpret_cast<const unsigned char*>(codes.data_ptr());
    auto sp = reinterpret_cast<const unsigned char*>(scales.data_ptr());
    if (window.scalar_type() == at::kHalf)
        prepare_kv_kernel<at::Half><<<grid, 256, 0, stream>>>(
            window.data_ptr<at::Half>(), wi, window_idx.scalar_type() == at::kLong,
            cp, sp, selected.data_ptr(), selected.scalar_type() == at::kLong,
            output.data_ptr<at::Half>(), indices.data_ptr<int>(), nw, codes.size(1), dim,
            queries, supplied, compress_len, window_size, index_topk, position);
    else
        prepare_kv_kernel<at::BFloat16><<<grid, 256, 0, stream>>>(
            window.data_ptr<at::BFloat16>(), wi, window_idx.scalar_type() == at::kLong,
            cp, sp, selected.data_ptr(), selected.scalar_type() == at::kLong,
            output.data_ptr<at::BFloat16>(), indices.data_ptr<int>(), nw, codes.size(1), dim,
            queries, supplied, compress_len, window_size, index_topk, position);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def _extension():
    return load_inline("ds41_kv_prepare_v1", cpp_sources=_CPP, cuda_sources=_CUDA,
        functions=["prepare_kv_cuda"], extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


def prepare_sparse_kv(window, window_idx, codes, scales, selected, compress_len,
                      window_size, index_topk, *, position=None):
    """Return (dense KV, canonical indices), or None for unsupported layouts.

    window [B,Nw,D], codes [B,capacity,D/2], scales E4M3 [B,capacity,D/16],
    selected [B,Q,I] offsets include Nw (same contract as Attention._index).
    Explicit window_idx is [B,Q,window_size]. For decode only, None derives
    ring indices from position; that requires Q=1 and Nw=window_size.
    Each query's compressed segment is index_topk rows, even if I is smaller.
    """
    tensors = (window, codes, scales, selected)
    if (not all(isinstance(t, torch.Tensor) and t.is_cuda and t.is_contiguous() for t in tensors)
            or any(t.device != window.device for t in tensors)
            or window.dtype not in (torch.float16, torch.bfloat16)
            or codes.dtype != torch.uint8 or scales.dtype != torch.float8_e4m3fn
            or selected.dtype not in (torch.int32, torch.int64)
            or any(t.ndim != 3 for t in tensors)
            or not 1 <= selected.shape[1] <= 8 or not 0 < window_size <= 256
            or index_topk not in (512, 1024) or not 0 <= selected.shape[2] <= index_topk):
        return None
    b, nw, dim = window.shape
    if (b < 1 or nw < 1 or dim < 16 or dim % 16 or codes.shape != (b, codes.shape[1], dim // 2)
            or scales.shape != (b, codes.shape[1], dim // 16) or selected.shape[0] != b
            or not 0 <= compress_len <= codes.shape[1]):
        return None
    if window_idx is None:
        if selected.shape[1] != 1 or nw != window_size or position is None or position < 1:
            return None
        window_idx = torch.empty(0, dtype=torch.int32, device=window.device)
    elif (not isinstance(window_idx, torch.Tensor) or not window_idx.is_cuda
          or not window_idx.is_contiguous() or window_idx.device != window.device
          or window_idx.dtype not in (torch.int32, torch.int64)
          or window_idx.shape != (b, selected.shape[1], window_size)):
        return None
    out = torch.empty(b, nw + selected.shape[1] * index_topk, dim, dtype=window.dtype, device=window.device)
    indices = torch.empty(b, selected.shape[1], window_size + index_topk, dtype=torch.int32, device=window.device)
    _extension().prepare_kv_cuda(window, window_idx, codes, scales, selected, out, indices,
                                 int(compress_len), int(window_size), int(index_topk), int(position or 0))
    return out, indices
