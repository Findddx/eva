"""Decode rope and sparse attention. One launch per layer instead of a chain of tiny ops."""
import torch
from torch.utils.cpp_extension import load_inline

_CPP = r"""
#include <torch/extension.h>
void rope_tail_cuda(torch::Tensor x, torch::Tensor freqs, int64_t inverse);
void sparse_decode_cuda(torch::Tensor q, torch::Tensor kv, torch::Tensor idx,
                        torch::Tensor sink, torch::Tensor out, torch::Tensor partial,
                        torch::Tensor stats, double scale);
"""

_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__global__ void rope_tail_kernel(half* x, const float* freqs, int rows, int stride, int inverse) {
    int row = blockIdx.x;
    int pair = threadIdx.x;
    if (row >= rows || pair >= 32) return;
    half* p = x + row * stride + pair * 2;
    float a = __half2float(p[0]);
    float b = __half2float(p[1]);
    float fr = freqs[pair * 2];
    float fi = freqs[pair * 2 + 1];
    if (inverse) fi = -fi;
    p[0] = __float2half(a * fr - b * fi);
    p[1] = __float2half(a * fi + b * fr);
}

void rope_tail_cuda(torch::Tensor x, torch::Tensor freqs, int64_t inverse) {
    const c10::cuda::CUDAGuard guard(x.device());
    int rows = x.numel() / 64;
    int stride = x.dim() >= 2 ? x.stride(-2) : 64;
    rope_tail_kernel<<<rows, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<half*>(x.data_ptr<at::Half>()),
        freqs.data_ptr<float>(),
        rows, stride, static_cast<int>(inverse)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_down_sync(0xffffffff, x, offset);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x = fmaxf(x, __shfl_down_sync(0xffffffff, x, offset));
    return x;
}

__global__ void sparse_decode_kernel(
    const half* __restrict__ q, const half* __restrict__ kv, const int* __restrict__ idx,
    const float* __restrict__ sink, half* __restrict__ out,
    float* __restrict__ partial, float* __restrict__ stats,
    int dim, int n, int all_topk, int parts, float scale
) {
    const int h = blockIdx.x;
    const int part = blockIdx.y;
    const int start = parts == 1 ? 0 : part * 64;
    const int topk = min(all_topk - start, 64);
    idx += start;
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int warps = nt >> 5;
    __shared__ float scores[1024];
    __shared__ float qf[512];
    __shared__ float reductions[32];
    __shared__ float maximum, inverse_sum;
    const half* qh = q + static_cast<long long>(h) * dim;
    for (int d = t; d < dim; d += nt) qf[d] = __half2float(qh[d]);
    __syncthreads();
    // A warp owns one key: adjacent lanes read adjacent channels. The former
    // thread-per-key loop made every warp issue 512 strided loads per dot.
    for (int k = warp; k < topk; k += warps) {
        int id = idx[k];
        float acc = 0.f;
        if (static_cast<unsigned>(id) < static_cast<unsigned>(n)) {
            const half* row = kv + static_cast<long long>(id) * dim;
            for (int d = lane; d < dim; d += 32)
                acc = fmaf(qf[d], __half2float(row[d]), acc);
        }
        acc = warp_sum(acc);
        if (lane == 0) scores[k] = static_cast<unsigned>(id) < static_cast<unsigned>(n)
            ? acc * scale : -CUDART_INF_F;
    }
    __syncthreads();
    // The sink participates in the maximum as well as the denominator. This
    // avoids overflow when the sink dominates or all selected rows are empty.
    float m = sink[h];
    for (int k = t; k < topk; k += nt) m = fmaxf(m, scores[k]);
    m = warp_max(m);
    if (lane == 0) reductions[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = lane < warps ? reductions[lane] : -CUDART_INF_F;
        m = warp_max(m);
        if (lane == 0) maximum = m;
    }
    __syncthreads();
    float sum = t == 0 && part == 0 ? expf(sink[h] - maximum) : 0.f;
    for (int k = t; k < topk; k += nt) {
        float e = expf(scores[k] - maximum);
        scores[k] = e;
        sum += e;
    }
    sum = warp_sum(sum);
    if (lane == 0) reductions[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < warps ? reductions[lane] : 0.f;
        sum = warp_sum(sum);
        if (lane == 0) {
            inverse_sum = 1.f / sum;
            if (parts > 1) {
                stats[(h * parts + part) * 2] = maximum;
                stats[(h * parts + part) * 2 + 1] = sum;
            }
        }
    }
    __syncthreads();
    half* oh = out + static_cast<long long>(h) * dim;
    for (int d = t; d < dim; d += nt) {
        float acc = 0.f;
        for (int k = 0; k < topk; ++k) {
            int id = idx[k];
            if (static_cast<unsigned>(id) < static_cast<unsigned>(n))
                acc = fmaf(scores[k], __half2float(kv[static_cast<long long>(id) * dim + d]), acc);
        }
        if (parts == 1) oh[d] = __float2half(acc * inverse_sum);
        else partial[(static_cast<long long>(h) * parts + part) * dim + d] = acc;
    }
}

__global__ void sparse_merge_kernel(const float* __restrict__ partial,
                                     const float* __restrict__ stats,
                                     half* __restrict__ out, int parts, int dim) {
    int h = blockIdx.x;
    int t = threadIdx.x;
    __shared__ float factors[16];
    __shared__ float inverse_sum;
    if (t == 0) {
        float maximum = -CUDART_INF_F;
        for (int p = 0; p < parts; ++p) maximum = fmaxf(maximum, stats[(h * parts + p) * 2]);
        float sum = 0.f;
        for (int p = 0; p < parts; ++p) {
            float f = expf(stats[(h * parts + p) * 2] - maximum);
            factors[p] = f;
            sum += f * stats[(h * parts + p) * 2 + 1];
        }
        inverse_sum = 1.f / sum;
    }
    __syncthreads();
    for (int d = t; d < dim; d += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < parts; ++p)
            acc = fmaf(factors[p], partial[(static_cast<long long>(h) * parts + p) * dim + d], acc);
        out[static_cast<long long>(h) * dim + d] = __float2half(acc * inverse_sum);
    }
}

void sparse_decode_cuda(torch::Tensor q, torch::Tensor kv, torch::Tensor idx,
                        torch::Tensor sink, torch::Tensor out, torch::Tensor partial,
                        torch::Tensor stats, double scale) {
    const c10::cuda::CUDAGuard guard(q.device());
    int heads = q.size(0);
    int dim = q.size(1);
    int n = kv.size(0);
    int topk = idx.numel();
    int parts = (topk + 63) / 64;
    auto stream = at::cuda::getCurrentCUDAStream();
    sparse_decode_kernel<<<dim3(heads, parts), 256, 0, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(kv.data_ptr<at::Half>()),
        idx.data_ptr<int>(),
        sink.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        partial.data_ptr<float>(), stats.data_ptr<float>(),
        dim, n, topk, parts, static_cast<float>(scale)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (parts > 1) {
        sparse_merge_kernel<<<heads, 256, 0, stream>>>(
            partial.data_ptr<float>(), stats.data_ptr<float>(),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()), parts, dim);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="ds41_attn_fast",
            cpp_sources=_CPP,
            cuda_sources=_SRC,
            functions=["rope_tail_cuda", "sparse_decode_cuda"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            with_cuda=True,
            verbose=False,
        )
    return _EXT


def rope_tail_(x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False) -> bool:
    """Rotate the last 64 channels in place. False means the caller should use the eager path."""
    if not x.is_cuda or not freqs.is_cuda or x.device != freqs.device or x.dtype != torch.float16 or x.shape[-1] < 64:
        return False
    sl = x[..., -64:] if x.shape[-1] != 64 else x
    if sl.numel() % 64 or sl.stride(-1) != 1:
        return False
    # The kernel flattens all leading axes with one row stride.
    stride = sl.stride(-2) if sl.dim() >= 2 else 64
    expected = stride
    for axis in range(sl.dim() - 2, -1, -1):
        if sl.size(axis) > 1 and sl.stride(axis) != expected:
            return False
        expected *= sl.size(axis)
    fr = torch.view_as_real(freqs.reshape(-1)).contiguous()
    if fr.numel() != 64 or fr.dtype != torch.float32:
        return False
    _ext().rope_tail_cuda(sl, fr, int(inverse))
    return True


def sparse_decode_(q, kv, sink, idx, scale) -> torch.Tensor | None:
    """q [b,s,h,d], kv [b,n,d], idx [b,s,topk]. None asks for the eager kernel."""
    if q.dim() != 4 or kv.dim() != 3 or q.dtype != torch.float16 or kv.dtype != torch.float16:
        return None
    if not q.is_cuda or any(t.device != q.device for t in (kv, sink, idx)):
        return None
    b, s, h, d = q.shape
    if s != 1 or b != 1 or d != 512 or h > 128:
        return None
    if kv.shape[0] != b or kv.shape[-1] != d or sink.numel() != h or idx.shape[:2] != (b, s):
        return None
    topk = idx.shape[-1]
    if topk <= 0 or topk > 1024:
        return None
    qq = q.reshape(h, d)
    kk = kv.reshape(-1, d)
    ii = idx.reshape(-1).to(torch.int32).contiguous()
    if not qq.is_contiguous() or not kk.is_contiguous():
        return None
    out = torch.empty_like(qq)
    parts = (topk + 63) // 64
    partial = torch.empty((h, parts, d) if parts > 1 else (0,), device=q.device, dtype=torch.float32)
    stats = torch.empty((h, parts, 2) if parts > 1 else (0,), device=q.device, dtype=torch.float32)
    _ext().sparse_decode_cuda(qq, kk, ii, sink.reshape(-1).float().contiguous(), out, partial, stats, float(scale))
    return out.view(b, s, h, d)
