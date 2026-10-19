"""CUDA decode HC collapse+RMS and HC residual mapping.

Unsupported inputs return None so callers retain their eager/CPU behavior.
The collapse is rounded to its input dtype BEFORE RMS, matching hc_pre + rms.
comb is indexed [input_stream, output_stream], matching the dim=2 reduction.
"""
import torch
from torch.utils.cpp_extension import load_inline

_CPP = r"""
#include <torch/extension.h>
void hc_pre_rms_cuda(torch::Tensor x, torch::Tensor pre, torch::Tensor norm,
                      torch::Tensor out, double eps);
void hc_post_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor post,
                  torch::Tensor comb, torch::Tensor out);
"""

_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_down_sync(0xffffffff, x, offset);
    return x;
}

template <typename data_t, typename norm_t>
__global__ void pre_rms_kernel(const data_t* __restrict__ x,
                               const float* __restrict__ pre,
                               const norm_t* __restrict__ norm,
                               data_t* __restrict__ out, int dim, float eps) {
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int warps = blockDim.x >> 5;
    extern __shared__ short collapsed_raw[];
    data_t* collapsed = reinterpret_cast<data_t*>(collapsed_raw);
    __shared__ float reductions[32];
    __shared__ float inverse_rms;
    const data_t* xr = x + static_cast<long long>(row) * 4 * dim;
    const float* p = pre + row * 4;
    float sum_sq = 0.f;
    for (int d = t; d < dim; d += blockDim.x) {
        // fmad=false keeps the same multiply then reduction rounding as eager.
        float h = p[0] * static_cast<float>(xr[d]);
        h += p[1] * static_cast<float>(xr[dim + d]);
        h += p[2] * static_cast<float>(xr[2 * dim + d]);
        h += p[3] * static_cast<float>(xr[3 * dim + d]);
        data_t rounded = static_cast<data_t>(h);
        collapsed[d] = rounded;
        float hf = static_cast<float>(rounded);
        sum_sq += hf * hf;
    }
    sum_sq = warp_sum(sum_sq);
    if (lane == 0) reductions[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        sum_sq = lane < warps ? reductions[lane] : 0.f;
        sum_sq = warp_sum(sum_sq);
        if (lane == 0) inverse_rms = rsqrtf(sum_sq / dim + eps);
    }
    __syncthreads();
    data_t* yr = out + static_cast<long long>(row) * dim;
    for (int d = t; d < dim; d += blockDim.x) {
        float y = static_cast<float>(norm[d]) * static_cast<float>(collapsed[d]);
        yr[d] = static_cast<data_t>(y * inverse_rms);
    }
}

template <typename data_t, typename residual_t>
__global__ void post_kernel(const data_t* __restrict__ x,
                            const residual_t* __restrict__ residual,
                            const float* __restrict__ post,
                            const float* __restrict__ comb,
                            data_t* __restrict__ out, int dim, long long count) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int d = i % dim;
    const int stream = (i / dim) % 4;
    const long long row = i / (4 * dim);
    const residual_t* r = residual + row * 4 * dim + d;
    const float* c = comb + row * 16 + stream;
    float mixed = c[0] * static_cast<float>(r[0]);
    mixed += c[4] * static_cast<float>(r[dim]);
    mixed += c[8] * static_cast<float>(r[2 * dim]);
    mixed += c[12] * static_cast<float>(r[3 * dim]);
    float projected = post[row * 4 + stream] * static_cast<float>(x[row * dim + d]);
    out[i] = static_cast<data_t>(projected + mixed);
}

template <typename T>
void launch_pre(torch::Tensor x, torch::Tensor pre, torch::Tensor norm,
                torch::Tensor out, double eps) {
    const int dim = x.size(-1);
    const int rows = x.numel() / (4 * dim);
    const int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto xp = x.data_ptr<T>();
    auto op = out.data_ptr<T>();
    if (norm.scalar_type() == at::kFloat)
        pre_rms_kernel<T, float><<<rows, threads, dim * sizeof(half), stream>>>(
            xp, pre.data_ptr<float>(), norm.data_ptr<float>(), op, dim, eps);
    else if (norm.scalar_type() == at::kHalf)
        pre_rms_kernel<T, at::Half><<<rows, threads, dim * sizeof(half), stream>>>(
            xp, pre.data_ptr<float>(), norm.data_ptr<at::Half>(), op, dim, eps);
    else
        pre_rms_kernel<T, at::BFloat16><<<rows, threads, dim * sizeof(half), stream>>>(
            xp, pre.data_ptr<float>(), norm.data_ptr<at::BFloat16>(), op, dim, eps);
}

void hc_pre_rms_cuda(torch::Tensor x, torch::Tensor pre, torch::Tensor norm,
                      torch::Tensor out, double eps) {
    const c10::cuda::CUDAGuard guard(x.device());
    if (x.scalar_type() == at::kHalf) launch_pre<at::Half>(x, pre, norm, out, eps);
    else launch_pre<at::BFloat16>(x, pre, norm, out, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hc_post_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor post,
                  torch::Tensor comb, torch::Tensor out) {
    const c10::cuda::CUDAGuard guard(x.device());
    long long count = residual.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    #define POST_DISPATCH(T, R) post_kernel<T, R><<<(count + 255) / 256, 256, 0, stream>>>( \
        x.data_ptr<T>(), residual.data_ptr<R>(), post.data_ptr<float>(), comb.data_ptr<float>(), \
        out.data_ptr<T>(), x.size(-1), count)
    if (x.scalar_type() == at::kHalf) {
        if (residual.scalar_type() == at::kHalf) { POST_DISPATCH(at::Half, at::Half); }
        else { POST_DISPATCH(at::Half, at::BFloat16); }
    } else {
        if (residual.scalar_type() == at::kHalf) { POST_DISPATCH(at::BFloat16, at::Half); }
        else { POST_DISPATCH(at::BFloat16, at::BFloat16); }
    }
    #undef POST_DISPATCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="ds41_hc_fused", cpp_sources=_CPP, cuda_sources=_SRC,
            functions=["hc_pre_rms_cuda", "hc_post_cuda"],
            extra_cuda_cflags=["-O3", "--fmad=false"], with_cuda=True, verbose=False,
        )
    return _EXT


def _compatible(*tensors):
    return all(t.is_cuda and t.is_contiguous() and t.device == tensors[0].device for t in tensors)


def hc_pre_rms(x, pre, norm, eps):
    if x.ndim < 2 or x.shape[-2] != 4 or x.dtype not in (torch.float16, torch.bfloat16) or pre.dtype != torch.float32:
        return None
    if not _compatible(x, pre, norm) or norm.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return None
    dim = x.shape[-1]
    if not dim or dim > 16384 or not x.numel() or norm.numel() != dim or pre.shape != x.shape[:-1]:
        return None
    out = torch.empty(*x.shape[:-2], dim, device=x.device, dtype=x.dtype)
    _ext().hc_pre_rms_cuda(x, pre, norm, out, float(eps))
    return out


def hc_post(x, residual, post, comb):
    if x.dtype not in (torch.float16, torch.bfloat16) or residual.dtype not in (torch.float16, torch.bfloat16) or any(t.dtype != torch.float32 for t in (post, comb)):
        return None
    if not _compatible(x, residual, post, comb) or residual.shape != (*x.shape[:-1], 4, x.shape[-1]):
        return None
    if not x.numel() or post.shape != (*x.shape[:-1], 4) or comb.shape != (*x.shape[:-1], 4, 4):
        return None
    out = torch.empty_like(residual, dtype=x.dtype)
    _ext().hc_post_cuda(x, residual, post, comb, out)
    return out
