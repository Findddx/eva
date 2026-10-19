"""One-launch Sinkhorn for hc_mult=4. The Python loop was most of a decode step."""
import torch
from torch.utils.cpp_extension import load_inline

_CPP = r"""
#include <torch/extension.h>
void hc_sinkhorn_cuda(torch::Tensor mixes, torch::Tensor scale, torch::Tensor base,
                      torch::Tensor pre, torch::Tensor post, torch::Tensor comb,
                      int64_t iters, double eps);
"""

_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

__global__ void hc_sinkhorn_kernel(
    const float* __restrict__ mixes,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    float* __restrict__ pre,
    float* __restrict__ post,
    float* __restrict__ comb,
    int n, int iters, float eps
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* m = mixes + i * 24;
    float s0 = scale[0], s1 = scale[1], s2 = scale[2];
    float* pre_o = pre + i * 4;
    float* post_o = post + i * 4;
    float* c_o = comb + i * 16;
    float c[4][4];
    for (int j = 0; j < 4; ++j) {
        pre_o[j] = 1.f / (1.f + expf(-(m[j] * s0 + base[j]))) + eps;
        post_o[j] = 2.f / (1.f + expf(-(m[4 + j] * s1 + base[4 + j])));
    }
    for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 4; ++k)
            c[j][k] = m[8 + j * 4 + k] * s2 + base[8 + j * 4 + k];
    for (int j = 0; j < 4; ++j) {
        float mx = c[j][0];
        for (int k = 1; k < 4; ++k) mx = fmaxf(mx, c[j][k]);
        float sum = 0.f;
        for (int k = 0; k < 4; ++k) {
            c[j][k] = expf(c[j][k] - mx);
            sum += c[j][k];
        }
        for (int k = 0; k < 4; ++k) c[j][k] = c[j][k] / sum + eps;
    }
    for (int k = 0; k < 4; ++k) {
        float sum = 0.f;
        for (int j = 0; j < 4; ++j) sum += c[j][k];
        float inv = 1.f / (sum + eps);
        for (int j = 0; j < 4; ++j) c[j][k] *= inv;
    }
    for (int t = 0; t < iters - 1; ++t) {
        for (int j = 0; j < 4; ++j) {
            float sum = 0.f;
            for (int k = 0; k < 4; ++k) sum += c[j][k];
            float inv = 1.f / (sum + eps);
            for (int k = 0; k < 4; ++k) c[j][k] *= inv;
        }
        for (int k = 0; k < 4; ++k) {
            float sum = 0.f;
            for (int j = 0; j < 4; ++j) sum += c[j][k];
            float inv = 1.f / (sum + eps);
            for (int j = 0; j < 4; ++j) c[j][k] *= inv;
        }
    }
    for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 4; ++k)
            c_o[j * 4 + k] = c[j][k];
}

void hc_sinkhorn_cuda(torch::Tensor mixes, torch::Tensor scale, torch::Tensor base,
                      torch::Tensor pre, torch::Tensor post, torch::Tensor comb,
                      int64_t iters, double eps) {
    const c10::cuda::CUDAGuard guard(mixes.device());
    int n = mixes.size(0);
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    hc_sinkhorn_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        mixes.data_ptr<float>(), scale.data_ptr<float>(), base.data_ptr<float>(),
        pre.data_ptr<float>(), post.data_ptr<float>(), comb.data_ptr<float>(),
        n, static_cast<int>(iters), static_cast<float>(eps)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="ds41_hc_sinkhorn",
            cpp_sources=_CPP,
            cuda_sources=_SRC,
            functions=["hc_sinkhorn_cuda"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            with_cuda=True,
            verbose=False,
        )
    return _EXT


def hc_split_sinkhorn_cuda(mixes, hc_scale, hc_base, hc_mult, iters, eps):
    if hc_mult != 4 or mixes.shape[-1] != 24:
        return None
    if not mixes.is_cuda or mixes.dtype != torch.float32 or any(
        t.device != mixes.device or t.dtype != torch.float32 for t in (hc_scale, hc_base)
    ):
        return None
    flat = mixes.reshape(-1, 24).contiguous()
    n = flat.shape[0]
    # Each graph/caller owns its outputs. Global ping-pong buffers silently
    # aliased outputs after three calls and raced across streams and graphs.
    pre = torch.empty(n, 4, device=flat.device, dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty(n, 4, 4, device=flat.device, dtype=torch.float32)
    scale = hc_scale if hc_scale.is_contiguous() else hc_scale.contiguous()
    base = hc_base if hc_base.is_contiguous() else hc_base.contiguous()
    _ext().hc_sinkhorn_cuda(flat, scale, base, pre, post, comb, int(iters), float(eps))
    lead = mixes.shape[:-1]
    return pre.view(*lead, 4), post.view(*lead, 4), comb.view(*lead, 4, 4)
