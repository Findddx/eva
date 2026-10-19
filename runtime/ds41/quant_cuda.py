"""Single-launch training-compatible activation quantization for decode."""
import torch
from torch.utils.cpp_extension import load_inline

_CPP = r"""
#include <torch/extension.h>
void fp8_roundtrip_cuda(torch::Tensor x, int64_t group);
void fp4_quant_cuda(torch::Tensor x, torch::Tensor codes, torch::Tensor scales,
                    torch::Tensor recon, int64_t group, bool power2);
void fp4_dequant_cuda(torch::Tensor codes, torch::Tensor scales, torch::Tensor out,
                      int64_t group, int64_t scale_kind);
"""

_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__device__ __forceinline__ float round_fp8(float v) {
    __nv_fp8_e4m3 q;
    q.__x = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    return static_cast<float>(q);
}

__device__ __forceinline__ float rounded_scale(float v) {
    unsigned bits = __float_as_uint(v);
    unsigned exponent = (bits >> 23) + ((bits & 0x7fffff) != 0);
    return __uint_as_float(exponent << 23);
}

template <typename scalar_t, bool FP4>
__global__ void quant_kernel(scalar_t* x, unsigned char* codes, unsigned char* scales,
                             scalar_t* recon, int groups, int group, bool power2) {
    int lane = threadIdx.x & 31;
    int g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (g >= groups) return;
    float v = lane < group ? static_cast<float>(x[g * group + lane]) : 0.f;
    float amax = fabsf(v);
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, offset));
    float scale = 0.f;
    if (lane == 0) {
        if (FP4) {
            scale = power2 ? rounded_scale(fmaxf(amax, 6.f * 0x1p-126f) / 6.f)
                           : round_fp8(fmaxf(amax, 6.f * 0x1p-9f) / 6.f);
            scales[g] = power2 ? static_cast<unsigned char>(__float_as_uint(scale) >> 23)
                              : __nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3);
        } else {
            scale = rounded_scale(fmaxf(amax, 1e-4f) / 448.f);
        }
    }
    scale = __shfl_sync(0xffffffff, scale, 0);
    if (FP4) {
        float a = fabsf(v / scale);
        int c = a <= .25f ? 0 : a < .75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3
              : a <= 2.5f ? 4 : a < 3.5f ? 5 : a <= 5.f ? 6 : 7;
        float mag = c == 0 ? 0.f : c == 1 ? .5f : c == 2 ? 1.f : c == 3 ? 1.5f
                  : c == 4 ? 2.f : c == 5 ? 3.f : c == 6 ? 4.f : 6.f;
        int nib = c | (v < 0.f ? 8 : 0);
        int next = __shfl_down_sync(0xffffffff, nib, 1);
        if (lane < group) {
            recon[g * group + lane] = static_cast<scalar_t>(copysignf(mag * scale, v));
            if ((lane & 1) == 0) codes[(g * group + lane) / 2] = (nib << 4) | next;
        }
    } else if (lane < group) {
        x[g * group + lane] = static_cast<scalar_t>(round_fp8(fminf(fmaxf(v / scale, -448.f), 448.f)) * scale);
    }
}

void fp8_roundtrip_cuda(torch::Tensor x, int64_t group) {
    const c10::cuda::CUDAGuard guard(x.device());
    int groups = x.numel() / group;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kHalf)
        quant_kernel<at::Half, false><<<(groups + 3) / 4, 128, 0, stream>>>(
            x.data_ptr<at::Half>(), nullptr, nullptr, nullptr, groups, group, true);
    else
        quant_kernel<at::BFloat16, false><<<(groups + 3) / 4, 128, 0, stream>>>(
            x.data_ptr<at::BFloat16>(), nullptr, nullptr, nullptr, groups, group, true);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp4_quant_cuda(torch::Tensor x, torch::Tensor codes, torch::Tensor scales,
                    torch::Tensor recon, int64_t group, bool power2) {
    const c10::cuda::CUDAGuard guard(x.device());
    int groups = x.numel() / group;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto cp = reinterpret_cast<unsigned char*>(codes.data_ptr());
    auto sp = reinterpret_cast<unsigned char*>(scales.data_ptr());
    if (x.scalar_type() == at::kHalf)
        quant_kernel<at::Half, true><<<(groups + 3) / 4, 128, 0, stream>>>(
            x.data_ptr<at::Half>(), cp, sp, recon.data_ptr<at::Half>(), groups, group, power2);
    else
        quant_kernel<at::BFloat16, true><<<(groups + 3) / 4, 128, 0, stream>>>(
            x.data_ptr<at::BFloat16>(), cp, sp, recon.data_ptr<at::BFloat16>(), groups, group, power2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ __forceinline__ float fp4_value(int nib) {
    int c = nib & 7;
    float mag = c == 0 ? 0.f : c == 1 ? .5f : c == 2 ? 1.f : c == 3 ? 1.5f
              : c == 4 ? 2.f : c == 5 ? 3.f : c == 6 ? 4.f : 6.f;
    return (nib & 8) ? -mag : mag;
}

template <typename scalar_t>
__global__ void dequant_kernel(const unsigned char* codes, const void* scales,
                               scalar_t* out, int count, int group, int kind) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int g = i * 2 / group;
    float scale;
    if (kind == 2) scale = static_cast<const float*>(scales)[g];
    else {
        unsigned char raw = static_cast<const unsigned char*>(scales)[g];
        if (kind == 1) scale = __uint_as_float(static_cast<unsigned>(raw) << 23);
        else {
            __nv_fp8_e4m3 v;
            v.__x = raw;
            scale = static_cast<float>(v);
        }
    }
    unsigned char packed = codes[i];
    out[i * 2] = static_cast<scalar_t>(fp4_value(packed >> 4) * scale);
    out[i * 2 + 1] = static_cast<scalar_t>(fp4_value(packed & 15) * scale);
}

void fp4_dequant_cuda(torch::Tensor codes, torch::Tensor scales, torch::Tensor out,
                      int64_t group, int64_t scale_kind) {
    const c10::cuda::CUDAGuard guard(codes.device());
    int count = codes.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto cp = codes.data_ptr<unsigned char>();
    auto sp = scales.data_ptr();
    if (out.scalar_type() == at::kHalf)
        dequant_kernel<at::Half><<<(count + 255) / 256, 256, 0, stream>>>(
            cp, sp, out.data_ptr<at::Half>(), count, group, scale_kind);
    else if (out.scalar_type() == at::kBFloat16)
        dequant_kernel<at::BFloat16><<<(count + 255) / 256, 256, 0, stream>>>(
            cp, sp, out.data_ptr<at::BFloat16>(), count, group, scale_kind);
    else
        dequant_kernel<float><<<(count + 255) / 256, 256, 0, stream>>>(
            cp, sp, out.data_ptr<float>(), count, group, scale_kind);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="ds41_quant_decode", cpp_sources=_CPP, cuda_sources=_SRC,
            functions=["fp8_roundtrip_cuda", "fp4_quant_cuda", "fp4_dequant_cuda"],
            extra_cuda_cflags=["-O3"], with_cuda=True, verbose=False,
        )
    return _EXT


def supported(x, group):
    return x.is_cuda and x.is_contiguous() and x.dtype in (torch.float16, torch.bfloat16) and group in (16, 32) and x.numel() > 0


def fp8_roundtrip_(x, group=32):
    _ext().fp8_roundtrip_cuda(x, group)
    return x


def fp4_quant(x, group=16, power2=False):
    scale_dtype = torch.float8_e8m0fnu if power2 else torch.float8_e4m3fn
    codes = torch.empty(*x.shape[:-1], x.shape[-1] // 2, dtype=torch.uint8, device=x.device)
    scales = torch.empty(*x.shape[:-1], x.shape[-1] // group, dtype=scale_dtype, device=x.device)
    recon = torch.empty_like(x)
    _ext().fp4_quant_cuda(x, codes, scales, recon, group, power2)
    return codes, scales, recon


def fp4_dequant(codes, scales, group=16, dtype=torch.float32):
    kind = 2 if scales.dtype == torch.float32 else 0 if scales.dtype == torch.float8_e4m3fn else 1
    out = torch.empty(*codes.shape[:-1], codes.shape[-1] * 2, device=codes.device, dtype=dtype)
    _ext().fp4_dequant_cuda(codes, scales, out, group, kind)
    return out
