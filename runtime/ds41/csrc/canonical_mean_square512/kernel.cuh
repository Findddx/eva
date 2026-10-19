// Arithmetic schedule reimplementation from pinned PyTorch ATen Reduce.cuh.
// PyTorch source is BSD-3-Clause; provenance.json records exact originals.
// This is a hypothesis of byte parity until the independent actual-GPU gate.
#pragma once
#include <cuda_runtime.h>

namespace ds41_mean_square512 {

template<int THREADS>
__global__ void square_mean_rows(const float* x, float* output, int width, float factor) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const float4* input = reinterpret_cast<const float4*>(x + row * width);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    // M1 vectorized ATen has four independent accumulators, step=block_width.
    for (int index = lane; index < width / 4; index += THREADS) {
        const float4 v = input[index];
        a0 = __fadd_rn(a0, __fmul_rn(v.x, v.x));
        a1 = __fadd_rn(a1, __fmul_rn(v.y, v.y));
        a2 = __fadd_rn(a2, __fmul_rn(v.z, v.z));
        a3 = __fadd_rn(a3, __fmul_rn(v.w, v.w));
    }
    float value = __fadd_rn(__fadd_rn(__fadd_rn(a0, a1), a2), a3);
    __shared__ float shared[THREADS];
    if constexpr (THREADS > 32) {
        shared[lane] = value;
        for (int offset = THREADS / 2; offset >= 32; offset >>= 1) {
            __syncthreads();
            if (lane < offset) {
                value = __fadd_rn(value, shared[lane + offset]);
                shared[lane] = value;
            }
        }
    }
    __syncthreads();
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    if (lane == 0) output[row] = __fmul_rn(value, factor);
}

}  // namespace ds41_mean_square512
