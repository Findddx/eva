// Project-local canonical M1 RoPE; exact source/patches in provenance.json.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ds41_canonical_rope512_ns {
__global__ void rope_tail_queries_kernel(half* x, const float* freqs, int rows, int stride, int inverse) {
    const long long query = blockIdx.y;
    x += query * rows * stride;
    freqs += query * 64;
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

}  // namespace ds41_canonical_rope512_ns
