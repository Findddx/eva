// Project-local canonical M1 source; see provenance.json for exact origin/patches.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace ds41_canonical_sparse512_ns {
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

__global__ void sparse_decode_queries_kernel(
    const half* __restrict__ q, const half* __restrict__ kv, const int* __restrict__ idx,
    const float* __restrict__ sink, half* __restrict__ out,
    float* __restrict__ partial, float* __restrict__ stats,
    int dim, int n, int all_topk, int parts, float scale
) {
    const long long query = blockIdx.z;
    q += query * 64 * dim;
    idx += query * all_topk;
    out += query * 64 * dim;
    partial += query * 64 * parts * dim;
    stats += query * 64 * parts * 2;
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

__global__ void sparse_merge_queries_kernel(const float* __restrict__ partial,
                                     const float* __restrict__ stats,
                                     half* __restrict__ out, int parts, int dim) {
    const long long query = blockIdx.y;
    partial += query * 64 * parts * dim;
    stats += query * 64 * parts * 2;
    out += query * 64 * dim;
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

}  // namespace ds41_canonical_sparse512_ns
