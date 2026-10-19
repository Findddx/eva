// Independent query-grid frontend; original per-pair math in kernels.cuh.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "kernels.cuh"

namespace ds41_canonical_rope_batch {

#ifndef DS41_CANONICAL_MAX_QUERIES
#define DS41_CANONICAL_MAX_QUERIES 128
#endif
static_assert(DS41_CANONICAL_MAX_QUERIES == 128 || DS41_CANONICAL_MAX_QUERIES == 256,
              "Only the original128 or explicitly built256 tier is supported");

void run_(const at::Tensor& x, const at::Tensor& frequencies, int64_t inverse) {
    TORCH_CHECK(x.is_cuda(), "canonical_rope_batch requires CUDA");
    const c10::cuda::CUDAGuard guard(x.device());
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props, x.device().index()));
    TORCH_CHECK(props.major == 12 && props.minor == 0, "canonical_rope_batch candidate requires SM120");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "canonical_rope_batch capture is unsupported");
    TORCH_CHECK(x.scalar_type() == at::kHalf && x.is_contiguous(), "canonical_rope_batch x dtype/contiguity");
    TORCH_CHECK((x.dim() == 3 && x.size(0) == 1 && x.size(2) == 512)
                || (x.dim() == 4 && x.size(0) == 1 && x.size(2) == 64 && x.size(3) == 512),
                "canonical_rope_batch x must be [1,M,512] or [1,M,64,512]");
    const int queries = (int)x.size(1);
    const int heads = x.dim() == 4 ? 64 : 1;
    TORCH_CHECK(queries >= 1 && queries <= DS41_CANONICAL_MAX_QUERIES, "canonical_rope_batch query capacity");
    TORCH_CHECK(frequencies.is_cuda() && frequencies.device() == x.device()
                && frequencies.scalar_type() == at::kFloat && frequencies.is_contiguous()
                && frequencies.dim() == 3 && frequencies.size(0) == queries
                && frequencies.size(1) == 32 && frequencies.size(2) == 2,
                "canonical_rope_batch real frequency view shape/dtype/device/contiguity");
    TORCH_CHECK(inverse == 0 || inverse == 1, "canonical_rope_batch inverse must be 0/1");
    const uintptr_t a = (uintptr_t)x.data_ptr(), b = (uintptr_t)frequencies.data_ptr();
    TORCH_CHECK(!(a < b+frequencies.numel()*frequencies.element_size()
                  && b < a+x.numel()*x.element_size()), "canonical_rope_batch writable input aliases frequencies");
    using namespace ds41_canonical_rope_batch_ns;
    // Same32threads, same per-query head rows and original Half row stride.
    // The base pointer is the same tail view the original wrapper passed.
    rope_tail_queries_kernel<<<dim3(heads,queries),32,0,stream>>>(
        (half*)x.data_ptr()+448,frequencies.data_ptr<float>(),heads,512,(int)inverse);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace ds41_canonical_rope_batch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run_", &ds41_canonical_rope_batch::run_, "Original M1 RoPE on independent query grids");
}
