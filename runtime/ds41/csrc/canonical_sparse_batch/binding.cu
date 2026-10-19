// Project-local independent query-grid launcher. No production symbol changes.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "kernels.cuh"

namespace ds41_canonical_sparse_batch {

#ifndef DS41_CANONICAL_MAX_QUERIES
#define DS41_CANONICAL_MAX_QUERIES 128
#endif
static_assert(DS41_CANONICAL_MAX_QUERIES == 128 || DS41_CANONICAL_MAX_QUERIES == 256,
              "Only the original128 or explicitly built256 tier is supported");
constexpr int max_queries = DS41_CANONICAL_MAX_QUERIES;
constexpr int max_kv_rows = 127 + max_queries * (1 + 512);

static void tensor(const at::Tensor& t, at::Device device, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.device() == device && t.scalar_type() == dtype && t.is_contiguous(),
                "canonical_sparse_batch: invalid ", name, " device/dtype/contiguity");
}

static bool overlaps(const at::Tensor& a, const at::Tensor& b) {
    if (!a.numel() || !b.numel()) return false;
    auto pa = (uintptr_t)a.data_ptr(), pb = (uintptr_t)b.data_ptr();
    return pa < pb + b.numel()*b.element_size() && pb < pa + a.numel()*a.element_size();
}

void run(const at::Tensor& q, const at::Tensor& kv, const at::Tensor& sink,
         const at::Tensor& idx, const at::Tensor& out, const at::Tensor& partial,
         const at::Tensor& stats, double scale) {
    TORCH_CHECK(q.is_cuda(), "canonical_sparse_batch requires CUDA");
    const c10::cuda::CUDAGuard guard(q.device());
    auto device = q.device();
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props, device.index()));
    TORCH_CHECK(props.major == 12 && props.minor == 0, "canonical_sparse_batch prototype requires SM120");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "canonical_sparse_batch capture is unsupported");
    tensor(q, device, at::kHalf, "q");
    tensor(kv, device, at::kHalf, "kv");
    tensor(sink, device, at::kFloat, "sink");
    tensor(idx, device, at::kInt, "indices");
    tensor(out, device, at::kHalf, "output");
    tensor(partial, device, at::kFloat, "partial");
    tensor(stats, device, at::kFloat, "statistics");
    TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(1) >= 1 && q.size(1) <= max_queries
                && q.size(2) == 64 && q.size(3) == 512, "canonical_sparse_batch q shape");
    TORCH_CHECK(kv.dim() == 3 && kv.size(0) == 1 && kv.size(1) >= 0 && kv.size(1) <= max_kv_rows
                && kv.size(2) == 512, "canonical_sparse_batch kv shape");
    TORCH_CHECK(sink.dim() == 1 && sink.numel() == 64, "canonical_sparse_batch sink shape");
    TORCH_CHECK(idx.dim() == 3 && idx.size(0) == 1 && idx.size(1) == q.size(1)
                && idx.size(2) >= 1 && idx.size(2) <= 640, "canonical_sparse_batch index shape");
    TORCH_CHECK(out.sizes() == q.sizes(), "canonical_sparse_batch output shape");
    const int queries = (int)q.size(1), n = (int)kv.size(1), keys = (int)idx.size(2);
    const int parts = (keys+63)/64;
    TORCH_CHECK(partial.dim() == 1 && stats.dim() == 1
                && (parts == 1 || (partial.numel() >= (int64_t)queries*64*parts*512
                                   && stats.numel() >= (int64_t)queries*64*parts*2)),
                "canonical_sparse_batch scratch capacity");
    TORCH_CHECK(std::isfinite((float)scale), "canonical_sparse_batch nonfinite Float scale");
    const at::Tensor reads[] = {q,kv,sink,idx};
    const at::Tensor writes[] = {out,partial,stats};
    for (const auto& write : writes)
        for (const auto& read : reads)
            TORCH_CHECK(!overlaps(write,read), "canonical_sparse_batch output/scratch aliases input");
    for (int a=0;a<3;++a)
        for (int b=a+1;b<3;++b)
            TORCH_CHECK(!overlaps(writes[a],writes[b]), "canonical_sparse_batch scratch aliases output");
    using namespace ds41_canonical_sparse_batch_ns;
    // Original per-query M1 launch: heads64, parts=ceil(K/64), threads256.
    // Only grid.z/query-base offsets are new; K is neither padded nor regrouped.
    sparse_decode_queries_kernel<<<dim3(64,parts,queries),256,0,stream>>>(
        (const half*)q.data_ptr(), (const half*)kv.data_ptr(), idx.data_ptr<int>(),
        sink.data_ptr<float>(), (half*)out.data_ptr(), partial.data_ptr<float>(), stats.data_ptr<float>(),
        512,n,keys,parts,(float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (parts > 1) {
        sparse_merge_queries_kernel<<<dim3(64,queries),256,0,stream>>>(
            partial.data_ptr<float>(),stats.data_ptr<float>(),(half*)out.data_ptr(),parts,512);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

}  // namespace ds41_canonical_sparse_batch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &ds41_canonical_sparse_batch::run, "Canonical M1 sparse attention on independent query grids");
}
