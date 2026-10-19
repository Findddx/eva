// Independent DS41 launcher. Vendored EXL3 kernels retain their MIT license;
// see LICENSE.exllamav3 and provenance.json. No upstream shared-library symbols.
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cmath>
#include <mutex>
#include <set>
#include <tuple>
#include <vector>

#define EXL3_MOE_COOP_DEFINE_ROT
#include "vendor/quant/exl3_moe_coop_kernel.cuh"

namespace ds41_coop512 {

static void checked(cudaError_t error) {
    TORCH_CHECK(error == cudaSuccess, "ds41_coop512 CUDA: ", cudaGetErrorString(error));
}

static void tensor(const at::Tensor& t, const at::Device& device,
                   at::ScalarType dtype, const char* label) {
    TORCH_CHECK(t.is_cuda() && t.device() == device && t.scalar_type() == dtype
                && t.is_contiguous(), "ds41_coop512: invalid ", label, " device/dtype/contiguity");
}

static void optin(void* kernel, int bytes, int device) {
    if (bytes <= 48 * 1024) return;
    static std::mutex mutex;
    static std::set<std::tuple<int, void*, int>> configured;
    std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_tuple(device, kernel, bytes);
    if (configured.count(key)) return;
    checked(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    configured.insert(key);
}

template<int K>
static void launch(MoeCoopParams p, int device, cudaStream_t stream) {
    using namespace ds41_moe_coop512_ns;
    // Deliberately no environment/autotuning/DevCtx path: the strict geometry.
    p.a_global = p.bsz > 1;
    p.ksplit_a = p.ksplit_b = 1;
    p.dbg = 0;
    constexpr int CB_MUL1 = 2;
    void* ka = (void*) exl3_moe_coop_a_kernel<K, CB_MUL1, false>;
    void* kb = (void*) exl3_moe_coop_b_kernel<K, CB_MUL1, false>;
    const int smem_a = smem_a_bytes<K>(p.Hi);
    const int smem_b = smem_b_bytes<K>();
    optin(ka, smem_a, device);
    optin(kb, smem_b, device);
    const int slots = p.bsz * p.topk;
    if (p.a_global) {
        const int items = slots * (p.Hi / 128) * 2;
        const int grid = (items + MOE_COOP_THREADS/32 - 1) / (MOE_COOP_THREADS/32);
        exl3_moe_coop_rot_kernel<<<grid, MOE_COOP_THREADS, 0, stream>>>(p);
        checked(cudaPeekAtLastError());
    }
    void* arguments[] = { (void*) &p };
    checked(cudaLaunchKernel(ka, dim3(slots * 2 * (p.I / MOE_COOP_COLS)),
                             dim3(MOE_COOP_THREADS), arguments, smem_a, stream));
    checked(cudaLaunchKernel(kb, dim3(slots * (p.Ho / MOE_COOP_COLS)),
                             dim3(MOE_COOP_THREADS), arguments, smem_b, stream));
    checked(cudaPeekAtLastError());
}

void run(const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
         const std::vector<at::Tensor>& tables, const std::vector<at::Tensor>& scratch,
         int64_t k, double limit, bool check_bounds) {
    TORCH_CHECK(x.is_cuda(), "ds41_coop512: CUDA input required");
    const c10::cuda::CUDAGuard guard(x.device());
    const auto device = x.device();
    cudaDeviceProp prop;
    checked(cudaGetDeviceProperties(&prop, device.index()));
    TORCH_CHECK(prop.major == 12 && prop.minor == 0, "ds41_coop512 prototype supports SM120 only");
    auto stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    cudaStreamCaptureStatus capture;
    checked(cudaStreamIsCapturing(stream, &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "ds41_coop512: graph capture is not supported");
    tensor(x, device, at::kHalf, "x");
    tensor(sel, device, at::kLong, "sel");
    tensor(rw, device, at::kHalf, "rw");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 5120 && x.size(0) >= 1 && x.size(0) <= 64,
                "ds41_coop512: x must be Half[1..64,5120]");
    const int rows = (int)x.size(0);
    TORCH_CHECK(sel.dim() == 2 && sel.size(0) == rows && sel.size(1) == 6 && rw.sizes() == sel.sizes(),
                "ds41_coop512: sel/rw must be [rows,6]");
    TORCH_CHECK((k == 2 || k == 3) && std::isfinite((float)limit) && limit >= 0,
                "ds41_coop512: only K2/K3 MUL1 gated SiLU with finite nonnegative limit");
    TORCH_CHECK(tables.size() == 9 && scratch.size() == 8, "ds41_coop512: nine tables/eight scratch tensors required");
    int n_local = -1;
    for (const auto& t : tables) {
        tensor(t, device, at::kLong, "pointer table");
        TORCH_CHECK(t.dim() == 1 && t.numel() >= 1 && t.numel() <= 384, "ds41_coop512: pointer table length");
        if (n_local < 0) n_local = (int)t.numel();
        TORCH_CHECK(t.numel() == n_local, "ds41_coop512: table lengths differ");
    }
    const int capacity = (int)scratch[7].size(0);
    TORCH_CHECK(capacity >= rows && capacity <= 64, "ds41_coop512: output capacity exceeded");
    const int slots = capacity * 6;
    const int widths[] = {5120,5120,2304,2304,2304,5120};
    for (int i = 0; i < 6; ++i) {
        tensor(scratch[i], device, i == 5 ? at::kFloat : at::kHalf, "scratch");
        TORCH_CHECK(scratch[i].dim() == 2 && scratch[i].size(0) == slots && scratch[i].size(1) == widths[i],
                    "ds41_coop512: scratch shape");
    }
    tensor(scratch[6], device, at::kInt, "counter");
    tensor(scratch[7], device, at::kFloat, "output");
    TORCH_CHECK(scratch[7].dim() == 2 && scratch[7].size(1) == 5120 && scratch[6].dim() == 1
                && scratch[6].numel() == exl3_moe_coop_ctr_len(slots, capacity, 2304, 5120),
                "ds41_coop512: output/counter shape");
    // Prototype default: two small reductions + host synchronization. The caller
    // may disable only after its own router/lease validation; never skip routing.
    if (check_bounds) {
        TORCH_CHECK(sel.min().item<int64_t>() >= 0 && sel.max().item<int64_t>() < n_local,
                    "ds41_coop512: expert index outside pointer tables");
    }
    MoeCoopParams p = {};
    p.x = (const half*)x.data_ptr(); p.x_stride = 5120;
    p.sel = (const int64_t*)sel.data_ptr(); p.rw = (const half*)rw.data_ptr();
    p.bsz = rows; p.topk = 6; p.H = p.Hi = p.Ho = p.H_out = 5120; p.I = 2304;
    p.min_expert = -1; p.max_expert = 0; p.n_local = n_local;
    p.g_trellis = tables[0].data_ptr<int64_t>(); p.g_suh = tables[1].data_ptr<int64_t>(); p.g_svh = tables[2].data_ptr<int64_t>();
    p.u_trellis = tables[3].data_ptr<int64_t>(); p.u_suh = tables[4].data_ptr<int64_t>(); p.u_svh = tables[5].data_ptr<int64_t>();
    p.d_trellis = tables[6].data_ptr<int64_t>(); p.d_suh = tables[7].data_ptr<int64_t>(); p.d_svh = tables[8].data_ptr<int64_t>();
    p.gated = true; p.act = MOE_COOP_ACT_SILU; p.act_limit = (float)limit; p.gu_f32 = false;
    p.had_g = (half*)scratch[0].data_ptr(); p.had_u = (half*)scratch[1].data_ptr();
    p.gu_g = scratch[2].data_ptr(); p.gu_u = scratch[3].data_ptr();
    p.act_out = (half*)scratch[4].data_ptr(); p.d_out = scratch[5].data_ptr<float>();
    p.slots_max = slots; p.rows_max = capacity;
    p.ctr_a_len = slots * (2304/128); p.ctr_b_len = capacity * (5120/128);
    p.ctr_a = scratch[6].data_ptr<int>(); p.ctr_b = p.ctr_a + p.ctr_a_len;
    p.runs = p.ctr_b + p.ctr_b_len;
    p.out = scratch[7].data_ptr<float>(); p.out_stride = 5120;
    checked(cudaMemsetAsync(p.ctr_a, 0, scratch[6].numel()*sizeof(int), stream));
    if (k == 2) launch<2>(p, device.index(), stream);
    else launch<3>(p, device.index(), stream);
}

}  // namespace ds41_coop512

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &ds41_coop512::run, "Independent K2/K3 MUL1 coop512 prototype");
}
