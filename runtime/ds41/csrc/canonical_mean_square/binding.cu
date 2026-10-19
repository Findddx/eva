#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include "kernel.cuh"

namespace ds41_mean_square {

int width_threads(int64_t width) {
    switch (width) {
    case 128:return 32;
    case 512:return 128;
    case 1280:return 256;
    case 5120:case 20480:return 512;
    default:TORCH_CHECK(false, "mean_square unsupported width");
    }
    return 0;
}

cudaStream_t validate(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "mean_square requires CUDA");
    const c10::cuda::CUDAGuard guard(x.device());
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props, x.device().index()));
    TORCH_CHECK(props.major == 12 && props.minor == 0, "mean_square requires SM120");
    TORCH_CHECK(x.scalar_type() == at::kFloat && x.dim() == 2 && x.is_contiguous()
                && x.size(0) >= 1 && x.size(0) <= 256, "mean_square needs Float[M1..256,D]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "mean_square needs actual16B input alignment");
    TORCH_CHECK(!x.is_conj() && !x.is_neg() && !x.requires_grad(), "mean_square inference values only");
    width_threads(x.size(1));
    auto stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();
    cudaStreamCaptureStatus status;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    TORCH_CHECK(status == cudaStreamCaptureStatusNone, "mean_square capture unsupported");
    return stream;
}

pybind11::dict aten_m1_config(const at::Tensor& x) {
    validate(x);
    const c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.size(0) == 1, "ATen configuration query is M1 only");
    auto out = at::empty({1, 1}, x.options());
    auto iter = at::TensorIterator::reduce_op(out, x);
    auto c = at::native::setReduceConfig<float, float, 4, 4>(iter);
    const float factor = static_cast<float>(iter.num_output_elements()) / iter.numel();
    uint32_t bits;std::memcpy(&bits, &factor, sizeof(bits));
    pybind11::dict result;
    result["block_x"] = c.block_width;result["block_y"] = c.block_height;
    result["ctas_per_output"] = c.ctas_per_output;result["vectorize_input"] = c.vectorize_input;
    result["output_vec_size"] = c.output_vec_size;result["step_input"] = c.step_input;
    result["step_output"] = c.step_output;result["input_mult_x"] = c.input_mult[0];
    result["input_mult_y"] = c.input_mult[1];result["input_mult_cta"] = c.input_mult[2];
    result["output_mult_x"] = c.output_mult[0];result["output_mult_y"] = c.output_mult[1];
    result["num_inputs"] = c.num_inputs;result["num_outputs"] = c.num_outputs;
    result["factor_bits"] = bits;
    return result;
}

at::Tensor run(const at::Tensor& x) {
    auto stream = validate(x);
    const c10::cuda::CUDAGuard guard(x.device());
    const int width = static_cast<int>(x.size(1)), rows = static_cast<int>(x.size(0));
    const float factor = 1.0f / static_cast<float>(width);  // Original host-rounded MeanOps factor.
    auto out = at::empty({rows, 1}, x.options());
    switch (width_threads(width)) {
    case 32:square_mean_rows<32><<<rows,32,0,stream>>>(x.data_ptr<float>(),out.data_ptr<float>(),width,factor);break;
    case 128:square_mean_rows<128><<<rows,128,0,stream>>>(x.data_ptr<float>(),out.data_ptr<float>(),width,factor);break;
    case 256:square_mean_rows<256><<<rows,256,0,stream>>>(x.data_ptr<float>(),out.data_ptr<float>(),width,factor);break;
    case 512:square_mean_rows<512><<<rows,512,0,stream>>>(x.data_ptr<float>(),out.data_ptr<float>(),width,factor);break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
}  // namespace ds41_mean_square

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aten_m1_config", &ds41_mean_square::aten_m1_config, "Actual installed ATen M1 configuration; no reduction launch");
    m.def("run", &ds41_mean_square::run, "Independent fixed-M1-tree square+mean rows");
}
