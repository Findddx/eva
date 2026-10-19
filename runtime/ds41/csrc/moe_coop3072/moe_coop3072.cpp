// Independent binding translation unit; all CUDA work uses its actual device stream.
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <vector>

namespace ds41_coop3072 {
void run(const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
         const std::vector<at::Tensor>& tables, const std::vector<at::Tensor>& scratch,
         int64_t k, double limit, bool check_bounds);
void prepare_routes(const at::Tensor& sel, const at::Tensor& rw,
                    const at::Tensor& runs, int64_t capacity, int64_t n_local);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &ds41_coop3072::run, "Independent K2/K3 MUL1 coop3072 M512 prototype");
    module.def("prepare_routes", &ds41_coop3072::prepare_routes, "Integer-only bounded metadata diagnostic");
}
