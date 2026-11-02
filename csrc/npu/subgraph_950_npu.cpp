#include "subgraph_npu.h"
#include "include/pytorch_npu_helper.hpp"
#include <climits>

// Independent 950 operator. It never calls the 910B aclnn package.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
subgraph_950_npu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor nodes) {
  TORCH_CHECK(rowptr.device().type() == c10::DeviceType::PrivateUse1 &&
                  col.device() == rowptr.device() && nodes.device() == rowptr.device(),
              "subgraph_950: tensors must share an NPU device");
  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc && std::string(soc).find("Ascend950") == 0,
              "subgraph_950 requires an Ascend950 device");
  TORCH_CHECK((rowptr.scalar_type() == at::kLong || rowptr.scalar_type() == at::kInt) &&
                  col.scalar_type() == rowptr.scalar_type() && nodes.scalar_type() == rowptr.scalar_type(),
              "subgraph_950: rowptr, col and nodes must have the same int32 or int64 dtype");
  TORCH_CHECK(rowptr.dim() == 1 && col.dim() == 1 && nodes.dim() == 1 && rowptr.numel() >= 1,
              "subgraph_950: expected one-dimensional CSR inputs");
  rowptr = rowptr.contiguous(); col = col.contiguous(); nodes = nodes.contiguous();
  const auto n = rowptr.numel() - 1, e = col.numel(), q = nodes.numel();
  TORCH_CHECK(n <= INT32_MAX && e <= INT32_MAX && q <= INT32_MAX,
              "subgraph_950: shape exceeds int32 tiling range");
  if (!e || !q) {
    auto empty = at::empty({0}, col.options());
    return {at::zeros({q + 1}, rowptr.options()), empty, empty,
            at::zeros({1}, rowptr.options())};
  }
  auto out_rowptr = at::empty({q + 1}, rowptr.options());
  auto out_col = at::empty({e}, col.options());
  auto out_eid = at::empty({e}, col.options());
  auto out_count = at::empty({1}, rowptr.options());
  auto mapping = at::empty({n}, rowptr.options().dtype(at::kInt));
  auto sums = at::empty({(q + 255) / 256}, rowptr.options());
  using Launcher = void (*)(void*, void*, void*, void*, void*, void*, void*,
                           void*, void*, uint32_t, uint32_t, bool, bool, void*);
  static Launcher launch = []() {
    Dl_info info{};
    TORCH_CHECK(dladdr(reinterpret_cast<void*>(&subgraph_950_npu), &info),
                "subgraph_950: cannot locate extension");
    std::string path(info.dli_fname);
    path = path.substr(0, path.find_last_of('/')) + "/libsubgraph950.so";
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    TORCH_CHECK(handle, "subgraph_950: build csrc/npu/950/build.sh: ", dlerror());
    auto fn = reinterpret_cast<Launcher>(dlsym(handle, "launch_subgraph950"));
    TORCH_CHECK(fn, "subgraph_950: missing kernel launcher");
    return fn;
  }();
  // Resolve on the caller thread. The OpCommand worker has its own stream TLS.
  auto stream = c10_npu::getCurrentNPUStream().stream(false);
  auto handler = [=]() -> int {
    launch(rowptr.data_ptr(), col.data_ptr(), nodes.data_ptr(), mapping.data_ptr(),
           out_rowptr.data_ptr(), out_col.data_ptr(), out_eid.data_ptr(),
           out_count.data_ptr(), sums.data_ptr(), n, q,
           rowptr.scalar_type() == at::kLong, e <= 16 * n, stream);
    return 0;
  };
  at_npu::native::OpCommand cmd;
  cmd.Name("ClusterSubgraph950").SetCustomHandler(handler).Run();
  return {out_rowptr, out_col, out_eid, out_count};
}
