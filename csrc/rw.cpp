// rw.cpp  (NPU-aware version)
#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>
#include <climits>

#include "cpu/rw_cpu.h"

#ifdef WITH_CUDA
#include "cuda/rw_cuda.h"
#endif

#ifdef WITH_NPU
#include "npu/rw_npu.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__rw_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__rw_cpu(void) { return NULL; }
#endif
#endif
#endif

#define CLUSTER_API

CLUSTER_API std::tuple<torch::Tensor, torch::Tensor>
random_walk(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start,
            int64_t walk_length, double p, double q) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    return random_walk_cuda(rowptr, col, start, walk_length, p, q);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else if (rowptr.device().type() == c10::DeviceType::PrivateUse1) {
#ifdef WITH_NPU
    return random_walk_npu(rowptr, col, start, walk_length, p, q);
#else
    AT_ERROR("Not compiled with NPU support");
#endif
  } else {
    return random_walk_cpu(rowptr, col, start, walk_length, p, q);
  }
}

// COO preprocessing belongs to the C++ host adapter. Sorting each integer
// column separately avoids row * num_nodes overflow for either index width.
static std::tuple<torch::Tensor, torch::Tensor> random_walk_coo(
    torch::Tensor row, torch::Tensor col, torch::Tensor start,
    int64_t walk_length, double p, double q, bool coalesced,
    std::optional<int64_t> optional_num_nodes) {
  TORCH_CHECK((row.scalar_type() == at::kInt || row.scalar_type() == at::kLong) &&
                  col.scalar_type() == row.scalar_type() &&
                  start.scalar_type() == row.scalar_type(),
              "random_walk: inputs must have the same int32 or int64 dtype");
  TORCH_CHECK(row.device() == col.device() && row.device() == start.device(),
              "random_walk: inputs must share a device");
  TORCH_CHECK(row.dim() == 1 && col.dim() == 1 && start.dim() == 1 &&
                  row.numel() == col.numel(), "random_walk: invalid COO shapes");
  int64_t n = optional_num_nodes.value_or(0);
  if (!optional_num_nodes.has_value()) {
    for (const auto& tensor : {row, col, start})
      if (tensor.numel()) n = std::max(n, tensor.max().item<int64_t>() + 1);
  }
  TORCH_CHECK(n >= 0 && (row.scalar_type() != at::kInt ||
                  (n <= INT32_MAX && row.numel() <= INT32_MAX)),
              "random_walk: shape exceeds index range");
  if (coalesced && row.numel()) {
    auto by_col = at::argsort(col, /*stable=*/true, /*dim=*/0, /*descending=*/false);
    row = row.index_select(0, by_col);
    col = col.index_select(0, by_col);
    auto by_row = at::argsort(row, /*stable=*/true, /*dim=*/0, /*descending=*/false);
    row = row.index_select(0, by_row);
    col = col.index_select(0, by_row);
  }
  // Bincount's integer accumulator is a host preprocessing workspace. Graph
  // indices retain their input dtype; CSR storage and the walk kernel do too.
  auto degree = at::bincount(row, {}, n);
  auto rowptr = at::zeros({n + 1}, row.options());
  rowptr.slice(0, 1).copy_(at::cumsum(degree, 0, row.scalar_type()));
  return random_walk(rowptr, col, start, walk_length, p, q);
}

static auto registry = torch::RegisterOperators()
    .op("torch_cluster::random_walk", &random_walk)
    .op("torch_cluster::random_walk_coo", &random_walk_coo);
