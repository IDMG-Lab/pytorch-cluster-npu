// radius_npu.cpp
// NPU middle-layer adapter for Radius-based neighbor search.
//
// Bridges torch_cluster::radius to Ascend NPU via EXEC_NPU_CMD.
// Custom CANN kernel: aclnnRadiusSearch
// Prototype definition: csrc/npu/impl/radius_npu.json
//
// Function signature (mirrors csrc/radius.cpp):
//   torch::Tensor radius(torch::Tensor x, torch::Tensor y,
//                        std::optional<torch::Tensor> ptr_x,
//                        std::optional<torch::Tensor> ptr_y,
//                        double r, int64_t max_num_neighbors,
//                        int64_t num_workers, bool ignore_same_index)
//
// Returns:
//   edge_index: (2, E) — variable-length edge list of found neighbors.
//
// NPU kernel API (to be implemented):
//   aclnnRadiusSearchGetWorkspaceSize(...)
//   aclnnRadiusSearch(...)

#include "radius_npu.h"
#include "include/pytorch_npu_helper.hpp"

torch::Tensor radius_npu(torch::Tensor x,
                          torch::Tensor y,
                          std::optional<torch::Tensor> ptr_x,
                          std::optional<torch::Tensor> ptr_y,
                          double r,
                          int64_t max_num_neighbors,
                          int64_t num_workers,
                          bool ignore_same_index) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
              "radius_npu: input 'x' must be on NPU device");
  TORCH_CHECK(y.device().type() == c10::DeviceType::PrivateUse1,
              "radius_npu: input 'y' must be on NPU device");

  int64_t n_y = y.size(0);
  // Pre-allocate worst-case output buffer: every y point has max_num_neighbors
  // neighbors.  The actual number of edges is returned inside out[0][0] by
  // convention (kernel fills valid entries and sets the count in a sidecar).
  auto out = at::full({2, n_y * max_num_neighbors},
                      /*fill_value=*/-1,
                      x.options().dtype(at::kLong));

  at::Tensor px = ptr_x.has_value()
                      ? ptr_x.value()
                      : at::tensor({0L, x.size(0)}, x.options().dtype(at::kLong));
  at::Tensor py = ptr_y.has_value()
                      ? ptr_y.value()
                      : at::tensor({0L, y.size(0)}, y.options().dtype(at::kLong));

  (void)num_workers; // NPU does not use CPU workers

  EXEC_NPU_CMD(aclnnRadiusSearch,
               x, y, px, py, r, max_num_neighbors, ignore_same_index, out);

  return out;
}
