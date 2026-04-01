// nearest_npu.cpp
// NPU middle-layer adapter for Nearest neighbor point assignment.
//
// Bridges torch_cluster::nearest to Ascend NPU via EXEC_NPU_CMD.
// Custom CANN kernel: aclnnNearestNeighbor
// Prototype definition: csrc/npu/impl/nearest_npu.json
//
// Function signature (mirrors csrc/nearest.cpp):
//   torch::Tensor nearest(torch::Tensor x, torch::Tensor y,
//                         torch::Tensor ptr_x, torch::Tensor ptr_y)
//
// Returns:
//   cluster: (N_x,) int64 — for each point in x, the index of the
//            nearest cluster centre in y.
//
// NPU kernel API (to be implemented):
//   aclnnNearestNeighborGetWorkspaceSize(...)
//   aclnnNearestNeighbor(...)

#include "nearest_npu.h"
#include "include/pytorch_npu_helper.hpp"

torch::Tensor nearest_npu(torch::Tensor x,
                           torch::Tensor y,
                           torch::Tensor ptr_x,
                           torch::Tensor ptr_y) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
              "nearest_npu: input 'x' must be on NPU device");
  TORCH_CHECK(y.device().type() == c10::DeviceType::PrivateUse1,
              "nearest_npu: input 'y' must be on NPU device");

  int64_t n_x = x.size(0);
  // Output: cluster assignment for each point in x, shape (N_x,)
  auto out = at::empty({n_x}, x.options().dtype(at::kLong));

  EXEC_NPU_CMD(aclnnNearestNeighbor, x, y, ptr_x, ptr_y, out);

  return out;
}
