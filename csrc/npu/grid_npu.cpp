// grid_npu.cpp
// NPU middle-layer adapter for Grid (VoxelGrid) clustering.
//
// Bridges torch_cluster::grid to Ascend NPU via EXEC_NPU_CMD.
// Custom CANN kernel: aclnnVoxelGrid
// Prototype definition: csrc/npu/impl/grid_npu.json
//
// Function signature (mirrors csrc/grid.cpp):
//   torch::Tensor grid(torch::Tensor pos, torch::Tensor size,
//                      std::optional<torch::Tensor> optional_start,
//                      std::optional<torch::Tensor> optional_end)
//
// NPU kernel API (to be implemented):
//   aclnnVoxelGridGetWorkspaceSize(...)
//   aclnnVoxelGrid(...)

#include "grid_npu.h"
#include "include/pytorch_npu_helper.hpp"

torch::Tensor grid_npu(torch::Tensor pos,
                       torch::Tensor size,
                       std::optional<torch::Tensor> optional_start,
                       std::optional<torch::Tensor> optional_end) {
  TORCH_CHECK(pos.device().type() == c10::DeviceType::PrivateUse1,
              "grid_npu: input 'pos' must be on NPU device");

  int64_t num_points = pos.size(0);
  // Output: cluster (voxel) index for each point, shape (N,)
  auto out = at::empty({num_points}, pos.options().dtype(at::kLong));

  // Use pos min/max as default bounds when not provided
  at::Tensor start = optional_start.has_value()
                         ? optional_start.value()
                         : pos.min(0).values;
  at::Tensor end = optional_end.has_value()
                       ? optional_end.value()
                       : pos.max(0).values;

  EXEC_NPU_CMD(aclnnVoxelGrid, pos, size, start, end, out);

  return out;
}
