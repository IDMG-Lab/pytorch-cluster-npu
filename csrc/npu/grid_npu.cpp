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

torch::Tensor grid_npu(
    torch::Tensor pos,
    torch::Tensor size,
    std::optional<torch::Tensor> optional_start,
    std::optional<torch::Tensor> optional_end) {
    TORCH_CHECK(
        pos.device().type() == c10::DeviceType::PrivateUse1,
        "grid_npu: input 'pos' must be on NPU device");

    // 原始输入:
    // pos: [N, D] (AoS
    int64_t num_points = pos.size(0);

    auto out = at::empty(
        {num_points},
        pos.options().dtype(at::kLong));

    if (pos.numel() == 0) {
        return out;
    }

    at::Tensor start =
        optional_start.has_value()
            ? optional_start.value().contiguous()
            : std::get<0>(pos.min(0));

    at::Tensor end =
        optional_end.has_value()
            ? optional_end.value().contiguous()
            : std::get<0>(pos.max(0));

    // AoS -> SoA
    // [N,D] -> [D,N]
    auto pos_soa = pos.transpose(0, 1).contiguous();

    // Kernel输入:
    // pos_soa: [D,N]
    EXEC_NPU_CMD(aclnnVoxelGrid, pos_soa, size, start, end, out);

    return out;
}