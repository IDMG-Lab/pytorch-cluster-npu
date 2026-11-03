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
#include "include/precise_dtype.hpp"
#include "include/finite_coordinates.hpp"
#include <ATen/core/grad_mode.h>
#include <algorithm>

namespace {
torch::Tensor grid_high_precision_npu(
    const torch::Tensor& pos, const torch::Tensor& size,
    const std::optional<torch::Tensor>& optional_start,
    const std::optional<torch::Tensor>& optional_end) {
    at::NoGradGuard guard;
    TORCH_CHECK(pos.dim() == 2 && size.numel() == pos.size(1),
                "grid_npu: expected [N,F] positions and F grid sizes");
    TORCH_CHECK(size.device() == pos.device() &&
                    (!optional_start || optional_start->device() == pos.device()) &&
                    (!optional_end || optional_end->device() == pos.device()),
                "grid_npu: all parameters must share the position device");
    // These are device operations, including Double arithmetic and exact
    // int64 overflow. Avoid MinDim, whose 910B API rejects Double; amin/amax
    // use the value-only reduction supported by this CANN version.
    const auto values = torch_cluster_npu::as_double(pos);
    const auto widths = torch_cluster_npu::as_double(size).view({1, -1});
    const auto first = optional_start ? torch_cluster_npu::as_double(*optional_start)
                                      : values.amin({0});
    const auto last = optional_end ? torch_cluster_npu::as_double(*optional_end)
                                   : values.amax({0});
    TORCH_CHECK(first.numel() == pos.size(1) && last.numel() == pos.size(1),
                "grid_npu: grid bounds must have F elements");
    const auto counts = ((last - first) / widths.view({-1})).to(at::kLong) + 1;
    const auto strides = at::cat({at::ones({1}, counts.options()), counts.cumprod(0)}, 0)
                             .narrow(0, 0, pos.size(1)).view({1, -1});
    auto output = at::empty({pos.size(0)}, pos.options().dtype(at::kLong));
    // Keep scratch space bounded even for the [32M,64] document case.
    for (int64_t begin = 0; begin < pos.size(0); begin += 65536) {
        const int64_t end = std::min<int64_t>(begin + 65536, pos.size(0));
        const auto coordinates = ((values.slice(0, begin, end) - first.view({1, -1})) / widths)
                                     .to(at::kLong);
        output.slice(0, begin, end).copy_((coordinates * strides).sum(1));
    }
    return output;
}
} // namespace

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
    TORCH_CHECK(pos.dim()==2 && size.numel()==pos.size(1) &&
                    (!optional_start || optional_start->numel()==pos.size(1)) &&
                    (!optional_end || optional_end->numel()==pos.size(1)),
                "grid_npu: size/start/end must contain one value per feature");
    TORCH_CHECK(size.device()==pos.device() &&
                    (!optional_start || optional_start->device()==pos.device()) &&
                    (!optional_end || optional_end->device()==pos.device()),
                "grid_npu: all parameters must share the position device");
    torch_cluster_npu::check_finite_coordinates(pos);
    TORCH_CHECK(at::logical_and(at::isfinite(size),size>0).all().item<bool>(),
                "grid_npu: grid sizes must be positive and finite");
    if(optional_start)torch_cluster_npu::check_finite_coordinates(*optional_start);
    if(optional_end)torch_cluster_npu::check_finite_coordinates(*optional_end);

    if (pos.scalar_type() == at::kDouble || pos.scalar_type() == at::kHalf ||
        pos.scalar_type() == at::kBFloat16) {
        return grid_high_precision_npu(pos, size, optional_start, optional_end);
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
