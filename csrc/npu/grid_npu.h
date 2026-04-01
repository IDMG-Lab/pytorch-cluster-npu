#pragma once
#include <torch/extension.h>

// NPU dispatch for Grid (VoxelGrid) clustering
torch::Tensor grid_npu(torch::Tensor pos,
                       torch::Tensor size,
                       std::optional<torch::Tensor> optional_start,
                       std::optional<torch::Tensor> optional_end);
