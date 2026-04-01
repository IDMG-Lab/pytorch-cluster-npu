#pragma once
#include <torch/extension.h>

// NPU dispatch for Nearest neighbor clustering
torch::Tensor nearest_npu(torch::Tensor x,
                           torch::Tensor y,
                           torch::Tensor ptr_x,
                           torch::Tensor ptr_y);
