#pragma once
#include <torch/extension.h>

// NPU dispatch for Radius-based neighbor search
torch::Tensor radius_npu(torch::Tensor x,
                          torch::Tensor y,
                          std::optional<torch::Tensor> ptr_x,
                          std::optional<torch::Tensor> ptr_y,
                          double r,
                          int64_t max_num_neighbors,
                          int64_t num_workers,
                          bool ignore_same_index);
