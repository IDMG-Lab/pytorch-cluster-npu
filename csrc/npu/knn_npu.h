#pragma once
#include <torch/extension.h>

// NPU dispatch for K-Nearest Neighbor search
torch::Tensor knn_npu(torch::Tensor x,
                      torch::Tensor y,
                      std::optional<torch::Tensor> ptr_x,
                      std::optional<torch::Tensor> ptr_y,
                      int64_t k,
                      bool cosine,
                      int64_t num_workers);
