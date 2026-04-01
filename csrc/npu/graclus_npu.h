#pragma once
#include <torch/extension.h>

// NPU dispatch for Graclus graph clustering
torch::Tensor graclus_npu(torch::Tensor rowptr,
                          torch::Tensor col,
                          std::optional<torch::Tensor> optional_weight);
