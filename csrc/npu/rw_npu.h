#pragma once
#include <torch/extension.h>
#include <tuple>

// NPU dispatch for Random Walk sampling
std::tuple<torch::Tensor, torch::Tensor>
random_walk_npu(torch::Tensor rowptr,
                torch::Tensor col,
                torch::Tensor start,
                int64_t walk_length,
                double p,
                double q);
