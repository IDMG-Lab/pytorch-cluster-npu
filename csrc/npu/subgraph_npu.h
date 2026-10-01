#pragma once

#include <torch/torch.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
subgraph_npu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor nodes);
