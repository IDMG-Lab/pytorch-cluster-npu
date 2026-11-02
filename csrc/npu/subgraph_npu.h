#pragma once

#include <torch/torch.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
subgraph_npu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor nodes);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
subgraph_950_npu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor nodes);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
subgraph_910b_npu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor nodes);
