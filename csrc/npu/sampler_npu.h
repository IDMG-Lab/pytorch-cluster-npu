#pragma once
#include <torch/extension.h>

// NPU dispatch for Neighbor Sampler
torch::Tensor neighbor_sampler_npu(torch::Tensor start,
                                   torch::Tensor rowptr,
                                   int64_t count,
                                   double factor);
