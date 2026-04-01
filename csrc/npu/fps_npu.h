#pragma once
#include <torch/extension.h>

// NPU dispatch for Farthest Point Sampling (FPS)
torch::Tensor fps_npu(torch::Tensor src,
                      torch::Tensor ptr,
                      torch::Tensor ratio,
                      bool random_start);
