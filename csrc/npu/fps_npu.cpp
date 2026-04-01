// fps_npu.cpp
// NPU middle-layer adapter for Farthest Point Sampling (FPS).
//
// This file bridges the pytorch_cluster FPS operator to Ascend NPU via the
// EXEC_NPU_CMD macro.  The actual kernel is provided by the custom CANN
// operator `aclnnFarthestPointSampling` which must be developed and deployed
// separately (see csrc/npu/impl/fps_npu.json for the prototype definition).
//
// Function signature (mirrors csrc/fps.cpp):
//   torch::Tensor fps(torch::Tensor src, torch::Tensor ptr,
//                     torch::Tensor ratio, bool random_start)
//
// NPU kernel API (to be implemented):
//   aclnnFarthestPointSamplingGetWorkspaceSize(...)
//   aclnnFarthestPointSampling(...)

#include "fps_npu.h"
#include "include/pytorch_npu_helper.hpp"

torch::Tensor fps_npu(torch::Tensor src,
                      torch::Tensor ptr,
                      torch::Tensor ratio,
                      bool random_start) {
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
              "fps_npu: input 'src' must be on NPU device");

  // Compute output size: number of sampled points
  // Each batch entry i has (ptr[i+1] - ptr[i]) points;
  // ratio selects that fraction.  The output holds the sampled indices.
  int64_t n_out = static_cast<int64_t>(
      std::ceil(ratio.item<float>() * static_cast<float>(src.size(0))));
  // Allocate output tensor on NPU
  auto out = at::empty({n_out}, src.options().dtype(at::kLong));

  // Invoke the NPU kernel via EXEC_NPU_CMD.
  // The kernel is responsible for farthest-point sampling given:
  //   src          - (N, D) float tensor of input points
  //   ptr          - (B+1,) int64 batch pointers
  //   ratio        - scalar float, sampling ratio
  //   random_start - bool, whether to randomise the starting point
  //   out          - (M,) int64 output sampled indices (preallocated)
  EXEC_NPU_CMD(aclnnFarthestPointSampling,
               src, ptr, ratio, random_start, out);

  return out;
}
