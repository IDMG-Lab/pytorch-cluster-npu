// sampler_npu.cpp
// NPU middle-layer adapter for Neighbor Sampler.
//
// Bridges torch_cluster::neighbor_sampler to Ascend NPU via EXEC_NPU_CMD.
// Custom CANN kernel: aclnnNeighborSampler
// Prototype definition: csrc/npu/impl/sampler_npu.json
//
// Function signature (mirrors csrc/sampler.cpp):
//   torch::Tensor neighbor_sampler(torch::Tensor start,
//                                  torch::Tensor rowptr,
//                                  int64_t count,
//                                  double factor)
//
// Returns:
//   sampled_nodes: (count * N_start,) int64 — sampled neighbor node indices
//
// NPU kernel API (to be implemented):
//   aclnnNeighborSamplerGetWorkspaceSize(...)
//   aclnnNeighborSampler(...)

#include "sampler_npu.h"
#include "include/pytorch_npu_helper.hpp"

torch::Tensor neighbor_sampler_npu(torch::Tensor start,
                                   torch::Tensor rowptr,
                                   int64_t count,
                                   double factor) {
  TORCH_CHECK(start.device().type() == c10::DeviceType::PrivateUse1,
              "neighbor_sampler_npu: input 'start' must be on NPU device");

  int64_t n_start = start.size(0);
  // Output: sampled neighbor indices for each start node
  auto out = at::empty({n_start * count}, start.options().dtype(at::kLong));

  EXEC_NPU_CMD(aclnnNeighborSampler, start, rowptr, count, factor, out);

  return out;
}
