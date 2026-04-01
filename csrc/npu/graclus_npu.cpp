// graclus_npu.cpp
// NPU middle-layer adapter for Graclus graph clustering.
//
// Bridges torch_cluster::graclus to Ascend NPU via EXEC_NPU_CMD.
// Custom CANN kernel: aclnnGraclus
// Prototype definition: csrc/npu/impl/graclus_npu.json
//
// Function signature (mirrors csrc/graclus.cpp):
//   torch::Tensor graclus(torch::Tensor rowptr, torch::Tensor col,
//                         std::optional<torch::Tensor> optional_weight)
//
// NPU kernel API (to be implemented):
//   aclnnGraclusGetWorkspaceSize(...)
//   aclnnGraclus(...)

#include "graclus_npu.h"
#include "include/pytorch_npu_helper.hpp"

torch::Tensor graclus_npu(torch::Tensor rowptr,
                           torch::Tensor col,
                           std::optional<torch::Tensor> optional_weight) {
  TORCH_CHECK(rowptr.device().type() == c10::DeviceType::PrivateUse1,
              "graclus_npu: input 'rowptr' must be on NPU device");

  int64_t num_nodes = rowptr.size(0) - 1;
  // Output: cluster assignment for each node, shape (N,)
  auto out = at::empty({num_nodes}, rowptr.options());

  // Use a zero tensor as weight placeholder when weight is not provided
  at::Tensor weight = optional_weight.has_value()
                          ? optional_weight.value()
                          : at::zeros({col.size(0)}, col.options().dtype(at::kFloat));

  EXEC_NPU_CMD(aclnnGraclus, rowptr, col, weight, out);

  return out;
}
