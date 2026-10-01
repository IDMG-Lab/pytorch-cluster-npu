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
#include <climits>

torch::Tensor graclus_npu(torch::Tensor rowptr,
                           torch::Tensor col,
                           std::optional<torch::Tensor> optional_weight) {
  TORCH_CHECK(rowptr.device().type() == c10::DeviceType::PrivateUse1,
              "graclus_npu: input 'rowptr' must be on NPU device");
  TORCH_CHECK(col.device() == rowptr.device() && rowptr.scalar_type() == at::kLong &&
                  col.scalar_type() == at::kLong && rowptr.dim() == 1 &&
                  col.dim() == 1 && rowptr.numel() >= 1,
              "graclus_npu: expected int64 CSR inputs on one device");
  const int64_t num_nodes = rowptr.size(0) - 1;
  TORCH_CHECK(num_nodes <= INT32_MAX && col.numel() <= INT32_MAX,
              "graclus_npu: shape exceeds tiling range");
  auto out = at::empty({num_nodes}, rowptr.options());
  if (num_nodes == 0) return out;
  const bool weighted = optional_weight.has_value();
  at::Tensor weight = weighted
      ? optional_weight.value().to(at::kFloat).contiguous()
      : at::empty({col.numel()}, rowptr.options().dtype(at::kFloat));
  TORCH_CHECK(!weighted || (optional_weight.value().device() == rowptr.device() &&
                            optional_weight.value().numel() == col.numel()),
              "graclus_npu: weight must match col on the same device");
  // The CPU baseline draws its node permutation from the CPU generator.
  // Drawing it there also lets seeded CPU/NPU calls compare labels exactly.
  auto permutation_cpu = at::randperm(
      num_nodes, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  auto permutation = permutation_cpu.to(rowptr.device());
  auto rp = rowptr.contiguous();
  auto ci = col.contiguous();
  EXEC_NPU_CMD(aclnnGraclus, rp, ci, weight, permutation, weighted, out);
  return out;
}
