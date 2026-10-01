// rw_npu.cpp (Ascend C)
// NPU middle-layer adapter for Random Walk sampling.
//
// Bridges torch_cluster::random_walk to Ascend NPU via EXEC_NPU_CMD.
// Custom CANN kernel: aclnnRandomWalk  (node2vec-style biased random walk)
// Prototype definition: csrc/npu/impl/rw_npu.json
//
// Function signature (mirrors csrc/rw.cpp):
//   std::tuple<torch::Tensor, torch::Tensor>
//   random_walk(torch::Tensor rowptr, torch::Tensor col,
//               torch::Tensor start, int64_t walk_length,
//               double p, double q)
//
// Returns:
//   walks:  (N_start, walk_length + 1) int64 — node indices along each walk
//   e_ids:  (N_start, walk_length)    int64 — edge indices traversed
//
// NPU kernel API (to be implemented):
//   aclnnRandomWalkGetWorkspaceSize(...)
//   aclnnRandomWalk(...)

#include "rw_npu.h"
#include "include/pytorch_npu_helper.hpp"
#include <climits>
#include <cmath>

std::tuple<torch::Tensor, torch::Tensor>
random_walk_npu(torch::Tensor rowptr,
                torch::Tensor col,
                torch::Tensor start,
                int64_t walk_length,
                double p,
                double q) {
  TORCH_CHECK(rowptr.device().type() == c10::DeviceType::PrivateUse1 &&
                  col.device() == rowptr.device() && start.device() == rowptr.device(),
              "random_walk_npu: inputs must share an NPU device");
  TORCH_CHECK(rowptr.scalar_type() == at::kLong && col.scalar_type() == at::kLong &&
                  start.scalar_type() == at::kLong,
              "random_walk_npu: CSR inputs must be int64");
  TORCH_CHECK(rowptr.dim() == 1 && col.dim() == 1 && start.dim() == 1 &&
                  rowptr.numel() >= 1, "random_walk_npu: expected 1D CSR inputs");
  TORCH_CHECK(walk_length >= 0 && walk_length <= INT32_MAX,
              "random_walk_npu: invalid walk length");
  TORCH_CHECK(std::isfinite(p) && std::isfinite(q) && p > 0.0 && q > 0.0,
              "random_walk_npu: p and q must be positive finite numbers");
  const int64_t count = start.numel();
  if (count == 0)
    return {at::empty({0, walk_length + 1}, start.options()),
            at::empty({0, walk_length}, start.options())};
  if (walk_length == 0)
    return {start.contiguous().view({count, 1}),
            at::empty({count, 0}, start.options())};
  const int64_t nodes_pitch = ((walk_length + 1 + 7) / 8) * 8;
  const int64_t edges_pitch = ((walk_length + 7) / 8) * 8;
  auto walks = at::empty({count, nodes_pitch}, start.options());
  auto e_ids = at::empty({count, edges_pitch}, start.options());
  auto random = at::rand({count, walk_length},
                         start.options().dtype(at::kFloat));
  auto rp = rowptr.contiguous();
  auto ci = col.contiguous();
  auto starts = start.contiguous();
  auto flags = at::empty({640}, start.options().dtype(at::kInt));
  const int64_t num_nodes = rp.numel() - 1;
  const int64_t steps = count * walk_length;
  // Amortize the complete CSR check across enough transition work. A shape
  // alone never enables the specialization: every row pointer is checked.
  const bool check_degree_one = ci.numel() == num_nodes && steps >= 32768 &&
                                num_nodes <= steps / 2;
  if (check_degree_one) {
    int64_t phase = 0;
    EXEC_NPU_CMD(aclnnRandomWalk, rp, ci, starts, random,
                 walk_length, p, q, phase, walks, e_ids, flags);
  }
  int64_t phase = check_degree_one ? 2 : 1;
  EXEC_NPU_CMD(aclnnRandomWalk, rp, ci, starts, random,
               walk_length, p, q, phase,
               walks, e_ids, flags);
  return {walks.narrow(1, 0, walk_length + 1),
          e_ids.narrow(1, 0, walk_length)};
}
