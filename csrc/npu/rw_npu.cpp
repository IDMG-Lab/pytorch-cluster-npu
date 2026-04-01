// rw_npu.cpp
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

std::tuple<torch::Tensor, torch::Tensor>
random_walk_npu(torch::Tensor rowptr,
                torch::Tensor col,
                torch::Tensor start,
                int64_t walk_length,
                double p,
                double q) {
  TORCH_CHECK(rowptr.device().type() == c10::DeviceType::PrivateUse1,
              "random_walk_npu: input 'rowptr' must be on NPU device");

  int64_t n_walks = start.size(0);
  // Walks tensor: shape (N_start, walk_length + 1)
  auto walks = at::empty({n_walks, walk_length + 1},
                         rowptr.options().dtype(at::kLong));
  // Edge-id tensor: shape (N_start, walk_length); -1 indicates no edge used
  auto e_ids = at::full({n_walks, walk_length},
                        /*fill_value=*/-1LL,
                        rowptr.options().dtype(at::kLong));

  EXEC_NPU_CMD(aclnnRandomWalk,
               rowptr, col, start, walk_length, p, q, walks, e_ids);

  return std::make_tuple(walks, e_ids);
}
