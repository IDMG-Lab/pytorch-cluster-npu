#include "subgraph_npu.h"
#include "include/pytorch_npu_helper.hpp"
#include <algorithm>
#include <climits>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
subgraph_npu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor nodes) {
  TORCH_CHECK(rowptr.device().type() == c10::DeviceType::PrivateUse1 &&
                  col.device() == rowptr.device() && nodes.device() == rowptr.device(),
              "subgraph: tensors must share an NPU device");
  TORCH_CHECK(rowptr.scalar_type() == at::kLong && col.scalar_type() == at::kLong &&
                  nodes.scalar_type() == at::kLong,
              "subgraph: rowptr, col and nodes must be int64");
  TORCH_CHECK(rowptr.dim() == 1 && col.dim() == 1 && nodes.dim() == 1 &&
                  rowptr.numel() >= 1,
              "subgraph: expected one-dimensional CSR inputs");
  rowptr = rowptr.contiguous();
  col = col.contiguous();
  nodes = nodes.contiguous();
  const auto n = rowptr.numel() - 1;
  const auto e = col.numel();
  const auto q = nodes.numel();
  TORCH_CHECK(n <= INT32_MAX && e <= INT32_MAX && q <= INT32_MAX,
              "subgraph: shape exceeds Ascend C tiling range");
  if (e == 0 || q == 0) {
    auto empty = at::empty({0}, col.options());
    return {at::zeros({q + 1}, rowptr.options()), empty, empty,
            at::zeros({1}, rowptr.options())};
  }
  auto out_rowptr = at::empty({q + 1}, rowptr.options());
  auto out_col = at::empty({e}, col.options());
  auto out_eid = at::empty({e}, col.options());
  auto out_count = at::empty({1}, rowptr.options());
  auto mapping = at::empty({n}, rowptr.options().dtype(at::kInt));
  const bool parallel = q >= 1024 &&
      static_cast<uint64_t>(e) * q / std::max<int64_t>(1, n) >= 16384;
  if (parallel) {
    // Stream ordering plus per-stage cache flushes publish the shared map
    // and prefix without an extra host synchronization between launches.
    for (int64_t phase = 0; phase < 4; ++phase) {
      EXEC_NPU_CMD(aclnnClusterSubgraph, rowptr, col, nodes, mapping, phase,
                   out_rowptr, out_col, out_eid, out_count);
    }
  } else {
    int64_t phase = 4;
    EXEC_NPU_CMD(aclnnClusterSubgraph, rowptr, col, nodes, mapping, phase,
                 out_rowptr, out_col, out_eid, out_count);
  }
  return {out_rowptr, out_col, out_eid, out_count};
}
