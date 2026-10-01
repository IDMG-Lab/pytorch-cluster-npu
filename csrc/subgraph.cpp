#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>
#include <vector>

#ifdef WITH_NPU
#include "npu/subgraph_npu.h"
#endif

// The two edge arrays have capacity E. The last tensor contains their
// logical length; the Python wrapper narrows them after the device finishes.
using SubgraphResult = std::tuple<torch::Tensor, torch::Tensor,
                                  torch::Tensor, torch::Tensor>;

static SubgraphResult subgraph_cpu(torch::Tensor rowptr, torch::Tensor col,
                                   torch::Tensor nodes) {
  TORCH_CHECK(rowptr.device().is_cpu() && col.device().is_cpu() &&
                  nodes.device().is_cpu(), "subgraph: tensors must share a device");
  TORCH_CHECK(rowptr.scalar_type() == at::kLong && col.scalar_type() == at::kLong &&
                  nodes.scalar_type() == at::kLong,
              "subgraph: rowptr, col and nodes must be int64");
  TORCH_CHECK(rowptr.dim() == 1 && col.dim() == 1 && nodes.dim() == 1 &&
                  rowptr.numel() >= 1, "subgraph: expected one-dimensional CSR inputs");
  rowptr = rowptr.contiguous();
  col = col.contiguous();
  nodes = nodes.contiguous();
  const auto n = rowptr.numel() - 1;
  const auto e = col.numel();
  const auto q = nodes.numel();
  const auto* rp = rowptr.data_ptr<int64_t>();
  const auto* ci = col.data_ptr<int64_t>();
  const auto* nd = nodes.data_ptr<int64_t>();
  TORCH_CHECK(rp[0] == 0 && rp[n] == e, "subgraph: invalid CSR offsets");
  std::vector<int64_t> mapping(n, -1);
  for (int64_t i = 0; i < n; ++i) {
    TORCH_CHECK(rp[i] <= rp[i + 1] && rp[i + 1] <= e,
                "subgraph: rowptr must be monotonic");
  }
  for (int64_t i = 0; i < q; ++i) {
    TORCH_CHECK(nd[i] >= 0 && nd[i] < n && mapping[nd[i]] < 0,
                "subgraph: nodes must be unique and in range");
    mapping[nd[i]] = i;
  }
  for (int64_t i = 0; i < e; ++i)
    TORCH_CHECK(ci[i] >= 0 && ci[i] < n,
                "subgraph: column index out of range");

  auto out_rowptr = at::empty({q + 1}, rowptr.options());
  auto out_col = at::empty({e}, col.options());
  auto out_eid = at::empty({e}, col.options());
  auto out_count = at::empty({1}, rowptr.options());
  auto* out_rp = out_rowptr.data_ptr<int64_t>();
  auto* out_ci = out_col.data_ptr<int64_t>();
  auto* out_ei = out_eid.data_ptr<int64_t>();
  int64_t count = 0;
  out_rp[0] = 0;
  for (int64_t i = 0; i < q; ++i) {
    for (int64_t j = rp[nd[i]]; j < rp[nd[i] + 1]; ++j) {
      const auto mapped = mapping[ci[j]];
      if (mapped >= 0) {
        out_ci[count] = mapped;
        out_ei[count] = j;
        ++count;
      }
    }
    out_rp[i + 1] = count;
  }
  out_count.data_ptr<int64_t>()[0] = count;
  return {out_rowptr, out_col, out_eid, out_count};
}

static SubgraphResult subgraph(torch::Tensor rowptr, torch::Tensor col,
                               torch::Tensor nodes) {
  if (rowptr.device().type() == c10::DeviceType::PrivateUse1) {
#ifdef WITH_NPU
    return subgraph_npu(rowptr, col, nodes);
#else
    AT_ERROR("Not compiled with NPU support");
#endif
  }
  return subgraph_cpu(rowptr, col, nodes);
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::subgraph", &subgraph);
