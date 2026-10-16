// knn_npu.cpp
// NPU middle-layer adapter for K-Nearest Neighbor (KNN) search.
//
// Bridges torch_cluster::knn to Ascend NPU via EXEC_NPU_CMD.
// Custom CANN kernel: aclnnKNNSearch
// Prototype definition: csrc/npu/impl/knn_npu.json
//
// Function signature (mirrors csrc/knn.cpp):
//   torch::Tensor knn(torch::Tensor x, torch::Tensor y,
//                     std::optional<torch::Tensor> ptr_x,
//                     std::optional<torch::Tensor> ptr_y,
//                     int64_t k, bool cosine, int64_t num_workers)
//
// Returns:
//   edge_index: (2, k * N_y) — pairs [source_idx, target_idx]
//
// NPU kernel API (to be implemented):
//   aclnnKNNSearchGetWorkspaceSize(...)
//   aclnnKNNSearch(...)

#include "knn_npu.h"
#include "include/pytorch_npu_helper.hpp"
#include <algorithm>
#include <climits>
#include <vector>

static torch::Tensor knn_native_npu(torch::Tensor x,
                      torch::Tensor y,
                      std::optional<torch::Tensor> ptr_x,
                      std::optional<torch::Tensor> ptr_y,
                      int64_t k,
                      bool cosine,
                      int64_t num_workers) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
              "knn_npu: input 'x' must be on NPU device");
  TORCH_CHECK(y.device().type() == c10::DeviceType::PrivateUse1,
              "knn_npu: input 'y' must be on NPU device");
  TORCH_CHECK(x.device() == y.device(), "knn_npu: x/y device mismatch");
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && x.size(1) == y.size(1),
              "knn_npu: x/y must be rank-2 with the same feature width");
  TORCH_CHECK(k > 0 && k <= 100, "knn_npu: k must be in [1,100]");
  TORCH_CHECK(x.size(0) <= INT32_MAX && y.size(0) <= INT32_MAX &&
                  x.size(1) <= INT32_MAX,
              "knn_npu: dimensions exceed tiling range");

  auto x32 = x.to(at::kFloat).contiguous();
  auto y32 = y.to(at::kFloat).contiguous();
  if (cosine) {
    auto x_norm = (x32 * x32).sum(1, true).sqrt();
    auto y_norm = (y32 * y32).sum(1, true).sqrt();
    TORCH_CHECK(!x_norm.eq(0).any().item<bool>() &&
                    !y_norm.eq(0).any().item<bool>(),
                "knn_npu: cosine distance is undefined for zero vectors");
    x32 = (x32 / x_norm).contiguous();
    y32 = (y32 / y_norm).contiguous();
  }
  const int64_t n_y = y.size(0);
  auto out = at::empty({2, k * n_y}, x.options().dtype(at::kLong));

  // Default batch pointers: single batch containing all points
  at::Tensor px = ptr_x.has_value()
                      ? ptr_x.value().to(at::kLong).contiguous()
                      : at::tensor({0L, x.size(0)}, x.options().dtype(at::kLong));
  at::Tensor py = ptr_y.has_value()
                      ? ptr_y.value().to(at::kLong).contiguous()
                      : at::tensor({0L, y.size(0)}, y.options().dtype(at::kLong));
  TORCH_CHECK(px.device() == x.device() && py.device() == x.device() &&
                  px.numel() == py.numel() && px.numel() >= 2,
              "knn_npu: invalid batch pointers");

  // num_workers is a CPU-side hint; pass k as int64
  (void)num_workers; // NPU uses its own thread model

  EXEC_NPU_CMD(aclnnKNNSearch, x32, y32, px, py, k, cosine, out);

  return out;
}

// The existing Python tiled Cube path now belongs to the registered operator.
// Both the Python API and torch.ops.torch_cluster.knn use this same path.
// Unsafe shortlist rows call the native kernel directly, avoiding recursion.
static torch::Tensor knn_tiled_npu(torch::Tensor x, torch::Tensor y, int64_t k) {
  const int64_t tile = std::max<int64_t>(64, std::min<int64_t>(1024, 8388608 / x.size(0)));
  const int64_t shortlist = std::min<int64_t>(x.size(0), k + 8);
  auto origin = x.slice(0, 0, 1);
  auto reference_points = x - origin;
  auto query_points = y - origin;
  auto reference_norm = (reference_points * reference_points).sum(1);
  auto query_norm = (query_points * query_points).sum(1);
  auto transposed = reference_points.t().contiguous();
  std::vector<torch::Tensor> references, boundaries;
  for (int64_t first = 0; first < y.size(0); first += tile) {
    const int64_t end = std::min<int64_t>(first + tile, y.size(0));
    auto distances = at::mm(query_points.slice(0, first, end), transposed) * -2;
    distances = distances + reference_norm.unsqueeze(0);
    distances = distances + query_norm.slice(0, first, end).unsqueeze(1);
    auto top = distances.topk(shortlist, 1, false, true);
    references.push_back(std::get<1>(top));
    boundaries.push_back(std::get<0>(top).select(1, shortlist - 1));
  }
  auto candidates = references.size() == 1 ? references[0] : at::cat(references, 0);
  candidates = std::get<0>(candidates.to(at::kFloat).sort(1)).to(at::kLong);
  auto neighbors = x.index_select(0, candidates.reshape({-1})).view(
      {y.size(0), shortlist, x.size(1)});
  neighbors.sub_(y.unsqueeze(1)).square_();
  auto exact = neighbors.sum(2);
  auto order = at::argsort(exact, true, 1, false);
  auto selected = candidates.gather(1, order.slice(1, 0, k));
  if (shortlist < x.size(0)) {
    auto boundary = boundaries.size() == 1 ? boundaries[0] : at::cat(boundaries, 0);
    auto error = (4 * x.size(1) * 1.1920928955078125e-7) *
        (1 + reference_norm.max() + query_norm);
    auto kth = exact.gather(1, order.slice(1, k - 1, k)).reshape({-1});
    auto unsafe = at::logical_or(at::logical_not(at::isfinite(boundary)),
                                 kth >= boundary - error);
    auto unsafe_ids = unsafe.nonzero().reshape({-1});
    if (unsafe_ids.numel()) {
      auto fallback = knn_native_npu(x, y.index_select(0, unsafe_ids).contiguous(),
                                    std::nullopt, std::nullopt, k, false, 1);
      selected.index_copy_(0, unsafe_ids, fallback.select(0, 1).view({-1, k}));
    }
  }
  auto output = at::empty({2, y.size(0) * k}, y.options().dtype(at::kLong));
  output.select(0, 0).view({y.size(0), k}).copy_(
      at::arange(y.size(0), y.options().dtype(at::kLong)).view({y.size(0), 1}));
  output.select(0, 1).copy_(selected.reshape({-1}));
  return output;
}

torch::Tensor knn_npu(torch::Tensor x, torch::Tensor y,
                      std::optional<torch::Tensor> ptr_x,
                      std::optional<torch::Tensor> ptr_y,
                      int64_t k, bool cosine, int64_t num_workers) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 &&
                  y.device() == x.device(), "knn_npu: x/y must share an NPU device");
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && x.size(1) == y.size(1),
              "knn_npu: x/y must be rank-2 with the same feature width");
  TORCH_CHECK(k > 0 && k <= 100, "knn_npu: k must be in [1,100]");
  TORCH_CHECK(x.size(0) <= INT32_MAX && y.size(0) <= INT32_MAX && x.size(1) <= INT32_MAX,
              "knn_npu: dimensions exceed tiling range");
  if (!ptr_x.has_value() && !ptr_y.has_value() && !cosine &&
      x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat &&
      k <= x.size(0) && x.size(0) <= 16777216 && y.size(0) > 0 &&
      static_cast<double>(x.size(0)) * y.size(0) * x.size(1) >= 1048576) {
    return knn_tiled_npu(x, y, k);
  }
  // Segment the existing exact-checked Cube path instead of sending batched
  // float32 queries through the scalar all-pairs kernel. Empty/short or
  // malformed batches retain the native implementation and its contracts.
  if (ptr_x.has_value() && ptr_y.has_value() && !cosine &&
      x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat &&
      x.size(0) <= 16777216 && y.size(0) > 0 &&
      ptr_x->device() == x.device() && ptr_y->device() == x.device() &&
      ptr_x->scalar_type() == at::kLong && ptr_y->scalar_type() == at::kLong &&
      ptr_x->dim() == 1 && ptr_y->dim() == 1 &&
      ptr_x->numel() == ptr_y->numel() && ptr_x->numel() >= 2) {
    auto px = ptr_x->cpu().contiguous();
    auto py = ptr_y->cpu().contiguous();
    const auto* ax = px.data_ptr<int64_t>();
    const auto* ay = py.data_ptr<int64_t>();
    const int64_t batches = px.numel() - 1;
    bool eligible = ax[0] == 0 && ay[0] == 0 &&
        ax[batches] == x.size(0) && ay[batches] == y.size(0);
    for (int64_t b = 0; b < batches && eligible; ++b) {
      eligible = ax[b] >= 0 && ax[b+1] >= ax[b] && ax[b+1] <= x.size(0) &&
          ay[b] >= 0 && ay[b+1] >= ay[b] && ay[b+1] <= y.size(0) &&
          ax[b+1] - ax[b] >= k;
    }
    if (eligible) {
      std::vector<torch::Tensor> chunks;
      for (int64_t b = 0; b < batches; ++b) {
        if (ay[b+1] == ay[b]) continue;
        auto result = knn_tiled_npu(x.slice(0, ax[b], ax[b+1]),
                                    y.slice(0, ay[b], ay[b+1]), k);
        result.select(0, 0).add_(ay[b]);
        result.select(0, 1).add_(ax[b]);
        chunks.push_back(result);
      }
      if (chunks.empty()) return at::empty({2, 0}, y.options().dtype(at::kLong));
      return chunks.size() == 1 ? chunks[0] : at::cat(chunks, 1);
    }
  }
  return knn_native_npu(x, y, ptr_x, ptr_y, k, cosine, num_workers);
}
