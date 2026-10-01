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
#include <climits>

torch::Tensor knn_npu(torch::Tensor x,
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
