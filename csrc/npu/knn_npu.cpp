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

  int64_t n_y = y.size(0);
  // Output: edge_index of shape (2, k * n_y), dtype int64
  auto out = at::empty({2, k * n_y}, x.options().dtype(at::kLong));

  // Default batch pointers: single batch containing all points
  at::Tensor px = ptr_x.has_value()
                      ? ptr_x.value()
                      : at::tensor({0L, x.size(0)}, x.options().dtype(at::kLong));
  at::Tensor py = ptr_y.has_value()
                      ? ptr_y.value()
                      : at::tensor({0L, y.size(0)}, y.options().dtype(at::kLong));

  // num_workers is a CPU-side hint; pass k as int64
  (void)num_workers; // NPU uses its own thread model

  EXEC_NPU_CMD(aclnnKNNSearch, x, y, px, py, k, cosine, out);

  return out;
}
