// radius_npu.cpp
// NPU middle-layer adapter for Radius-based neighbor search.
//
// Bridges torch_cluster::radius to Ascend NPU via EXEC_NPU_CMD.
// Custom CANN kernel: aclnnRadius
// Kernel project:       csrc/npu/impl/op_project
//
// 对外接口（torch.ops.torch_cluster.radius）保持不变；本文件内部额外构造 hash-grid
// 所需的辅助张量并随 aclnnRadius 一起下发（host 建表，kernel 查询）。
//
// Returns edge_index (2, E) int64；row0 = y-index，row1 = x-index（与 CPU/CUDA 一致）。

#include "radius_npu.h"
#include "include/pytorch_npu_helper.hpp"

#include <ATen/ATen.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int64_t GRID_MAX_DIM = 4;      // 仅 F<=4 启用 hash-grid
constexpr int64_t GRID_MAX_CELLS = 1 << 20;

} // namespace

torch::Tensor radius_npu(torch::Tensor x,
                         torch::Tensor y,
                         std::optional<torch::Tensor> ptr_x,
                         std::optional<torch::Tensor> ptr_y,
                         double r,
                         int64_t max_num_neighbors,
                         int64_t num_workers,
                         bool ignore_same_index) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
              "radius_npu: input 'x' must be on NPU device");
  TORCH_CHECK(y.device().type() == c10::DeviceType::PrivateUse1,
              "radius_npu: input 'y' must be on NPU device");
  TORCH_CHECK(x.dim() == 2, "radius_npu: 'x' must be 2D, but got ", x.dim());
  TORCH_CHECK(y.dim() == 2, "radius_npu: 'y' must be 2D, but got ", y.dim());
  TORCH_CHECK(x.size(1) == y.size(1),
              "radius_npu: 'x' and 'y' must have the same feature dimension");
  (void)num_workers;

  auto x_c = x.contiguous();
  auto y_c = y.contiguous();

  at::Tensor px;
  at::Tensor py;
  if (ptr_x.has_value() && ptr_x->defined()) {
    px = ptr_x->to(at::kInt).contiguous();
  }
  if (ptr_y.has_value() && ptr_y->defined()) {
    py = ptr_y->to(at::kInt).contiguous();
  }

  const int64_t N = x_c.size(0);
  const int64_t M = y_c.size(0);
  const int64_t F = x_c.size(1);

  // 内核使用 SoA 布局 [F, N] / [F, M]
  auto x_soa = x_c.transpose(0, 1).contiguous();
  auto y_soa = y_c.transpose(0, 1).contiguous();

  const auto dev = x_c.device();
  const auto opt_i32 = x_c.options().dtype(at::kInt);
  const auto opt_f32 = x_c.options().dtype(at::kFloat);
  const auto opt_i64 = x_c.options().dtype(at::kLong);
  const float r_f = static_cast<float>(r);

  // ---- 段边界 ----
  std::vector<int64_t> seg;
  if (px.defined()) {
    auto p = px.to(at::kCPU).to(at::kLong).contiguous();
    const int64_t* pd = p.data_ptr<int64_t>();
    for (int64_t i = 0; i <= p.numel() - 1; ++i) {
      seg.push_back(pd[i]);
    }
  } else {
    seg.push_back(0);
    seg.push_back(N);
  }
  const int64_t B = static_cast<int64_t>(seg.size()) - 1;

  bool use_grid = (N > 0) && (M > 0) && (F >= 1) && (F <= GRID_MAX_DIM) && (r > 0.0);

  at::Tensor sorted_x, order_t, cell_start, cell_start_off, grid_min, grid_g, use_grid_t;

  if (use_grid) {
    std::vector<at::Tensor> sorted_segs, order_segs, cs_segs;
    std::vector<int64_t> cs_off;
    cs_off.push_back(0);
    std::vector<float> min_vec;
    std::vector<int32_t> g_vec;
    int64_t cs_total = 0;

    for (int64_t q = 0; q < B && use_grid; ++q) {
      const int64_t xs = seg[q];
      const int64_t xe = seg[q + 1];
      const int64_t n = xe - xs;
      if (n <= 0) {
        cs_segs.push_back(at::zeros({1}, opt_i32));
        cs_total += 1;
        cs_off.push_back(cs_total);
        order_segs.push_back(at::empty({0}, opt_i32));
        sorted_segs.push_back(at::empty({F, 0}, x_c.options()));
        for (int64_t d = 0; d < F; ++d) {
          min_vec.push_back(0.0f);
          g_vec.push_back(1);
        }
        continue;
      }

      auto seg_soa = x_soa.slice(1, xs, xe).contiguous();   // [F, n]
      auto mn = seg_soa.amin(1);                            // [F]
      auto mx = seg_soa.amax(1);                            // [F]
      auto gd = at::clamp_min(at::floor((mx - mn) / r_f).to(at::kLong) + 1, 1);

      auto gd_cpu = gd.to(at::kCPU).to(at::kLong);
      auto mn_cpu = mn.to(at::kCPU).to(at::kFloat);

      int64_t G = 1;
      bool overflow = false;
      for (int64_t d = 0; d < F; ++d) {
        int64_t g = gd_cpu[d].item<int64_t>();
        if (g <= 0) g = 1;
        if (g > 0 && G > GRID_MAX_CELLS / g) { overflow = true; break; }
        G *= g;
      }
      if (overflow || G > GRID_MAX_CELLS) {
        use_grid = false;
        break;
      }

      std::vector<int64_t> stride(F);
      int64_t s = 1;
      for (int64_t d = 0; d < F; ++d) {
        stride[d] = s;
        s *= gd_cpu[d].item<int64_t>();
      }

      auto cell_d = at::floor((seg_soa - mn.unsqueeze(1)) / r_f).to(at::kLong); // [F,n]
      auto cell_id = at::zeros({n}, opt_i64);
      for (int64_t d = 0; d < F; ++d) {
        cell_id = cell_id + cell_d[d] * stride[d];
      }
      cell_id = at::clamp(cell_id, 0, G - 1);

      // Argsort/Bincount 仅 AiCore 支持 float32/int32；G<=2^20 时 float32 精确表示 cell id
      auto cell_id_f = cell_id.to(at::kFloat);
      auto ord = at::argsort(cell_id_f, /*stable=*/true, /*dim=*/0, /*descending=*/false); // [n]
      auto sorted_seg = seg_soa.index_select(1, ord).contiguous();                        // [F,n]

      auto ord_i32 = (ord + xs).to(at::kCPU).to(at::kInt);   // 全局原下标
      auto counts = at::bincount(cell_id.to(at::kInt), c10::nullopt, G); // [G] long
      auto cs = at::cat({at::zeros({1}, counts.options()), at::cumsum(counts, 0)}, 0); // [G+1]

      sorted_segs.push_back(sorted_seg);
      order_segs.push_back(ord_i32.to(opt_i32));
      cs_segs.push_back(cs.to(at::kInt));
      cs_total += (G + 1);
      cs_off.push_back(cs_total);
      for (int64_t d = 0; d < F; ++d) {
        min_vec.push_back(mn_cpu[d].item<float>());
        g_vec.push_back(static_cast<int32_t>(gd_cpu[d].item<int64_t>()));
      }
    }

    if (use_grid) {
      sorted_x = at::cat(sorted_segs, 1).contiguous();      // [F,N]
      order_t = at::cat(order_segs, 0).to(at::kInt).contiguous();
      cell_start = at::cat(cs_segs, 0).to(at::kInt).contiguous();
      cell_start_off =
          at::from_blob(cs_off.data(), {static_cast<long>(cs_off.size())}, at::kInt)
              .clone().to(dev);
      grid_min =
          at::from_blob(min_vec.data(), {static_cast<long>(min_vec.size())}, at::kFloat)
              .clone().to(dev);
      grid_g =
          at::from_blob(g_vec.data(), {static_cast<long>(g_vec.size())}, at::kInt)
              .clone().to(dev);
      use_grid_t = at::tensor({1}, opt_i32);
    }
  }

  if (!use_grid) {
    sorted_x = at::empty({1}, x_c.options());
    order_t = at::zeros({1}, opt_i32);
    cell_start = at::zeros({1}, opt_i32);
    cell_start_off = at::zeros({1}, opt_i32);
    grid_min = at::zeros({1}, opt_f32);
    grid_g = at::zeros({1}, opt_i32);
    use_grid_t = at::zeros({1}, opt_i32);
  }

  const int64_t y_size = y_c.size(0);
  const int64_t out_dim = y_size * max_num_neighbors;
  auto result = at::full({3, out_dim}, /*fill_value=*/-1, opt_i32);

  EXEC_NPU_CMD(aclnnRadius,
               x_soa, y_soa, px, py,
               sorted_x, order_t, cell_start, cell_start_off, grid_min, grid_g, use_grid_t,
               r, max_num_neighbors, ignore_same_index,
               result);

  auto valid = result[0].ne(-1);
  auto idx = at::nonzero(valid).squeeze(1);
  auto row = result[1].index_select(0, idx);
  auto col = result[0].index_select(0, idx);
  return torch::stack({row, col}, 0).to(at::kLong).contiguous();
}
