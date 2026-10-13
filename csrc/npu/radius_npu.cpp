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

torch::Tensor radius_native_single(torch::Tensor x, torch::Tensor y, double r,
                                  int64_t cap, bool ignore, bool shared_view) {
  const int64_t n = x.size(1), m = y.size(1), f = x.size(0);
  const auto i32 = x.options().dtype(at::kInt);
  int64_t native_cells = 4096;
  int64_t query_mode = n == m ? (shared_view ? 2 : 3) : 1;
  auto sorted_x = at::empty_like(x);
  auto order = at::empty({n}, i32);
  auto starts = at::empty({native_cells + 1}, i32);
  auto offsets = at::empty({2}, i32);
  auto origin = at::empty({f}, x.options());
  auto dims = at::empty({f}, i32);
  auto enabled = at::empty({1}, i32);
  auto state = at::empty({656 + n + 40 * ((native_cells + 8) / 8 * 8)}, i32);
  auto packed = at::empty({2, m * cap}, x.options().dtype(at::kLong));
  auto result = at::empty({2, m * cap}, i32);
  torch::Tensor ptr_x, ptr_y;
  auto dispatch = [&](int64_t phase) {
    EXEC_NPU_CMD(aclnnRadius, x, y, ptr_x, ptr_y, sorted_x, order, starts,
                 offsets, origin, dims, enabled, state, packed,
                 r, cap, ignore, phase, native_cells, query_mode, result);
  };
  dispatch(7);  // per-core bounds, no host transfer
  dispatch(8);  // device grid metadata, or complete-search fallback flag
  dispatch(3);
  dispatch(4);
  dispatch(5);
  if (query_mode == 3) dispatch(6);
  dispatch(0);
  dispatch(1);
  dispatch(2);
  const int64_t count = state[640].item<int32_t>();
  return packed.view({-1}).narrow(0, 0, count * 2).view({2, count});
}

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
  TORCH_CHECK(x.device() == y.device() && x.scalar_type() == y.scalar_type(),
              "radius_npu: x and y must share device and dtype");
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
  auto y_soa = (x_c.data_ptr() == y_c.data_ptr() && x_c.sizes() == y_c.sizes())
      ? x_soa : y_c.transpose(0, 1).contiguous();

  const auto dev = x_c.device();
  const auto opt_i32 = x_c.options().dtype(at::kInt);
  const auto opt_f32 = x_c.options().dtype(at::kFloat);
  const auto opt_i64 = x_c.options().dtype(at::kLong);
  const float r_f = static_cast<float>(r);
  if (!px.defined() && !py.defined() && x_c.scalar_type() == at::kFloat &&
      F >= 1 && F <= GRID_MAX_DIM && r_f > 0 && N > 0 && M > 0 && max_num_neighbors > 0) {
    return radius_native_single(x_soa, y_soa, r, max_num_neighbors, ignore_same_index,
                                x_c.data_ptr() == y_c.data_ptr());
  }

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
  int64_t native_cells = 0;
  int64_t query_mode = 0;
  bool group_candidate = false;

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
      if (B == 1 && G <= 4096 && x_c.scalar_type() == at::kFloat) {
        native_cells = G;
        sorted_x = at::empty({F, N}, x_c.options());
        order_t = at::empty({N}, opt_i32);
        cell_start = at::empty({G + 1}, opt_i32);
        cs_off.push_back(G + 1);
        for (int64_t d = 0; d < F; ++d) {
          min_vec.push_back(mn_cpu[d].item<float>());
          g_vec.push_back(static_cast<int32_t>(gd_cpu[d].item<int64_t>()));
        }
        continue;
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

      auto ord_i32 = (ord + xs).to(at::kInt);   // 全局原下标，留在设备端
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
      if (!native_cells) {
        sorted_x = at::cat(sorted_segs, 1).contiguous();
        order_t = at::cat(order_segs, 0).to(at::kInt).contiguous();
        cell_start = at::cat(cs_segs, 0).to(at::kInt).contiguous();
      }
      cell_start_off =
          at::from_blob(cs_off.data(), {static_cast<long>(cs_off.size())}, at::kLong)
              .clone().to(opt_i32);
      grid_min =
          at::from_blob(min_vec.data(), {static_cast<long>(min_vec.size())}, at::kFloat)
              .clone().to(dev);
      grid_g =
          at::from_blob(g_vec.data(), {static_cast<long>(g_vec.size())}, at::kInt)
              .clone().to(dev);
      const int64_t cells = cell_start.numel() - 1;
      const bool same_candidate = B == 1 && N == M && x_c.scalar_type() == at::kFloat &&
          max_num_neighbors <= 64 && (N >= 2 * max_num_neighbors * cells ||
          (N >= 10240 && N >= max_num_neighbors * cells / 2));
      group_candidate = same_candidate;
      const bool same_view = same_candidate && x_c.data_ptr() == y_c.data_ptr();
      use_grid_t = at::tensor({same_view ? 2 : (same_candidate ? 3 : 1)}, opt_i32);
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
  if (out_dim == 0 || N == 0) return at::empty({2, 0}, opt_i64);
  auto result = at::full({2, out_dim}, /*fill_value=*/-1, opt_i32);
  const int64_t native_stride = ((native_cells + 8) / 8) * 8;
  auto compact_state = at::empty({656 + (native_cells ? N + 40 * native_stride : 0)}, opt_i32);
  auto packed = at::empty({2, out_dim}, opt_i64);
  int64_t phase = 0;
  if (native_cells) {
    for (phase = 3; phase <= 5; ++phase) {
      EXEC_NPU_CMD(aclnnRadius,
                   x_soa, y_soa, px, py,
                   sorted_x, order_t, cell_start, cell_start_off, grid_min, grid_g, use_grid_t,
                   compact_state, packed, r, max_num_neighbors, ignore_same_index, phase, native_cells, query_mode, result);
    }
    phase = 0;
  }
  // Equal point clouds may arrive as separate tensors. Verify their contents
  // on device for this call before reusing x's spatial ordering for queries.
  if (group_candidate &&
      x_c.data_ptr() != y_c.data_ptr()) {
    phase = 6;
    EXEC_NPU_CMD(aclnnRadius,
                 x_soa, y_soa, px, py,
                 sorted_x, order_t, cell_start, cell_start_off, grid_min, grid_g, use_grid_t,
                 compact_state, packed, r, max_num_neighbors, ignore_same_index, phase, native_cells, query_mode, result);
    phase = 0;
  }

  EXEC_NPU_CMD(aclnnRadius,
               x_soa, y_soa, px, py,
               sorted_x, order_t, cell_start, cell_start_off, grid_min, grid_g, use_grid_t,
               compact_state, packed,
               r, max_num_neighbors, ignore_same_index, phase, native_cells, query_mode,
               result);
  for (phase = 1; phase <= 2; ++phase) {
    EXEC_NPU_CMD(aclnnRadius,
                 x_soa, y_soa, px, py,
                 sorted_x, order_t, cell_start, cell_start_off, grid_min, grid_g, use_grid_t,
                 compact_state, packed,
                 r, max_num_neighbors, ignore_same_index, phase, native_cells, query_mode, result);
  }
  const int64_t count = compact_state[640].item<int32_t>();
  return packed.view({-1}).narrow(0, 0, count * 2).view({2, count});
}
