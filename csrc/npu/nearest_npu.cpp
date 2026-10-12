// nearest_npu.cpp
// NPU middle-layer adapter for Nearest neighbor point assignment.
//
// Bridges torch_cluster::nearest to Ascend NPU via EXEC_NPU_CMD.
// Custom CANN kernel: aclnnNearestNeighbor
// Prototype definition: csrc/npu/impl/nearest_npu.json
//
// Function signature (mirrors csrc/nearest.cpp):
//   torch::Tensor nearest(torch::Tensor x, torch::Tensor y,
//                         torch::Tensor ptr_x, torch::Tensor ptr_y)
//
// Returns:
//   cluster: (N_x,) int64 — for each point in x, the index of the
//            nearest cluster centre in y.
//
// Fast path (float32, non-empty): argmin_j ||x_i - y_j||^2 is computed via the
// norm expansion ||x||^2 - 2 x.y^T + ||y||^2 folded into a single Cube matmul
// over the augmented features [-2x, ||x||^2, 1] and [y, 1, ||y||^2]. The
// distance tile is produced in one pass (no elementwise adds) and reduced with
// a min. This is a multi-core SIMD/Cube implementation.
// The fallback (non-float32) still calls aclnnNearestNeighbor, whose Ascend C
// kernel is a multi-core AIV (SIMD) implementation.

#include "nearest_npu.h"
#include "include/pytorch_npu_helper.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cstdint>

namespace {

// Keep the per-tile distance matrix near 8M float32 elements: large enough to
// amortize launches, small enough to avoid the memory-bound slowdown observed
// for bigger Cube tiles on 910B.
constexpr int64_t DIST_WORKSPACE_FLOATS = 8 * 1024 * 1024;
constexpr int64_t DIST_TILE_MAX = 8192;

// Nearest y-index for every row of a single (x, y) segment.
// Returns int64[N] holding indices local to `y`.
at::Tensor NearestSegment(const at::Tensor& x, const at::Tensor& y) {
  const int64_t n = x.size(0);
  const int64_t m = y.size(0);
  auto out = at::empty({n}, x.options().dtype(at::kLong));
  if (n == 0 || m == 0) {
    return out;
  }

  // Center to bound cancellation in the norm expansion.
  auto origin = x.slice(0, 0, 1);
  auto xr = x - origin;
  auto yr = y - origin;
  auto xr_norm = (xr * xr).sum(1);       // [N]
  auto yr_norm = (yr * yr).sum(1);       // [M]

  // Fold the whole squared-distance expansion into a single Cube matmul:
  //   d[i,j] = ||x_i||^2 - 2 x_i.y_j + ||y_j||^2
  //          = [ -2 x_i , ||x_i||^2 , 1 ] . [ y_j , 1 , ||y_j||^2 ]
  // so the distance tile is produced in one pass with no elementwise adds.
  auto one_n = at::ones({n, 1}, xr.options());
  auto one_m = at::ones({1, m}, yr.options());
  auto x_aug = at::cat({xr * -2.0, xr_norm.unsqueeze(1), one_n}, 1).contiguous();
  // Build the [F+2, M] operand directly (concat along rows yields contiguous).
  auto y_aug_t = at::cat({yr.t(), one_m, yr_norm.unsqueeze(0)}, 0).contiguous();

  const int64_t tile = std::max<int64_t>(
      64, std::min<int64_t>({DIST_TILE_MAX, DIST_WORKSPACE_FLOATS / std::max<int64_t>(1, m), n}));

  for (int64_t first = 0; first < n; first += tile) {
    const int64_t last = std::min(n, first + tile);
    auto d = at::matmul(x_aug.slice(0, first, last), y_aug_t);  // [T, M]
    out.slice(0, first, last).copy_(std::get<1>(d.min(/*dim=*/1)));
  }
  return out;
}

}  // namespace

torch::Tensor nearest_npu(torch::Tensor x,
                          torch::Tensor y,
                          torch::Tensor ptr_x,
                          torch::Tensor ptr_y) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
              "nearest_npu: input 'x' must be on NPU device");
  TORCH_CHECK(y.device().type() == c10::DeviceType::PrivateUse1,
              "nearest_npu: input 'y' must be on NPU device");

  auto xc = x.contiguous();
  auto yc = y.contiguous();
  if (xc.dim() == 1) {
    xc = xc.view({-1, 1});
  }
  if (yc.dim() == 1) {
    yc = yc.view({-1, 1});
  }

  const int64_t n_x = xc.size(0);
  auto out = at::empty({n_x}, xc.options().dtype(at::kLong));

  const bool fast = xc.scalar_type() == at::kFloat &&
                    yc.scalar_type() == at::kFloat &&
                    n_x > 0 && yc.size(0) > 0;
  if (fast) {
    auto pxc = ptr_x.contiguous();
    auto pyc = ptr_y.contiguous();
    // A two-element CSR pointer is necessarily [0, total] (one segment), so the
    // batch offset never needs a device-to-host read in the common path.
    if (pxc.numel() == 2 && pyc.numel() == 2) {
      out.copy_(NearestSegment(xc, yc));
      return out;
    }
    auto px = pxc.to(at::kCPU).to(at::kLong).contiguous();
    auto py = pyc.to(at::kCPU).to(at::kLong).contiguous();
    const int64_t* pxd = px.data_ptr<int64_t>();
    const int64_t* pyd = py.data_ptr<int64_t>();
    const int64_t segs = px.numel() - 1;
    for (int64_t q = 0; q < segs; ++q) {
      const int64_t xs = pxd[q], xe = pxd[q + 1];
      const int64_t ys = pyd[q], ye = pyd[q + 1];
      if (xe <= xs || ye <= ys) {
        continue;
      }
      auto seg = NearestSegment(xc.slice(0, xs, xe), yc.slice(0, ys, ye));
      out.slice(0, xs, xe).copy_(seg + ys);
    }
    return out;
  }

  EXEC_NPU_CMD(aclnnNearestNeighbor, x, y, ptr_x, ptr_y, out);
  return out;
}
