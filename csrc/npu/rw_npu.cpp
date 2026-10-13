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
#include <c10/core/DeviceGuard.h>
#include <mutex>
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#include <torch_npu/csrc/core/npu/NPUGraphsUtils.h>

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
  TORCH_CHECK((rowptr.scalar_type() == at::kLong || rowptr.scalar_type() == at::kInt) &&
                  col.scalar_type() == rowptr.scalar_type() &&
                  start.scalar_type() == rowptr.scalar_type(),
              "random_walk_npu: CSR inputs must have the same int32 or int64 dtype");
  TORCH_CHECK(rowptr.dim() == 1 && col.dim() == 1 && start.dim() == 1 &&
                  rowptr.numel() >= 1, "random_walk_npu: expected 1D CSR inputs");
  TORCH_CHECK(walk_length >= 0 && walk_length <= INT32_MAX,
              "random_walk_npu: invalid walk length");
  TORCH_CHECK(std::isfinite(p) && std::isfinite(q) && p > 0.0 && q > 0.0,
              "random_walk_npu: p and q must be positive finite numbers");
  const c10::DeviceGuard device_guard(start.device());
  const int64_t count = start.numel();
  if (count == 0)
    return {at::empty({0, walk_length + 1}, start.options()),
            at::empty({0, walk_length}, start.options())};
  if (walk_length == 0)
    return {start.contiguous().view({count, 1}),
            at::empty({count, 0}, start.options())};
  TORCH_CHECK(rowptr.numel() - 1 <= INT32_MAX && col.numel() <= INT32_MAX &&
                  count <= INT32_MAX, "random_walk_npu: shape exceeds tiling range");
  const int64_t line_elements = 64 / start.element_size();
  const int64_t num_nodes = rowptr.numel() - 1;
  const int64_t steps = count * walk_length;
  // This only chooses the internal layout. Device validation still proves
  // regularity and all index bounds before the cooperative sampler runs.
  const bool shard_shape = (col.numel() == num_nodes * 8 && num_nodes <= 524288) ||
      (col.numel() == num_nodes * 4 && num_nodes <= 1048576 && count >= 65536);
  const int64_t output_mode = start.scalar_type() == at::kInt &&
      p == 1.0 && q == 1.0 && num_nodes >= 8192 && shard_shape &&
      (num_nodes & (num_nodes - 1)) == 0 && col.numel() % 32768 == 0 &&
      count >= 8192 && count <= 262144 && steps >= 32768 ? 1 : 0;
  const int64_t nodes_pitch = output_mode ? (count + 15) / 16 * 16 :
      ((walk_length + line_elements) / line_elements) * line_elements;
  const int64_t edges_pitch = output_mode ? nodes_pitch :
      ((walk_length + line_elements - 1) / line_elements) * line_elements;
  auto walks = output_mode ? at::empty({walk_length + 1, nodes_pitch}, start.options()) :
      at::empty({count, nodes_pitch}, start.options());
  auto e_ids = output_mode ? at::empty({walk_length, edges_pitch}, start.options()) :
      at::empty({count, edges_pitch}, start.options());
  auto rp = rowptr.contiguous();
  auto ci = col.contiguous();
  auto starts = start.contiguous();
  at::Tensor random;
  // Exact CANN float32 Philox, computed directly in the requested layout.
  // The bounded case has a 32-bit counter group index. Graph capture keeps
  // the framework's graph-safe RNG path and every other dtype/layout stays
  // on the original at::rand implementation.
  const bool flat_vector_rng = !output_mode && start.scalar_type() == at::kInt &&
      p == 1.0 && q == 1.0 && count >= 8192 && steps >= 32768 && walk_length % 16 == 0;
  const bool vector_rng = (output_mode || flat_vector_rng) && walk_length % 4 == 0 &&
      steps <= UINT32_MAX &&
      // The NPU input/device guard guarantees an initialized context. The
      // exported query avoids the inline helper's hidden IsContextInitialized
      // symbol in the installed torch_npu build.
      c10_npu::currentStreamCaptureStatusMayInitCtx() == c10_npu::CaptureStatus::None;
  if (vector_rng) {
    random = output_mode ? at::empty({walk_length, count}, start.options().dtype(at::kFloat)) :
        at::empty({count, walk_length}, start.options().dtype(at::kFloat));
    auto seed_cpu = at::empty({4}, at::TensorOptions().dtype(at::kInt)
                                        .device(at::kCPU).pinned_memory(true));
    std::pair<uint64_t, uint64_t> state;
    auto* generator = at_npu::detail::getDefaultNPUGenerator(start.device().index())
                          .get<at_npu::NPUGeneratorImpl>();
    {
      std::lock_guard<std::mutex> lock(generator->mutex_);
      // Native Uniform advances by 12 (its requested increment rounded to
      // a four-word boundary), independently of shape. Preserve that state
      // so interleaved torch.rand calls see the identical following values.
      state = generator->philox_engine_inputs(12);
    }
    auto* words = seed_cpu.data_ptr<int32_t>();
    words[0] = static_cast<int32_t>(state.first);
    words[1] = static_cast<int32_t>(state.first >> 32);
    words[2] = static_cast<int32_t>(state.second);
    words[3] = static_cast<int32_t>(state.second >> 32);
    auto seed_device = seed_cpu.to(start.device(), at::kInt, true, false);
    auto random_words = random.view(at::kInt).reshape({steps});
    const int64_t rng_phase = 4;
    EXEC_NPU_CMD(aclnnRandomWalk, seed_device, ci, starts, random,
                 walk_length, p, q, rng_phase, output_mode,
                 walks, e_ids, random_words);
  } else {
    random = at::rand({count, walk_length}, start.options().dtype(at::kFloat));
    if (output_mode) random = random.transpose(0, 1).contiguous();
  }
  const int64_t parts = col.numel() / 32768;
  const int64_t queue_capacity = nodes_pitch + 640 * (parts + 1);
  const int64_t flag_count = output_mode ?
      640 + nodes_pitch + 80 * parts + 3 * queue_capacity : 640;
  auto flags = at::empty({flag_count}, start.options().dtype(at::kInt));
  // Amortize the complete CSR check across enough transition work. A shape
  // alone never enables the specialization: every row pointer is checked.
  const bool check_regular = num_nodes > 0 && ci.numel() > 0 &&
      ci.numel() % num_nodes == 0 && steps >= 32768 && num_nodes <= steps / 2 &&
      ((p == 1.0 && q == 1.0) || ci.numel() == num_nodes);
  if (check_regular) {
    int64_t phase = 0;
    EXEC_NPU_CMD(aclnnRandomWalk, rp, ci, starts, random,
                 walk_length, p, q, phase, output_mode, walks, e_ids, flags);
  }
  int64_t phase = check_regular ? 2 : 1;
  EXEC_NPU_CMD(aclnnRandomWalk, rp, ci, starts, random,
               walk_length, p, q, phase, output_mode,
               walks, e_ids, flags);
  if (output_mode)
    return {walks.narrow(1, 0, count).transpose(0, 1),
            e_ids.narrow(1, 0, count).transpose(0, 1)};
  return {walks.narrow(1, 0, walk_length + 1),
          e_ids.narrow(1, 0, walk_length)};
}
