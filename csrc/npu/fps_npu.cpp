#include "fps_npu.h"
#include "include/pytorch_npu_helper.hpp"

#include <ATen/ATen.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

static inline int64_t AlignUp4Int64(int64_t x) {
    return ((x + 3) / 4) * 4;
}

static inline int64_t AlignUp8Int64(int64_t x) {
    return ((x + 7) / 8) * 8;
}

constexpr int64_t FPS_FIXED_BLOCK_DIM = 40;

// mode
constexpr int32_t FPS_MODE_BATCH_PARALLEL = 0;
constexpr int32_t FPS_MODE_INNER_40 = 1;

// hybrid threshold
constexpr int64_t FPS_INNER_MIN_POINTS = 16384;
constexpr int64_t FPS_BALANCED_BATCH_THRESHOLD = 20;
constexpr double FPS_IMBALANCE_THRESHOLD = 2.0;

// DataCopy 32B slot
constexpr int64_t LOCAL_VAL_SLOT_FLOAT = 8;   // 8 float = 32B
constexpr int64_t LOCAL_IDX_SLOT_INT64 = 4;   // 4 int64 = 32B
constexpr int64_t CHOSEN_SLOT_INT64 = 4;      // 4 int64 = 32B

// SyncAll workspace: 40 cores * 32B = 1280B = 320 int32
constexpr int64_t FPS_SYNC_INT32_ELEMS = FPS_FIXED_BLOCK_DIM * 8;

} // namespace

torch::Tensor fps_npu(torch::Tensor src,
                      torch::Tensor ptr,
                      torch::Tensor ratio,
                      bool random_start) {
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
                "fps_npu: input 'src' must be on NPU device");
    TORCH_CHECK(ptr.device().type() == c10::DeviceType::PrivateUse1,
                "fps_npu: input 'ptr' must be on NPU device");
    TORCH_CHECK(ratio.device().type() == c10::DeviceType::PrivateUse1,
                "fps_npu: input 'ratio' must be on NPU device");

    TORCH_CHECK(ptr.dim() == 1,
                "fps_npu: input 'ptr' must be 1D");
    TORCH_CHECK(src.dim() >= 2,
                "fps_npu: input 'src' must have at least 2 dimensions");

    // 与 CUDA 版一致：src.view({src.size(0), -1}).contiguous()
    auto src_aos = src.view({src.size(0), -1}).contiguous();

    const int64_t dim = src_aos.size(1);
    TORCH_CHECK(dim > 0,
                "fps_npu: flattened feature dimension must be greater than 0");

    auto ptr_contig = ptr.to(at::kLong).contiguous();

    const int64_t batch_size = ptr_contig.numel() - 1;
    TORCH_CHECK(batch_size > 0,
                "fps_npu: ptr must contain at least two elements");

    // 同步 ptr 到 CPU，用于构造 padded_point_ptr / padded_src_index / hybrid mode。
    auto ptr_cpu = ptr_contig.to(at::kCPU).contiguous();
    const int64_t* ptr_data = ptr_cpu.data_ptr<int64_t>();

    TORCH_CHECK(ptr_data[0] == 0,
                "fps_npu: ptr[0] must be 0");
    TORCH_CHECK(ptr_data[batch_size] == src_aos.size(0),
                "fps_npu: ptr[-1] must equal src.size(0)");

    for (int64_t b = 0; b < batch_size; ++b) {
        TORCH_CHECK(ptr_data[b + 1] >= ptr_data[b],
                    "fps_npu: ptr must be non-decreasing");
    }

    // 与 CUDA 版一致：
    // deg.toType(ratio.scalar_type()) * ratio -> ceil -> long -> cumsum
    auto deg = ptr_contig.narrow(0, 1, batch_size) -
               ptr_contig.narrow(0, 0, batch_size);

    auto out_ptr = deg.to(ratio.scalar_type()) * ratio;
    out_ptr = out_ptr.ceil().to(at::kLong).cumsum(0);
    out_ptr = at::cat({at::zeros({1}, ptr_contig.options()), out_ptr}, 0);

    torch::Tensor start;
    if (random_start) {
        start = at::rand({batch_size}, src_aos.options());
        start = (start * deg.to(ratio.scalar_type())).to(at::kLong);
    } else {
        start = at::zeros({batch_size}, ptr_contig.options());
    }

    // 同步 out_ptr 到 CPU，用于构造 padded_out_ptr / gather_index / hybrid work。
    auto out_ptr_cpu = out_ptr.to(at::kCPU).contiguous();
    const int64_t* out_ptr_data = out_ptr_cpu.data_ptr<int64_t>();

    const int64_t n_out = out_ptr_data[batch_size];

    if (n_out == 0) {
        return at::empty({0}, out_ptr.options());
    }

    // ============================================================
    // hybrid mode selection
    // ============================================================
    int64_t max_points = 0;
    double total_work = 0.0;
    double max_work = 0.0;

    for (int64_t b = 0; b < batch_size; ++b) {
        const int64_t points = ptr_data[b + 1] - ptr_data[b];
        const int64_t samples = out_ptr_data[b + 1] - out_ptr_data[b];

        max_points = std::max(max_points, points);

        const double work =
            static_cast<double>(points) *
            static_cast<double>(samples) *
            static_cast<double>(dim);

        total_work += work;
        max_work = std::max(max_work, work);
    }

    const double avg_work =
        batch_size > 0 ? total_work / static_cast<double>(batch_size) : 0.0;

    const double imbalance =
        avg_work > 0.0 ? max_work / avg_work : 1.0;

    const bool small_data = max_points < FPS_INNER_MIN_POINTS;
    const bool low_dim = dim <= 3;
    const bool many_balanced_batches =
        batch_size >= FPS_BALANCED_BATCH_THRESHOLD &&
        imbalance < FPS_IMBALANCE_THRESHOLD;

    const bool use_inner40 =
        (!small_data) &&
        (!low_dim) &&
        (!many_balanced_batches);

    auto mode_cpu = at::empty({1}, ptr_cpu.options().dtype(at::kInt));
    int32_t* mode_data = mode_cpu.data_ptr<int32_t>();
    mode_data[0] = use_inner40 ? FPS_MODE_INNER_40 : FPS_MODE_BATCH_PARALLEL;

    auto mode = mode_cpu.to(ptr.device(), at::kInt, true, true);

    // ============================================================
    // 1. 构造 padded_point_ptr 和 padded_src_index
    //
    // 每个 batch 的点存储长度 AlignUp8(num_points)。
    // ============================================================
    auto padded_point_ptr_cpu = at::empty({batch_size + 1}, ptr_cpu.options());
    int64_t* padded_point_ptr_data = padded_point_ptr_cpu.data_ptr<int64_t>();

    int64_t padded_point_running = 0;
    padded_point_ptr_data[0] = 0;

    for (int64_t b = 0; b < batch_size; ++b) {
        const int64_t begin = ptr_data[b];
        const int64_t end = ptr_data[b + 1];
        const int64_t num_points = end - begin;

        const int64_t padded_begin = padded_point_running;
        const int64_t padded_len = AlignUp8Int64(num_points);

        padded_point_ptr_data[b] = padded_begin;
        padded_point_running += padded_len;
        padded_point_ptr_data[b + 1] = padded_point_running;
    }

    const int64_t padded_total_points = padded_point_ptr_data[batch_size];

    auto padded_src_index_cpu = at::empty({padded_total_points}, ptr_cpu.options());
    int64_t* padded_src_index_data = padded_src_index_cpu.data_ptr<int64_t>();

    for (int64_t b = 0; b < batch_size; ++b) {
        const int64_t compact_begin = ptr_data[b];
        const int64_t compact_end = ptr_data[b + 1];
        const int64_t num_points = compact_end - compact_begin;

        const int64_t padded_begin = padded_point_ptr_data[b];
        const int64_t padded_end = padded_point_ptr_data[b + 1];
        const int64_t padded_len = padded_end - padded_begin;

        for (int64_t j = 0; j < num_points; ++j) {
            padded_src_index_data[padded_begin + j] = compact_begin + j;
        }

        for (int64_t j = num_points; j < padded_len; ++j) {
            padded_src_index_data[padded_begin + j] =
                (num_points > 0) ? compact_begin : 0;
        }
    }

    auto padded_point_ptr = padded_point_ptr_cpu.to(ptr.device(), at::kLong, true, true);
    auto padded_src_index = padded_src_index_cpu.to(ptr.device(), at::kLong, true, true);

    // padded AoS: [padded_total_points, dim]
    auto padded_src_aos = src_aos.index_select(0, padded_src_index).contiguous();

    // padded SoA: [dim, padded_total_points]
    auto src_soa = padded_src_aos.transpose(0, 1).contiguous();

    // ============================================================
    // 2. 构造 padded_out_ptr 和 gather_index
    //
    // int64 输出：4 个元素 = 32B。
    // 每个 batch 输出长度 AlignUp4(sample_num)。
    // ============================================================
    auto padded_out_ptr_cpu = at::empty({batch_size + 1}, out_ptr_cpu.options());
    int64_t* padded_out_ptr_data = padded_out_ptr_cpu.data_ptr<int64_t>();

    auto gather_index_cpu = at::empty({n_out}, out_ptr_cpu.options());
    int64_t* gather_data = gather_index_cpu.data_ptr<int64_t>();

    int64_t padded_out_running = 0;
    int64_t gather_pos = 0;

    padded_out_ptr_data[0] = 0;

    for (int64_t b = 0; b < batch_size; ++b) {
        const int64_t compact_begin = out_ptr_data[b];
        const int64_t compact_end = out_ptr_data[b + 1];
        const int64_t sample_num = compact_end - compact_begin;

        TORCH_CHECK(sample_num >= 0,
                    "fps_npu: out_ptr must be non-decreasing");

        const int64_t padded_begin = padded_out_running;
        const int64_t padded_len = AlignUp4Int64(sample_num);

        padded_out_ptr_data[b] = padded_begin;

        for (int64_t j = 0; j < sample_num; ++j) {
            gather_data[gather_pos++] = padded_begin + j;
        }

        padded_out_running += padded_len;
        padded_out_ptr_data[b + 1] = padded_out_running;
    }

    TORCH_CHECK(gather_pos == n_out,
                "fps_npu: internal gather index size mismatch");

    const int64_t padded_n_out = padded_out_ptr_data[batch_size];

    auto padded_out_ptr = padded_out_ptr_cpu.to(ptr.device(), at::kLong, true, true);
    auto gather_index = gather_index_cpu.to(ptr.device(), at::kLong, true, true);

    auto padded_out = at::empty({padded_n_out}, out_ptr.options());

    // dist GM workspace: [padded_total_points]
    auto dist = at::empty({padded_total_points}, src_soa.options().dtype(at::kFloat));

    // ============================================================
    // 3. extra workspaces for inner-40 mode
    //
    // mode=0 时这些 workspace 不使用，但为了统一 op signature 仍然传入。
    // ============================================================
    auto local_max_val = at::empty(
        {FPS_FIXED_BLOCK_DIM * LOCAL_VAL_SLOT_FLOAT},
        src_soa.options().dtype(at::kFloat));

    auto local_max_idx = at::empty(
        {FPS_FIXED_BLOCK_DIM * LOCAL_IDX_SLOT_INT64},
        ptr_contig.options());

    auto chosen_local = at::empty(
        {CHOSEN_SLOT_INT64},
        ptr_contig.options());

    auto sync_workspace = at::zeros(
        {FPS_SYNC_INT32_ELEMS},
        ptr_contig.options().dtype(at::kInt));

    EXEC_NPU_CMD(aclnnFarthestPointSampling,
                 src_soa,
                 ptr_contig,
                 padded_point_ptr,
                 out_ptr,
                 padded_out_ptr,
                 start,
                 mode,
                 dist,
                 local_max_val,
                 local_max_idx,
                 chosen_local,
                 sync_workspace,
                 padded_out);

    auto out = padded_out.index_select(0, gather_index);

    return out;
}