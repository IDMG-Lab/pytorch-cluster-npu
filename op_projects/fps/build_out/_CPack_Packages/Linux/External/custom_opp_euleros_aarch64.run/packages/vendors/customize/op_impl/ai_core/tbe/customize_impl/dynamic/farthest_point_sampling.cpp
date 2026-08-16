#include "kernel_operator.h"

using namespace AscendC;

typedef int64_t IndexType;

constexpr uint32_t TILE_SIZE = 1024;
constexpr uint32_t OUT_TILE_SIZE = 1024;
constexpr uint32_t FAST_DIM_3D = 3;

constexpr uint32_t FPS_FIXED_BLOCK_DIM = 40;

constexpr int32_t FPS_MODE_BATCH_PARALLEL = 0;
constexpr int32_t FPS_MODE_INNER_40 = 1;

constexpr uint32_t LOCAL_VAL_SLOT_FLOAT = 8;   // 8 float = 32B
constexpr uint32_t LOCAL_IDX_SLOT_INT64 = 4;   // 4 int64 = 32B
constexpr uint32_t CHOSEN_SLOT_INT64 = 4;      // 4 int64 = 32B

constexpr uint32_t LOCAL_VAL_TOTAL_FLOAT =
    FPS_FIXED_BLOCK_DIM * LOCAL_VAL_SLOT_FLOAT;
constexpr uint32_t LOCAL_IDX_TOTAL_INT64 =
    FPS_FIXED_BLOCK_DIM * LOCAL_IDX_SLOT_INT64;

constexpr uint32_t FPS_SYNC_INT32_ELEMS = FPS_FIXED_BLOCK_DIM * 8;

// center point 预加载最多支持 64 维。
// dim > 64 时自动回退到 LoadScalarFromSoA。
constexpr uint32_t MAX_CENTER_DIM = 64;

constexpr float INIT_DIST = 1.0e10f;
constexpr float NEG_INF = -1.0f;

class KernelFPS {
public:
    __aicore__ inline KernelFPS() {}

    __aicore__ inline void Init(GM_ADDR src,
                                GM_ADDR ptr,
                                GM_ADDR padded_point_ptr,
                                GM_ADDR out_ptr,
                                GM_ADDR padded_out_ptr,
                                GM_ADDR start,
                                GM_ADDR mode,
                                GM_ADDR dist,
                                GM_ADDR local_max_val,
                                GM_ADDR local_max_idx,
                                GM_ADDR chosen_local,
                                GM_ADDR sync_workspace,
                                GM_ADDR out,
                                GM_ADDR tiling_gm) {
        GET_TILING_DATA(tilingData, tiling_gm);

        dim = tilingData.dim;
        total_points = tilingData.totalPoints;
        batch_size = tilingData.batchSize;

        block_id = GetBlockIdx();

        ptr_gm = (__gm__ IndexType*)ptr;
        padded_point_ptr_gm = (__gm__ IndexType*)padded_point_ptr;
        out_ptr_gm = (__gm__ IndexType*)out_ptr;
        padded_out_ptr_gm = (__gm__ IndexType*)padded_out_ptr;
        raw_start_ptr = (__gm__ IndexType*)start;
        mode_ptr = (__gm__ int32_t*)mode;

        srcGm.SetGlobalBuffer((__gm__ float*)src);
        distGm.SetGlobalBuffer((__gm__ float*)dist);
        outGm.SetGlobalBuffer((__gm__ IndexType*)out);

        localMaxValGm.SetGlobalBuffer((__gm__ float*)local_max_val);
        localMaxIdxGm.SetGlobalBuffer((__gm__ IndexType*)local_max_idx);
        chosenLocalGm.SetGlobalBuffer((__gm__ IndexType*)chosen_local);
        syncGm.SetGlobalBuffer((__gm__ int32_t*)sync_workspace);

        for (uint32_t i = 0; i < 2; ++i) {
            pipe.InitBuffer(bufX[i], TILE_SIZE * sizeof(float));
            pipe.InitBuffer(bufY[i], TILE_SIZE * sizeof(float));
            pipe.InitBuffer(bufZ[i], TILE_SIZE * sizeof(float));
            pipe.InitBuffer(sqSum[i], TILE_SIZE * sizeof(float));
            pipe.InitBuffer(distTileBuf[i], TILE_SIZE * sizeof(float));
            pipe.InitBuffer(tileMaxBuf[i], 8 * sizeof(float));
            pipe.InitBuffer(reduceWorkBuf[i], TILE_SIZE * sizeof(float));
        }

        pipe.InitBuffer(outUbBuf, OUT_TILE_SIZE * sizeof(IndexType));

        // 用于加载一个 8-float 对齐片段。
        pipe.InitBuffer(scalarBuf, 8 * sizeof(float));

        // center point 预加载 buffer。
        pipe.InitBuffer(centerBuf, MAX_CENTER_DIM * sizeof(float));

        // 32B slot write/read buffers。
        pipe.InitBuffer(localValSlotBuf, LOCAL_VAL_SLOT_FLOAT * sizeof(float));
        pipe.InitBuffer(localIdxSlotBuf, LOCAL_IDX_SLOT_INT64 * sizeof(IndexType));
        pipe.InitBuffer(chosenSlotBuf, CHOSEN_SLOT_INT64 * sizeof(IndexType));

        // block0 批量读取 local max。
        pipe.InitBuffer(allLocalValBuf, LOCAL_VAL_TOTAL_FLOAT * sizeof(float));
        pipe.InitBuffer(allLocalIdxBuf, LOCAL_IDX_TOTAL_INT64 * sizeof(IndexType));

        // SyncAll soft-sync UB workspace。
        pipe.InitBuffer(syncUbBuf, FPS_SYNC_INT32_ELEMS * sizeof(int32_t));
    }

    __aicore__ inline void Process() {
        if (dim == 0 || batch_size == 0) {
            return;
        }

        int32_t mode = mode_ptr[0];

        if (mode == FPS_MODE_INNER_40) {
            ProcessInner40Mode();
        } else {
            ProcessBatchParallelMode();
        }
    }

private:
    // ============================================================
    // common
    // ============================================================

    __aicore__ inline uint32_t AlignUp8(uint32_t x) {
        return (x + 7) / 8 * 8;
    }

    __aicore__ inline uint32_t AlignUp4(uint32_t x) {
        return (x + 3) / 4 * 4;
    }

    __aicore__ inline void LoadBatchMeta(uint32_t batch_id) {
        batch_start_idx = ptr_gm[batch_id];
        batch_end_idx = ptr_gm[batch_id + 1];
        num_points = static_cast<uint32_t>(batch_end_idx - batch_start_idx);

        padded_point_start = padded_point_ptr_gm[batch_id];
        padded_point_end = padded_point_ptr_gm[batch_id + 1];
        padded_point_num = static_cast<uint32_t>(padded_point_end - padded_point_start);

        compact_out_start_pos = out_ptr_gm[batch_id];
        compact_out_end_pos = out_ptr_gm[batch_id + 1];
        num_to_sample = static_cast<uint32_t>(compact_out_end_pos - compact_out_start_pos);

        padded_out_start_pos = padded_out_ptr_gm[batch_id];
        padded_out_end_pos = padded_out_ptr_gm[batch_id + 1];
        padded_num_to_sample = static_cast<uint32_t>(padded_out_end_pos - padded_out_start_pos);
    }

    __aicore__ inline bool CheckBatchValid() {
        if (num_points == 0 || num_to_sample == 0) {
            return false;
        }

        if ((padded_point_start & 7) != 0 || (padded_point_num & 7) != 0) {
            return false;
        }

        if (padded_point_num < num_points) {
            return false;
        }

        if ((padded_out_start_pos & 3) != 0 ||
            (padded_num_to_sample & 3) != 0) {
            return false;
        }

        if (padded_num_to_sample < num_to_sample) {
            return false;
        }

        return true;
    }

    __aicore__ inline void SyncAll40() {
        LocalTensor<int32_t> syncUb = syncUbBuf.Get<int32_t>();

        PipeBarrier<PIPE_ALL>();

        AscendC::SyncAll<true>(
            syncGm,
            syncUb,
            static_cast<int32_t>(FPS_FIXED_BLOCK_DIM));

        PipeBarrier<PIPE_ALL>();
    }

    // ============================================================
    // center point preload
    // ============================================================

    __aicore__ inline void PreloadCenter(IndexType current_storage_idx) {
        center_preloaded = false;

        if (dim > MAX_CENTER_DIM) {
            return;
        }

        LocalTensor<float> center = centerBuf.Get<float>();
        LocalTensor<float> scalar = scalarBuf.Get<float>();

        IndexType aligned_idx = (current_storage_idx / 8) * 8;
        uint32_t shift = static_cast<uint32_t>(current_storage_idx - aligned_idx);

        for (uint32_t axis = 0; axis < dim; ++axis) {
            DataCopy(scalar,
                     srcGm[static_cast<IndexType>(axis) * total_points + aligned_idx],
                     8);

            PipeBarrier<PIPE_ALL>();

            center.SetValue(axis, scalar.GetValue(shift));
        }

        PipeBarrier<PIPE_ALL>();

        center_preloaded = true;
    }

    __aicore__ inline float LoadScalarFromSoA(uint32_t axis,
                                              IndexType storage_idx) {
        LocalTensor<float> scalar = scalarBuf.Get<float>();

        IndexType aligned_idx = (storage_idx / 8) * 8;
        uint32_t shift = static_cast<uint32_t>(storage_idx - aligned_idx);

        DataCopy(scalar,
                 srcGm[static_cast<IndexType>(axis) * total_points + aligned_idx],
                 8);

        PipeBarrier<PIPE_ALL>();

        return scalar.GetValue(shift);
    }

    __aicore__ inline float GetCenter(uint32_t axis,
                                      IndexType current_storage_idx) {
        if (center_preloaded && axis < MAX_CENTER_DIM) {
            LocalTensor<float> center = centerBuf.Get<float>();
            return center.GetValue(axis);
        }

        return LoadScalarFromSoA(axis, current_storage_idx);
    }

    // ============================================================
    // mode 0: old batch-parallel path
    // ============================================================

    __aicore__ inline void ProcessBatchParallelMode() {
        for (uint32_t batch_id = block_id;
             batch_id < batch_size;
             batch_id += FPS_FIXED_BLOCK_DIM) {
            LoadBatchMeta(batch_id);
            ProcessOneBatchParallel(batch_id);
        }
    }

    __aicore__ inline void ProcessOneBatchParallel(uint32_t batch_id) {
        if (!CheckBatchValid()) {
            return;
        }

        LocalTensor<IndexType> outUb = outUbBuf.Get<IndexType>();

        uint32_t out_cache_count = 0;
        uint32_t out_flush_offset = 0;

        InitDistWorkspaceParallel();

        IndexType first_pt_local = raw_start_ptr[batch_id];

        if (first_pt_local < 0 ||
            first_pt_local >= static_cast<IndexType>(num_points)) {
            return;
        }

        IndexType current_chosen_local = first_pt_local;
        IndexType current_chosen_global = batch_start_idx + current_chosen_local;

        StoreOutputToUb(outUb,
                        current_chosen_global,
                        out_cache_count,
                        out_flush_offset);

        for (uint32_t m = 1; m < num_to_sample; ++m) {
            if (current_chosen_local < 0 ||
                current_chosen_local >= static_cast<IndexType>(num_points)) {
                return;
            }

            IndexType current_storage_idx =
                padded_point_start + current_chosen_local;

            PreloadCenter(current_storage_idx);

            float global_max_dist = NEG_INF;
            uint32_t best_offset = 0;
            uint32_t best_actual_tile = 0;

            uint32_t tile_id = 0;
            for (uint32_t offset = 0;
                 offset < num_points;
                 offset += TILE_SIZE, ++tile_id) {
                uint32_t db = tile_id & 1;

                uint32_t actual_tile =
                    (num_points - offset < TILE_SIZE) ?
                    (num_points - offset) : TILE_SIZE;

                uint32_t vec_align_len = AlignUp8(actual_tile);

                IndexType storage_offset =
                    padded_point_start + static_cast<IndexType>(offset);

                LocalTensor<float> distTile = distTileBuf[db].Get<float>();

                DataCopy(distTile,
                         distGm[storage_offset],
                         vec_align_len);

                PipeBarrier<PIPE_ALL>();

                if (dim == FAST_DIM_3D) {
                    ComputeDistance3D(db,
                                      current_storage_idx,
                                      storage_offset,
                                      vec_align_len,
                                      actual_tile,
                                      distTile);
                } else {
                    ComputeDistanceGeneric(db,
                                           current_storage_idx,
                                           storage_offset,
                                           vec_align_len,
                                           actual_tile,
                                           distTile);
                }

                PipeBarrier<PIPE_ALL>();

                DataCopy(distGm[storage_offset],
                         distTile,
                         vec_align_len);

                PipeBarrier<PIPE_ALL>();

                LocalTensor<float> tileMax = tileMaxBuf[db].Get<float>();
                LocalTensor<float> reduceWork = reduceWorkBuf[db].Get<float>();

                AscendC::ReduceMax<float>(
                    tileMax,
                    distTile,
                    reduceWork,
                    static_cast<int32_t>(actual_tile),
                    false);

                PipeBarrier<PIPE_ALL>();

                float tile_max_v = tileMax.GetValue(0);

                if (tile_max_v > global_max_dist) {
                    global_max_dist = tile_max_v;
                    best_offset = offset;
                    best_actual_tile = actual_tile;
                }
            }

            if (best_actual_tile == 0) {
                return;
            }

            uint32_t best_i =
                FindBestIndexInTile(best_offset, best_actual_tile);

            current_chosen_local =
                static_cast<IndexType>(best_offset) +
                static_cast<IndexType>(best_i);

            if (current_chosen_local < 0 ||
                current_chosen_local >= static_cast<IndexType>(num_points)) {
                return;
            }

            current_chosen_global = batch_start_idx + current_chosen_local;

            StoreOutputToUb(outUb,
                            current_chosen_global,
                            out_cache_count,
                            out_flush_offset);
        }

        FlushOutputTail(outUb,
                        out_cache_count,
                        out_flush_offset);

        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitDistWorkspaceParallel() {
        uint32_t tile_id = 0;

        for (uint32_t offset = 0;
             offset < num_points;
             offset += TILE_SIZE, ++tile_id) {
            uint32_t db = tile_id & 1;

            uint32_t actual_tile =
                (num_points - offset < TILE_SIZE) ?
                (num_points - offset) : TILE_SIZE;

            uint32_t vec_align_len = AlignUp8(actual_tile);

            LocalTensor<float> distTile = distTileBuf[db].Get<float>();
            Duplicate(distTile, INIT_DIST, vec_align_len);

            PipeBarrier<PIPE_ALL>();

            IndexType storage_offset =
                padded_point_start + static_cast<IndexType>(offset);

            DataCopy(distGm[storage_offset],
                     distTile,
                     vec_align_len);

            PipeBarrier<PIPE_ALL>();
        }
    }

    // ============================================================
    // mode 1: inner-40 path
    // ============================================================

    __aicore__ inline void ProcessInner40Mode() {
        // 所有 40 个 block 必须执行完全一致的 SyncAll 序列。
        for (uint32_t batch_id = 0; batch_id < batch_size; ++batch_id) {
            LoadBatchMeta(batch_id);
            ProcessOneBatchInner40(batch_id);
        }
    }

    __aicore__ inline void ProcessOneBatchInner40(uint32_t batch_id) {
        if (!CheckBatchValid()) {
            return;
        }

        LocalTensor<IndexType> outUb = outUbBuf.Get<IndexType>();

        uint32_t out_cache_count = 0;
        uint32_t out_flush_offset = 0;

        InitDistWorkspaceShard();
        SyncAll40();

        IndexType first_pt_local = raw_start_ptr[batch_id];

        if (first_pt_local < 0 ||
            first_pt_local >= static_cast<IndexType>(num_points)) {
            return;
        }

        if (block_id == 0) {
            WriteChosenLocal(first_pt_local);

            IndexType first_global = batch_start_idx + first_pt_local;
            StoreOutputToUb(outUb,
                            first_global,
                            out_cache_count,
                            out_flush_offset);
        }

        SyncAll40();

        for (uint32_t m = 1; m < num_to_sample; ++m) {
            IndexType current_chosen_local = ReadChosenLocal();

            if (current_chosen_local < 0 ||
                current_chosen_local >= static_cast<IndexType>(num_points)) {
                current_chosen_local = 0;
            }

            IndexType current_storage_idx =
                padded_point_start + current_chosen_local;

            // 每个 sample 每个 block 只预加载一次 center。
            PreloadCenter(current_storage_idx);

            ComputeShardLocalMax(current_storage_idx);

            SyncAll40();

            if (block_id == 0) {
                IndexType next_local = ReduceGlobalBestFromLocalMaxBulk();

                if (next_local < 0 ||
                    next_local >= static_cast<IndexType>(num_points)) {
                    next_local = 0;
                }

                WriteChosenLocal(next_local);

                IndexType next_global = batch_start_idx + next_local;
                StoreOutputToUb(outUb,
                                next_global,
                                out_cache_count,
                                out_flush_offset);
            }

            SyncAll40();
        }

        if (block_id == 0) {
            FlushOutputTail(outUb,
                            out_cache_count,
                            out_flush_offset);
        }

        SyncAll40();
    }

    __aicore__ inline void InitDistWorkspaceShard() {
        uint32_t tile_id = 0;

        for (uint32_t offset = block_id * TILE_SIZE;
             offset < num_points;
             offset += FPS_FIXED_BLOCK_DIM * TILE_SIZE, ++tile_id) {
            uint32_t db = tile_id & 1;

            uint32_t actual_tile =
                (num_points - offset < TILE_SIZE) ?
                (num_points - offset) : TILE_SIZE;

            uint32_t vec_align_len = AlignUp8(actual_tile);

            LocalTensor<float> distTile = distTileBuf[db].Get<float>();
            Duplicate(distTile, INIT_DIST, vec_align_len);

            PipeBarrier<PIPE_ALL>();

            IndexType storage_offset =
                padded_point_start + static_cast<IndexType>(offset);

            DataCopy(distGm[storage_offset],
                     distTile,
                     vec_align_len);

            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ComputeShardLocalMax(IndexType current_storage_idx) {
        float local_best_v = NEG_INF;
        uint32_t local_best_offset = 0;
        uint32_t local_best_actual_tile = 0;

        uint32_t tile_id = 0;

        for (uint32_t offset = block_id * TILE_SIZE;
             offset < num_points;
             offset += FPS_FIXED_BLOCK_DIM * TILE_SIZE, ++tile_id) {
            uint32_t db = tile_id & 1;

            uint32_t actual_tile =
                (num_points - offset < TILE_SIZE) ?
                (num_points - offset) : TILE_SIZE;

            uint32_t vec_align_len = AlignUp8(actual_tile);

            IndexType storage_offset =
                padded_point_start + static_cast<IndexType>(offset);

            LocalTensor<float> distTile = distTileBuf[db].Get<float>();

            DataCopy(distTile,
                     distGm[storage_offset],
                     vec_align_len);

            PipeBarrier<PIPE_ALL>();

            if (dim == FAST_DIM_3D) {
                ComputeDistance3D(db,
                                  current_storage_idx,
                                  storage_offset,
                                  vec_align_len,
                                  actual_tile,
                                  distTile);
            } else {
                ComputeDistanceGeneric(db,
                                       current_storage_idx,
                                       storage_offset,
                                       vec_align_len,
                                       actual_tile,
                                       distTile);
            }

            PipeBarrier<PIPE_ALL>();

            DataCopy(distGm[storage_offset],
                     distTile,
                     vec_align_len);

            PipeBarrier<PIPE_ALL>();

            LocalTensor<float> tileMax = tileMaxBuf[db].Get<float>();
            LocalTensor<float> reduceWork = reduceWorkBuf[db].Get<float>();

            AscendC::ReduceMax<float>(
                tileMax,
                distTile,
                reduceWork,
                static_cast<int32_t>(actual_tile),
                false);

            PipeBarrier<PIPE_ALL>();

            float tile_max_v = tileMax.GetValue(0);

            if (tile_max_v > local_best_v) {
                local_best_v = tile_max_v;
                local_best_offset = offset;
                local_best_actual_tile = actual_tile;
            }
        }

        if (local_best_actual_tile == 0) {
            WriteLocalMaxSlot(NEG_INF, 0);
            return;
        }

        uint32_t best_i =
            FindBestIndexInTile(local_best_offset, local_best_actual_tile);

        IndexType local_idx =
            static_cast<IndexType>(local_best_offset) +
            static_cast<IndexType>(best_i);

        WriteLocalMaxSlot(local_best_v, local_idx);
    }

    // ============================================================
    // cross-core communication: 32B DataCopy slots
    // ============================================================

    __aicore__ inline void WriteLocalMaxSlot(float value, IndexType idx) {
        LocalTensor<float> valSlot = localValSlotBuf.Get<float>();
        LocalTensor<IndexType> idxSlot = localIdxSlotBuf.Get<IndexType>();

        Duplicate(valSlot, NEG_INF, LOCAL_VAL_SLOT_FLOAT);

        for (uint32_t i = 0; i < LOCAL_IDX_SLOT_INT64; ++i) {
            idxSlot.SetValue(i, idx);
        }

        PipeBarrier<PIPE_ALL>();

        valSlot.SetValue(0, value);

        PipeBarrier<PIPE_ALL>();

        DataCopy(localMaxValGm[block_id * LOCAL_VAL_SLOT_FLOAT],
                 valSlot,
                 LOCAL_VAL_SLOT_FLOAT);

        DataCopy(localMaxIdxGm[block_id * LOCAL_IDX_SLOT_INT64],
                 idxSlot,
                 LOCAL_IDX_SLOT_INT64);

        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline IndexType ReduceGlobalBestFromLocalMaxBulk() {
        LocalTensor<float> allVal = allLocalValBuf.Get<float>();
        LocalTensor<IndexType> allIdx = allLocalIdxBuf.Get<IndexType>();

        // 优化点 2：
        // block0 原来每轮做 40 次 val slot DataCopy + 40 次 idx slot DataCopy。
        // 现在改为 2 次大 DataCopy。
        DataCopy(allVal,
                 localMaxValGm[0],
                 LOCAL_VAL_TOTAL_FLOAT);

        DataCopy(allIdx,
                 localMaxIdxGm[0],
                 LOCAL_IDX_TOTAL_INT64);

        PipeBarrier<PIPE_ALL>();

        float best_v = NEG_INF;
        IndexType best_idx = 0;
        bool has_value = false;

        for (uint32_t i = 0; i < FPS_FIXED_BLOCK_DIM; ++i) {
            float v = allVal.GetValue(i * LOCAL_VAL_SLOT_FLOAT);
            IndexType idx = allIdx.GetValue(i * LOCAL_IDX_SLOT_INT64);

            if (v < 0.0f) {
                continue;
            }

            if (!has_value ||
                v > best_v ||
                (v == best_v && idx < best_idx)) {
                best_v = v;
                best_idx = idx;
                has_value = true;
            }
        }

        if (!has_value) {
            return 0;
        }

        return best_idx;
    }

    __aicore__ inline void WriteChosenLocal(IndexType idx) {
        LocalTensor<IndexType> chosenSlot = chosenSlotBuf.Get<IndexType>();

        for (uint32_t i = 0; i < CHOSEN_SLOT_INT64; ++i) {
            chosenSlot.SetValue(i, idx);
        }

        PipeBarrier<PIPE_ALL>();

        DataCopy(chosenLocalGm[0],
                 chosenSlot,
                 CHOSEN_SLOT_INT64);

        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline IndexType ReadChosenLocal() {
        LocalTensor<IndexType> chosenSlot = chosenSlotBuf.Get<IndexType>();

        DataCopy(chosenSlot,
                 chosenLocalGm[0],
                 CHOSEN_SLOT_INT64);

        PipeBarrier<PIPE_ALL>();

        return chosenSlot.GetValue(0);
    }

    // ============================================================
    // distance compute
    // ============================================================

    __aicore__ inline void ComputeDistance3D(uint32_t db,
                                             IndexType current_storage_idx,
                                             IndexType storage_offset,
                                             uint32_t vec_align_len,
                                             uint32_t actual_tile,
                                             LocalTensor<float>& distTile) {
        float cx = GetCenter(0, current_storage_idx);
        float cy = GetCenter(1, current_storage_idx);
        float cz = GetCenter(2, current_storage_idx);

        LocalTensor<float> xComp = bufX[db].Get<float>();
        LocalTensor<float> yComp = bufY[db].Get<float>();
        LocalTensor<float> zComp = bufZ[db].Get<float>();
        LocalTensor<float> sumBuf = sqSum[db].Get<float>();

        DataCopy(xComp,
                 srcGm[0 * total_points + storage_offset],
                 vec_align_len);

        DataCopy(yComp,
                 srcGm[1 * total_points + storage_offset],
                 vec_align_len);

        DataCopy(zComp,
                 srcGm[2 * total_points + storage_offset],
                 vec_align_len);

        PipeBarrier<PIPE_ALL>();

        Adds(xComp, xComp, -cx, vec_align_len);
        Mul(sumBuf, xComp, xComp, vec_align_len);

        Adds(yComp, yComp, -cy, vec_align_len);
        Mul(yComp, yComp, yComp, vec_align_len);
        Add(sumBuf, sumBuf, yComp, vec_align_len);

        Adds(zComp, zComp, -cz, vec_align_len);
        Mul(zComp, zComp, zComp, vec_align_len);
        Add(sumBuf, sumBuf, zComp, vec_align_len);

        PipeBarrier<PIPE_ALL>();

        for (uint32_t i = actual_tile; i < vec_align_len; ++i) {
            sumBuf.SetValue(i, INIT_DIST);
        }

        PipeBarrier<PIPE_ALL>();

        Min(distTile, distTile, sumBuf, vec_align_len);
    }

    __aicore__ inline void ComputeDistanceGeneric(uint32_t db,
                                                  IndexType current_storage_idx,
                                                  IndexType storage_offset,
                                                  uint32_t vec_align_len,
                                                  uint32_t actual_tile,
                                                  LocalTensor<float>& distTile) {
        LocalTensor<float> axisBuf = bufX[db].Get<float>();
        LocalTensor<float> tmpBuf = bufY[db].Get<float>();
        LocalTensor<float> sumBuf = sqSum[db].Get<float>();

        Duplicate(sumBuf, 0.0f, vec_align_len);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t axis = 0; axis < dim; ++axis) {
            float center = GetCenter(axis, current_storage_idx);

            DataCopy(axisBuf,
                     srcGm[static_cast<IndexType>(axis) * total_points + storage_offset],
                     vec_align_len);

            PipeBarrier<PIPE_ALL>();

            Adds(axisBuf, axisBuf, -center, vec_align_len);
            Mul(tmpBuf, axisBuf, axisBuf, vec_align_len);
            Add(sumBuf, sumBuf, tmpBuf, vec_align_len);

            PipeBarrier<PIPE_ALL>();
        }

        for (uint32_t i = actual_tile; i < vec_align_len; ++i) {
            sumBuf.SetValue(i, INIT_DIST);
        }

        PipeBarrier<PIPE_ALL>();

        Min(distTile, distTile, sumBuf, vec_align_len);
    }

    __aicore__ inline uint32_t FindBestIndexInTile(uint32_t best_offset,
                                                   uint32_t best_actual_tile) {
        uint32_t best_i = 0;
        float best_v = NEG_INF;

        IndexType best_storage_offset =
            padded_point_start + static_cast<IndexType>(best_offset);

        LocalTensor<float> distTile = distTileBuf[0].Get<float>();
        uint32_t best_vec_align_len = AlignUp8(best_actual_tile);

        DataCopy(distTile,
                 distGm[best_storage_offset],
                 best_vec_align_len);

        PipeBarrier<PIPE_ALL>();

        for (uint32_t i = 0; i < best_actual_tile; ++i) {
            float v = distTile.GetValue(i);
            if (v > best_v) {
                best_v = v;
                best_i = i;
            }
        }

        return best_i;
    }

    // ============================================================
    // output
    // ============================================================

    __aicore__ inline void StoreOutputToUb(LocalTensor<IndexType>& outUb,
                                           IndexType value,
                                           uint32_t& out_cache_count,
                                           uint32_t& out_flush_offset) {
        outUb.SetValue(out_cache_count, value);
        out_cache_count += 1;

        if (out_cache_count == OUT_TILE_SIZE) {
            PipeBarrier<PIPE_ALL>();

            DataCopy(outGm[padded_out_start_pos + out_flush_offset],
                     outUb,
                     OUT_TILE_SIZE);

            PipeBarrier<PIPE_ALL>();

            out_flush_offset += OUT_TILE_SIZE;
            out_cache_count = 0;
        }
    }

    __aicore__ inline void FlushOutputTail(LocalTensor<IndexType>& outUb,
                                           uint32_t& out_cache_count,
                                           uint32_t& out_flush_offset) {
        if (out_cache_count == 0) {
            return;
        }

        uint32_t copy_count = AlignUp4(out_cache_count);

        IndexType last_value = outUb.GetValue(out_cache_count - 1);
        for (uint32_t i = out_cache_count; i < copy_count; ++i) {
            outUb.SetValue(i, last_value);
        }

        PipeBarrier<PIPE_ALL>();

        DataCopy(outGm[padded_out_start_pos + out_flush_offset],
                 outUb,
                 copy_count);

        PipeBarrier<PIPE_ALL>();

        out_flush_offset += copy_count;
        out_cache_count = 0;
    }

private:
    TPipe pipe;

    TBuf<QuePosition::VECCALC> bufX[2];
    TBuf<QuePosition::VECCALC> bufY[2];
    TBuf<QuePosition::VECCALC> bufZ[2];
    TBuf<QuePosition::VECCALC> sqSum[2];
    TBuf<QuePosition::VECCALC> distTileBuf[2];

    TBuf<QuePosition::VECCALC> tileMaxBuf[2];
    TBuf<QuePosition::VECCALC> reduceWorkBuf[2];

    TBuf<QuePosition::VECCALC> outUbBuf;
    TBuf<QuePosition::VECCALC> scalarBuf;
    TBuf<QuePosition::VECCALC> centerBuf;

    TBuf<QuePosition::VECCALC> localValSlotBuf;
    TBuf<QuePosition::VECCALC> localIdxSlotBuf;
    TBuf<QuePosition::VECCALC> chosenSlotBuf;

    TBuf<QuePosition::VECCALC> allLocalValBuf;
    TBuf<QuePosition::VECCALC> allLocalIdxBuf;

    TBuf<QuePosition::VECCALC> syncUbBuf;

    GlobalTensor<float> srcGm;
    GlobalTensor<float> distGm;
    GlobalTensor<IndexType> outGm;

    GlobalTensor<float> localMaxValGm;
    GlobalTensor<IndexType> localMaxIdxGm;
    GlobalTensor<IndexType> chosenLocalGm;

    GlobalTensor<int32_t> syncGm;

    __gm__ IndexType* ptr_gm;
    __gm__ IndexType* padded_point_ptr_gm;
    __gm__ IndexType* out_ptr_gm;
    __gm__ IndexType* padded_out_ptr_gm;
    __gm__ IndexType* raw_start_ptr;
    __gm__ int32_t* mode_ptr;

    uint32_t dim;
    uint32_t total_points;
    uint32_t batch_size;
    uint32_t block_id;

    bool center_preloaded;

    uint32_t num_points;
    uint32_t padded_point_num;

    uint32_t num_to_sample;
    uint32_t padded_num_to_sample;

    IndexType batch_start_idx;
    IndexType batch_end_idx;

    IndexType padded_point_start;
    IndexType padded_point_end;

    IndexType compact_out_start_pos;
    IndexType compact_out_end_pos;

    IndexType padded_out_start_pos;
    IndexType padded_out_end_pos;
};

extern "C" __global__ __aicore__ void farthest_point_sampling(
    GM_ADDR src,
    GM_ADDR ptr,
    GM_ADDR padded_point_ptr,
    GM_ADDR out_ptr,
    GM_ADDR padded_out_ptr,
    GM_ADDR start,
    GM_ADDR mode,
    GM_ADDR dist,
    GM_ADDR local_max_val,
    GM_ADDR local_max_idx,
    GM_ADDR chosen_local,
    GM_ADDR sync_workspace,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    KernelFPS op;
    op.Init(src,
            ptr,
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
            out,
            tiling);
    op.Process();
}