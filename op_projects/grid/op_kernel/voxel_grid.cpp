#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t BUFFER_POINTS = 64;
constexpr uint32_t BUFFER_NUM = 2;

class KernelVoxelGrid {
public:
    __aicore__ inline KernelVoxelGrid() {}

    __aicore__ inline void Init(
        GM_ADDR pos,
        GM_ADDR size,
        GM_ADDR start,
        GM_ADDR cluster,
        const VoxelGridTilingData* tilingData) {
        this->workPoints = tilingData->numPoints;
        if (this->workPoints == 0)
            return;

        // 多核信息
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        uint32_t pointsPerCore = tilingData->blockPoints;
        uint32_t tailPoints = tilingData->tailPoints;

        // 当前Core负责的起始位置
        coreOffset = blockIdx * pointsPerCore;
        // 当前Core负责的数据量
        corePoints = pointsPerCore;
        // 最后一个Core处理tail
        if (blockIdx == blockNum - 1) {
            corePoints += tailPoints;
        }

        posGm.SetGlobalBuffer((__gm__ float*)pos);
        clusterGm.SetGlobalBuffer((__gm__ int64_t*)cluster);
        sizeGmPtr.SetGlobalBuffer((__gm__ float*)size);
        valSize = sizeGmPtr.GetValue(0);
        valStart = 0.0f;
        if (start != nullptr) {
            startGmPtr.SetGlobalBuffer((__gm__ float*)start);
            valStart = startGmPtr.GetValue(0);
        }

        pipe.InitBuffer(inQue, BUFFER_NUM, BUFFER_POINTS * sizeof(float));
        pipe.InitBuffer(outQue, BUFFER_NUM, BUFFER_POINTS * sizeof(int64_t));
    }

    __aicore__ inline void Process() {
        if (corePoints == 0)
            return;

        // 使用corePoints
        uint32_t loop = (corePoints + BUFFER_POINTS - 1) / BUFFER_POINTS;
        for (int32_t i = 0; i < loop + BUFFER_NUM; i++) {
            // CopyIn
            if (i < loop) {
                // 全局offset
                uint32_t offset = coreOffset + i * BUFFER_POINTS;
                uint32_t count = GetCount(offset);
                CopyIn(offset, count);
            }

            // Compute
            if (i >= 1 && i < loop + 1) {
                uint32_t computeIdx = i - 1;
                // 全局offset
                uint32_t offset = coreOffset + computeIdx * BUFFER_POINTS;
                uint32_t count = GetCount(offset);
                Compute(count);
            }

            // CopyOut
            if (i >= 2) {
                uint32_t outIdx = i - 2;
                // 全局offset
                uint32_t offset = coreOffset + outIdx * BUFFER_POINTS;
                uint32_t count = GetCount(offset);
                CopyOut(offset, count);
            }
        }
    }

private:
    // GetCount适配多核
    __aicore__ inline uint32_t GetCount(uint32_t offset) {
        uint32_t localOffset = offset - coreOffset;
        uint32_t remain = corePoints - localOffset;
        return remain > BUFFER_POINTS ? BUFFER_POINTS : remain;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        LocalTensor<float> posLocal = inQue.AllocTensor<float>();

        // 使用DataCopyPad进行非32B对齐搬运
        // burstLen单位是Bytes
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};

        // pad到32B对齐
        // float类型32B = 8个float
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0};

        DataCopyPad(posLocal, posGm[offset], copyParams, padParams);

        inQue.EnQue(posLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        constexpr uint32_t ALIGN_BYTES = 32;
        uint32_t aligned_bytes = ((count * sizeof(float) + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
        uint32_t alignedCount = aligned_bytes / sizeof(float);

        LocalTensor<float> posLocal = inQue.DeQue<float>();
        LocalTensor<int64_t> clusterLocal = outQue.AllocTensor<int64_t>();

        // (pos - start)
        Adds(posLocal, posLocal, -valStart, alignedCount);
        // / size
        float invSize = 1.0f / valSize;
        Muls(posLocal, posLocal, invSize, alignedCount);
        // floor
        Cast(clusterLocal, posLocal, RoundMode::CAST_FLOOR, alignedCount);

        outQue.EnQue(clusterLocal);
        inQue.FreeTensor(posLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        LocalTensor<int64_t> outLocal = outQue.DeQue<int64_t>();

        // 使用DataCopyPad进行非32B对齐写回
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(int64_t)), 0, 0, 0};

        DataCopyPad(clusterGm[offset], outLocal, copyParams);

        outQue.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECCALC, BUFFER_NUM> tmpQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    GlobalTensor<float> posGm;
    GlobalTensor<int64_t> clusterGm;
    GlobalTensor<float> sizeGmPtr;
    GlobalTensor<float> startGmPtr;
    uint32_t workPoints;
    uint32_t coreOffset;
    uint32_t corePoints;
    float valSize;
    float valStart;
};

extern "C" __global__
    __aicore__ void
    voxel_grid(
        GM_ADDR pos,
        GM_ADDR size,
        GM_ADDR start,
        GM_ADDR end,
        GM_ADDR cluster,
        GM_ADDR workspace,
        GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelVoxelGrid op;
    op.Init(pos, size, start, cluster, &tiling_data);
    op.Process();
}