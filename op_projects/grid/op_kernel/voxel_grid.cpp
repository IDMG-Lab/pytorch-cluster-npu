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
        GM_ADDR end,
        GM_ADDR cluster,
        const VoxelGridTilingData* tilingData) {
        // =========================
        // 基本信息
        // =========================
        this->dim = tilingData->dim;
        this->workPoints =
            tilingData->numPoints;
        if (this->workPoints == 0) {
            return;
        }

        // =========================
        // 多核切分
        // =========================
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        uint32_t pointsPerCore = tilingData->blockPoints;
        uint32_t tailPoints = tilingData->tailPoints;

        // 当前Core起始点
        coreOffset = blockIdx * pointsPerCore;

        // 当前Core处理点数
        corePoints = pointsPerCore;

        // 最后一个Core处理tail
        if (blockIdx == blockNum - 1) {
            corePoints += tailPoints;
        }

        // =========================
        // GM绑定
        // =========================
        posGm.SetGlobalBuffer((__gm__ float*)pos);
        clusterGm.SetGlobalBuffer((__gm__ int64_t*)cluster);
        sizeGm.SetGlobalBuffer((__gm__ float*)size);
        startGm.SetGlobalBuffer((__gm__ float*)start);
        endGm.SetGlobalBuffer((__gm__ float*)end);

        // =========================
        // Queue Buffer
        // =========================
        // 输入:
        // BUFFER_POINTS 个点
        // 每个点 dim 个float
        pipe.InitBuffer(inQue, BUFFER_NUM, BUFFER_POINTS * dim * sizeof(float));

        // 输出:
        // BUFFER_POINTS 个 int64
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
                uint32_t pointOffset = coreOffset + i * BUFFER_POINTS;
                uint32_t count = GetCount(pointOffset);
                CopyIn(pointOffset, count);
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

        uint32_t copyCount = count * dim;

        // 使用DataCopyPad进行非32B对齐搬运
        // burstLen单位是Bytes
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyCount * sizeof(float)), 0, 0, 0};

        // pad到32B对齐
        // float类型32B = 8个float
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0};

        DataCopyPad(posLocal, posGm[offset * dim], copyParams, padParams);

        inQue.EnQue(posLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        LocalTensor<float> posLocal = inQue.DeQue<float>();

        LocalTensor<int64_t> clusterLocal = outQue.AllocTensor<int64_t>();

        for (uint32_t i = 0; i < count; i++) {
            int64_t cluster = 0;

            int64_t stride = 1;

            for (uint32_t d = 0; d < dim; d++) {
                uint32_t idx = i * dim + d;

                float val = posLocal.GetValue(idx);
                float sizeVal = sizeGm.GetValue(d);
                float startVal = startGm.GetValue(d);
                float endVal = endGm.GetValue(d);

                // CUDA一致
                float coord = (val - startVal) / sizeVal;
                int64_t grid = static_cast<int64_t>(coord);
                if (coord < 0 && coord != grid) {
                    grid -= 1;
                }

                cluster += grid * stride;

                int64_t gridSize = static_cast<int64_t>((endVal - startVal) / sizeVal) + 1;

                stride *= gridSize;
            }
            clusterLocal.SetValue(i, cluster);
        }
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
    uint32_t workPoints;
    uint32_t coreOffset;
    uint32_t corePoints;
    GlobalTensor<float> sizeGm;
    GlobalTensor<float> startGm;
    GlobalTensor<float> endGm;
    uint32_t dim;
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
    op.Init(pos, size, start, end, cluster, &tiling_data);
    op.Process();
}