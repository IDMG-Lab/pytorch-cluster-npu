#include "kernel_operator.h"
using namespace AscendC;

// BUFFER_POINTS = 4096
// UB 占用: 4096 × 36B ≈ 144KB (56% of 256KB on 910B)
// 相比 64 点 (0.9% UB), 循环次数减少 64x, 显著降低循环开销
constexpr uint32_t BUFFER_POINTS = 4096;
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
        dim = tilingData->dim;
        workPoints = tilingData->numPoints;

        if (workPoints == 0) {
            coreOffset = 0;
            corePoints = 0;
            return;
        }

        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();

        uint32_t pointsPerCore = tilingData->blockPoints;
        uint32_t tailPoints = tilingData->tailPoints;

        coreOffset = blockIdx * pointsPerCore;
        corePoints = pointsPerCore;

        if (blockIdx == blockNum - 1) {
            corePoints += tailPoints;
        }

        // SoA: pos shape = [D, N]
        posGm.SetGlobalBuffer((__gm__ float*)pos);
        sizeGm.SetGlobalBuffer((__gm__ float*)size);
        startGm.SetGlobalBuffer((__gm__ float*)start);
        endGm.SetGlobalBuffer((__gm__ float*)end);
        clusterGm.SetGlobalBuffer((__gm__ int64_t*)cluster);

        pipe.InitBuffer(inQue, BUFFER_NUM, BUFFER_POINTS * sizeof(float));
        pipe.InitBuffer(outQue, BUFFER_NUM, BUFFER_POINTS * sizeof(int32_t));

        pipe.InitBuffer(tmpBuf, BUFFER_POINTS * sizeof(float));
        pipe.InitBuffer(gridI32Buf, BUFFER_POINTS * sizeof(int32_t));
        pipe.InitBuffer(strideBuf, BUFFER_POINTS * sizeof(int32_t));
        pipe.InitBuffer(castBuf, BUFFER_POINTS * sizeof(int64_t));
    }

    __aicore__ inline void Process() {
        if (corePoints == 0) {
            return;
        }

        uint32_t loop = (corePoints + BUFFER_POINTS - 1) / BUFFER_POINTS;

        for (uint32_t t = 0; t < loop; ++t) {
            uint32_t pointOffset = coreOffset + t * BUFFER_POINTS;
            uint32_t count = GetCount(pointOffset);

            LocalTensor<int32_t> clusterLocal = outQue.AllocTensor<int32_t>();

            Duplicate<int32_t>(clusterLocal, 0, count);

            int32_t stride = 1;

            for (uint32_t d = 0; d < dim; ++d) {
                CopyIn(pointOffset, count, d);
                Compute(count, d, stride, clusterLocal);

                float s = sizeGm.GetValue(d);
                float st = startGm.GetValue(d);
                float e = endGm.GetValue(d);

                int32_t gridSize =
                    static_cast<int32_t>((e - st) / s) + 1;

                stride *= gridSize;
            }

            outQue.EnQue(clusterLocal);
            CopyOut(pointOffset, count);
        }
    }

private:
    __aicore__ inline uint32_t GetCount(uint32_t offset) {
        uint32_t localOffset = offset - coreOffset;
        uint32_t remain = corePoints - localOffset;
        return remain > BUFFER_POINTS ? BUFFER_POINTS : remain;
    }

    __aicore__ inline void CopyIn(
        uint32_t offset,
        uint32_t count,
        uint32_t d) {
        LocalTensor<float> posLocal = inQue.AllocTensor<float>();

        // SoA contiguous read:
        // pos[d, offset : offset + count]
        uint32_t gmOffset = d * workPoints + offset;

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};

        DataCopyPadExtParams<float> padParams{true, 0, 0, 0};

        DataCopyPad(posLocal, posGm[gmOffset], copyParams, padParams);

        inQue.EnQue(posLocal);
    }

    __aicore__ inline void Compute(
        uint32_t count,
        uint32_t d,
        int32_t stride,
        LocalTensor<int32_t>& clusterLocal) {
        LocalTensor<float> posLocal = inQue.DeQue<float>();

        LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

        LocalTensor<int32_t> gridLocal = gridI32Buf.Get<int32_t>();

        LocalTensor<int32_t> strideLocal = strideBuf.Get<int32_t>();

        float startVal = startGm.GetValue(d);
        float sizeVal = sizeGm.GetValue(d);

        Adds(tmpLocal, posLocal, -startVal, count);
        Muls(tmpLocal, tmpLocal, 1.0f / sizeVal, count);

        Cast(gridLocal, tmpLocal, RoundMode::CAST_FLOOR, count);

        Duplicate<int32_t>(strideLocal, stride, count);

        Mul(gridLocal, gridLocal, strideLocal, count);

        Add(clusterLocal, clusterLocal, gridLocal, count);

        inQue.FreeTensor(posLocal);
    }

    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t count) {
        LocalTensor<int32_t> outLocal32 = outQue.DeQue<int32_t>();

        LocalTensor<int64_t> outLocal64 = castBuf.Get<int64_t>();

        Cast(outLocal64, outLocal32, RoundMode::CAST_NONE, count);

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(int64_t)), 0, 0, 0};

        DataCopyPad(clusterGm[offset], outLocal64, copyParams);

        outQue.FreeTensor(outLocal32);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;

    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> gridI32Buf;
    TBuf<QuePosition::VECCALC> strideBuf;
    TBuf<QuePosition::VECCALC> castBuf;

    GlobalTensor<float> posGm;
    GlobalTensor<float> sizeGm;
    GlobalTensor<float> startGm;
    GlobalTensor<float> endGm;
    GlobalTensor<int64_t> clusterGm;

    uint32_t workPoints = 0;
    uint32_t coreOffset = 0;
    uint32_t corePoints = 0;
    uint32_t dim = 0;
};

extern "C" __global__ __aicore__ void voxel_grid(
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