#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t BUFFER_POINTS = 64;

class KernelVoxelGrid {
public:
    __aicore__ inline KernelVoxelGrid() {}

    __aicore__ inline void Init(
        GM_ADDR pos,
        GM_ADDR size,
        GM_ADDR start,
        GM_ADDR cluster,
        const VoxelGridTilingData* tilingData) {
        // 单核模式：不做 core 切分
        this->workPoints = tilingData->numPoints;

        if (this->workPoints == 0)
            return;

        // GM 绑定（1D）
        posGm.SetGlobalBuffer((__gm__ float*)pos);
        clusterGm.SetGlobalBuffer((__gm__ int64_t*)cluster);

        // 标量
        GlobalTensor<float> sizeGmPtr;
        sizeGmPtr.SetGlobalBuffer((__gm__ float*)size);
        valSize = sizeGmPtr.GetValue(0);

        if (start != nullptr) {
            GlobalTensor<float> startGmPtr;
            startGmPtr.SetGlobalBuffer((__gm__ float*)start);
            valStart = startGmPtr.GetValue(0);
        } else {
            valStart = 0.0f;
        }

        // UB buffer（只用 float + int64）
        pipe.InitBuffer(inQuePos, 1, BUFFER_POINTS * sizeof(float));
        pipe.InitBuffer(outQueCluster, 1, BUFFER_POINTS * sizeof(int64_t));
    }

    __aicore__ inline void Process() {
        if (workPoints == 0)
            return;

        uint32_t loop = workPoints / BUFFER_POINTS;
        uint32_t tail = workPoints % BUFFER_POINTS;

        uint32_t offset = 0;

        for (uint32_t i = 0; i < loop; i++) {
            Compute(BUFFER_POINTS, offset);
            offset += BUFFER_POINTS;
        }

        if (tail > 0) {
            Compute(tail, offset);
        }
    }

private:
    __aicore__ inline void Compute(uint32_t count, uint32_t offset) {
        LocalTensor<float> posLocal = inQuePos.AllocTensor<float>();
        LocalTensor<int64_t> clusterLocal = outQueCluster.AllocTensor<int64_t>();

        // GM → UB
        DataCopy(posLocal, posGm[offset], count);
        inQuePos.EnQue(posLocal);

        posLocal = inQuePos.DeQue<float>();

        for (uint32_t i = 0; i < count; i++) {
            float x = posLocal.GetValue(i);

            float v = (x - valStart) / valSize;

            int64_t idx = (int64_t)v;

            clusterLocal.SetValue(i, idx);
        }

        outQueCluster.EnQue(clusterLocal);

        LocalTensor<int64_t> outLocal = outQueCluster.DeQue<int64_t>();

        DataCopy(clusterGm[offset], outLocal, count);

        inQuePos.FreeTensor(posLocal);
        outQueCluster.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQuePos;
    TQue<QuePosition::VECOUT, 1> outQueCluster;

    GlobalTensor<float> posGm;
    GlobalTensor<int64_t> clusterGm;

    uint32_t workPoints;
    float valSize;
    float valStart;
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
    op.Init(pos, size, start, cluster, &tiling_data);
    op.Process();
}