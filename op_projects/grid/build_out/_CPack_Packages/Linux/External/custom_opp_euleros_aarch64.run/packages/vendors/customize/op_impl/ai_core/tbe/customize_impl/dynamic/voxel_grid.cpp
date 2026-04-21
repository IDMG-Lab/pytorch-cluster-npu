#include "kernel_operator.h"
using namespace AscendC;

// 1. 必须考虑对齐：float 是 4B，int64 是 8B。
// Ascend C 的 DataCopy 建议 32B 对齐。128 个点（float）是 512B，满足对齐要求。
constexpr uint32_t BUFFER_POINTS = 128;

class KernelVoxelGrid {
public:
    __aicore__ inline KernelVoxelGrid() {}
    __aicore__ inline void Init(GM_ADDR pos, GM_ADDR size, GM_ADDR start, GM_ADDR cluster, const VoxelGridTilingData* tilingData) {
        uint32_t coreIdx = GetBlockIdx();

        // 强制统一为一维逻辑
        this->dim = 1;
        uint32_t startOffset = coreIdx * tilingData->blockPoints;
        this->workPoints = (coreIdx == 7) ? (tilingData->blockPoints + tilingData->tailPoints) : tilingData->blockPoints;

        // 设置 Global Buffer 偏移
        posGm.SetGlobalBuffer((__gm__ float*)pos + startOffset);
        sizeGm.SetGlobalBuffer((__gm__ float*)size);
        startGm.SetGlobalBuffer((__gm__ float*)start);
        clusterGm.SetGlobalBuffer((__gm__ int64_t*)cluster + startOffset);

        // 初始化 Pipe 内存池
        // Que 深度通常设为 2（Double Buffer）以实现搬运和计算并行，这里先用 1 演示
        pipe.InitBuffer(inQuePos, 1, BUFFER_POINTS * sizeof(float));
        pipe.InitBuffer(outQueCluster, 1, BUFFER_POINTS * sizeof(int64_t));

        // 申请一个临时空间用于存放中间计算结果 (float32)
        pipe.InitBuffer(tmpBuf, BUFFER_POINTS * sizeof(float));
    }

    __aicore__ inline void Process() {
        uint32_t loopCount = workPoints / BUFFER_POINTS;
        uint32_t lastPoints = workPoints % BUFFER_POINTS;

        uint32_t i = 0;
        for (; i < loopCount; i++) {
            CopyIn(BUFFER_POINTS, i * BUFFER_POINTS);
            Compute(BUFFER_POINTS);
            CopyOut(BUFFER_POINTS, i * BUFFER_POINTS);
        }
        if (lastPoints > 0) {
            // 注意：尾部处理在实际中也要对齐，此处假设 tiling 保证了 count 是 8 的倍数（对齐 32 字节）
            CopyIn(lastPoints, i * BUFFER_POINTS);
            Compute(lastPoints);
            CopyOut(lastPoints, i * BUFFER_POINTS);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t count, uint32_t offset) {
        LocalTensor<float> posLocal = inQuePos.AllocTensor<float>();
        // 将数据从 GM 搬运到 UB
        DataCopy(posLocal, posGm[offset], count);
        inQuePos.EnQue(posLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        LocalTensor<float> posLocal = inQuePos.DeQue<float>();
        LocalTensor<int64_t> clusterLocal = outQueCluster.AllocTensor<int64_t>();
        LocalTensor<float> tempLocal = tmpBuf.Get<float>();

        // 从 GM 获取标量参数（这里可以使用 Scalar 类型或直接 GetValue）
        float sVal = startGm.GetValue(0);
        float szVal = sizeGm.GetValue(0);

        // --- 向量化计算开始 ---
        // 1. temp = pos - start
        Adds(tempLocal, posLocal, -sVal, count);

        // 2. temp = temp / size (使用倒数乘法：temp * (1/size))
        float invSize = 1.0f / szVal;
        Muls(tempLocal, tempLocal, invSize, count);

        // 3. 类型转换：float32 -> int64_t 并取整
        // 注意：某些架构不支持直接跨字节 Cast，可能需要 float32 -> int32 -> int64
        Cast(clusterLocal, tempLocal, RoundMode::CAST_FLOOR, count);
        // --- 向量化计算结束 ---

        outQueCluster.EnQue(clusterLocal);
        inQuePos.FreeTensor(posLocal);
    }

    __aicore__ inline void CopyOut(uint32_t count, uint32_t offset) {
        LocalTensor<int64_t> clusterLocal = outQueCluster.DeQue<int64_t>();
        // 将结果从 UB 搬运回 GM
        DataCopy(clusterGm[offset], clusterLocal, count);
        outQueCluster.FreeTensor(clusterLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQuePos;
    TQue<QuePosition::VECOUT, 1> outQueCluster;
    TBuf<QuePosition::VECCALC> tmpBuf;

    GlobalTensor<float> posGm, sizeGm, startGm;
    GlobalTensor<int64_t> clusterGm;
    uint32_t dim, workPoints;
};

extern "C" __global__ __aicore__ void voxel_grid(GM_ADDR pos, GM_ADDR size, GM_ADDR start, GM_ADDR end, GM_ADDR cluster, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelVoxelGrid op;
    op.Init(pos, size, start, cluster, &tiling_data);
    op.Process();
}