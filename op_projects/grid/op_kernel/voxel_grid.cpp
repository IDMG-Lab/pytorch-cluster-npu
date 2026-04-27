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
        this->workPoints = tilingData->numPoints;
        if (this->workPoints == 0)
            return;

        posGm.SetGlobalBuffer((__gm__ float*)pos);
        clusterGm.SetGlobalBuffer((__gm__ int64_t*)cluster);

        sizeGmPtr.SetGlobalBuffer((__gm__ float*)size);
        valSize = sizeGmPtr.GetValue(0);

        valStart = 0.0f;
        if (start != nullptr) {
            startGmPtr.SetGlobalBuffer((__gm__ float*)start);
            valStart = startGmPtr.GetValue(0);
        }

        // 双缓冲（为后续 pipeline 做准备）
        pipe.InitBuffer(inQue, 2, BUFFER_POINTS * sizeof(float));
        pipe.InitBuffer(outQue, 2, BUFFER_POINTS * sizeof(int64_t));
        pipe.InitBuffer(tmpQue, 2, BUFFER_POINTS * sizeof(float));
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
        if (offset >= workPoints)
            return;
        if (offset + count > workPoints)
            count = workPoints - offset;

        constexpr uint32_t ALIGN_BYTES = 32;
        constexpr uint32_t ALIGN_FLOATS = ALIGN_BYTES / sizeof(float);  // 8个float
        constexpr uint32_t ALIGN_INTS = ALIGN_BYTES / sizeof(int64_t);  // 4个int64_t

        // 计算对齐后的总大小（用于buffer分配和向量计算）
        uint32_t aligned_bytes = ((count * sizeof(float) + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
        uint32_t alignedCount = aligned_bytes / sizeof(float);

        // 计算对齐拷贝部分
        uint32_t aligned_copy_count_float = (count / ALIGN_FLOATS) * ALIGN_FLOATS;
        uint32_t remainder_float = count % ALIGN_FLOATS;

        uint32_t aligned_copy_count_int64 = (count / ALIGN_INTS) * ALIGN_INTS;
        uint32_t remainder_int64 = count % ALIGN_INTS;

        // 分配buffer
        LocalTensor<float> posLocal = inQue.AllocTensor<float>();
        LocalTensor<float> tmpLocal = tmpQue.AllocTensor<float>();
        LocalTensor<int64_t> clusterLocal = outQue.AllocTensor<int64_t>();

        // 清零整个buffer
        Duplicate(posLocal, 0.0f, alignedCount);
        Duplicate(tmpLocal, 0.0f, alignedCount);

        // ========= 1. 输入：GM -> UB =========
        // 第一段：拷贝对齐的部分
        if (aligned_copy_count_float > 0) {
            DataCopy(posLocal, posGm[offset], aligned_copy_count_float);
        }

        // 第二段：拷贝剩余部分
        if (remainder_float > 0) {
            for (uint32_t i = 0; i < remainder_float; i++) {
                float val = posGm.GetValue(offset + aligned_copy_count_float + i);
                posLocal.SetValue(aligned_copy_count_float + i, val);
            }
        }

        // 等待DMA完成
        if (aligned_copy_count_float > 0) {
            inQue.EnQue(posLocal);
            posLocal = inQue.DeQue<float>();
        }

        // ========= 2. 计算 =========
        Muls(tmpLocal, posLocal, 1.0f, alignedCount);
        Adds(tmpLocal, tmpLocal, -valStart, alignedCount);

        float invSize = 1.0f / valSize;
        Muls(tmpLocal, tmpLocal, invSize, alignedCount);
        Cast(clusterLocal, tmpLocal, RoundMode::CAST_FLOOR, alignedCount);

        // ========= 3. 输出：UB -> GM =========
        outQue.EnQue(clusterLocal);

        // 获取出队指针（等待之前的传输完成）
        LocalTensor<int64_t> outLocal = outQue.DeQue<int64_t>();

        // 第一段：使用DataCopy拷贝对齐的部分
        if (aligned_copy_count_int64 > 0) {
            DataCopy(clusterGm[offset], outLocal, aligned_copy_count_int64);

            // 等待DMA完成后再处理剩余部分
            outQue.EnQue(outLocal);
            outQue.DeQue<int64_t>();
        }

        // 第二段：拷贝剩余部分（使用标量操作）
        if (remainder_int64 > 0) {
            for (uint32_t i = 0; i < remainder_int64; i++) {
                int64_t val = outLocal.GetValue(aligned_copy_count_int64 + i);
                clusterGm.SetValue(offset + aligned_copy_count_int64 + i, val);
            }
        }

        // 释放
        inQue.FreeTensor(posLocal);
        tmpQue.FreeTensor(tmpLocal);
        outQue.FreeTensor(outLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, 2> inQue;
    TQue<QuePosition::VECOUT, 2> outQue;
    TQue<QuePosition::VECCALC, 2> tmpQue;

    GlobalTensor<float> posGm;
    GlobalTensor<int64_t> clusterGm;
    GlobalTensor<float> sizeGmPtr;
    GlobalTensor<float> startGmPtr;

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