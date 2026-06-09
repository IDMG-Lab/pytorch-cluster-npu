#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t BUFFER_POINTS = 128;  // 每次处理128个点

class KernelNearestNeighbor {
public:
    __aicore__ inline KernelNearestNeighbor() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR ptr_x, GM_ADDR ptr_y, GM_ADDR cluster, const NearestNeighborTilingData* tilingData)
    {
        uint32_t coreIdx = GetBlockIdx();
        this->dim = tilingData->dim;
        uint32_t startOffset = coreIdx * tilingData->blockPoints;
        this->workPoints = (coreIdx == 7) ? (tilingData->blockPoints + tilingData->tailPoints) : tilingData->blockPoints;
        this->batchSize = tilingData->batchSize; // 获取批次数量

        // 初始化 Global Buffer
        xGm.SetGlobalBuffer((__gm__ float*)x + startOffset * dim);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        ptr_xGm.SetGlobalBuffer((__gm__ int64_t*)ptr_x);
        ptr_yGm.SetGlobalBuffer((__gm__ int64_t*)ptr_y);
        clusterGm.SetGlobalBuffer((__gm__ int64_t*)cluster + startOffset);

        pipe.Init();
    }

    __aicore__ inline void Process()
    {
        uint32_t loopCount = workPoints / BUFFER_POINTS;
        uint32_t lastPoints = workPoints % BUFFER_POINTS;

        for (uint32_t i = 0; i < loopCount; i++) {
            Compute(BUFFER_POINTS, i * BUFFER_POINTS);
        }
        if (lastPoints > 0) {
            Compute(lastPoints, loopCount * BUFFER_POINTS);
        }
    }

private:
    __aicore__ inline void Compute(uint32_t count, uint32_t offset)
    {
        for (uint32_t i = 0; i < count; i++) {
            int64_t best_idx = -1;  // 初始化最小距离的索引
            float best_dist = 1e38f;  // 设置一个较大的初始距离

            // 获取当前点所在的批次
            uint32_t batch_idx = GetBatchIdx(offset + i);

            // 遍历每个 y 点，计算距离并找到最近的邻居
            int64_t y_start = ptr_yGm.GetValue(batch_idx);  // 获取批次对应的 y 起始位置
            int64_t y_end = ptr_yGm.GetValue(batch_idx + 1);  // 获取批次对应的 y 结束位置

            for (int64_t n_y = y_start; n_y < y_end; ++n_y) {
                float dist = 0.f;

                // 计算每个 x 和 y 的距离
                for (uint32_t j = 0; j < dim; j++) {
                    float xi = xGm.GetValue(offset * dim + i * dim + j);  // 使用 GetValue 获取 x 点的值
                    float yi = yGm.GetValue(n_y * dim + j);  // 使用 GetValue 获取 y 点的值

                    // 计算差值并更新最小距离
                    float diff = xi - yi;
                    dist += diff * diff;
                }

                // 更新最近邻点
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = n_y;  // 更新最佳索引
                }
            }

            // 将最近邻的索引存入 cluster
            clusterGm.SetValue(offset + i, best_idx);  // 使用 SetValue 将最近邻的索引写回全局内存
        }
    }

    __aicore__ inline uint32_t GetBatchIdx(uint32_t i)
    {
        int64_t batch_idx = 0;

        // 遍历 ptr_x 来确定每个 x 所在的批次
        for (uint32_t b = 0; b < batchSize; ++b) {
            int64_t x_start = ptr_xGm.GetValue(b);
            int64_t x_end = ptr_xGm.GetValue(b + 1);
            if (i >= x_start && i < x_end) {
                batch_idx = b;
                break;
            }
        }

        return batch_idx;
    }


private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    GlobalTensor<int64_t> ptr_xGm;
    GlobalTensor<int64_t> ptr_yGm;
    GlobalTensor<int64_t> clusterGm;
    TPipe pipe;
    uint32_t dim, workPoints, batchSize;
};

extern "C" __global__ __aicore__ void nearest_neighbor(GM_ADDR x, GM_ADDR y, GM_ADDR ptr_x, GM_ADDR ptr_y, GM_ADDR cluster, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelNearestNeighbor op;
    op.Init(x, y, ptr_x, ptr_y, cluster, &tiling_data);
    op.Process();
}