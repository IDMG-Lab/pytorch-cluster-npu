#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // 每个队列中包含2个缓冲区（用于实现双缓冲）
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
        xGm.SetGlobalBuffer((__gm__ float*)x + startOffset * dim, this->workPoints * dim);
        yGm.SetGlobalBuffer((__gm__ float*)y, this->workPoints * dim);
        ptr_xGm.SetGlobalBuffer((__gm__ int64_t*)ptr_x, batchSize);
        ptr_yGm.SetGlobalBuffer((__gm__ int64_t*)ptr_y, batchSize);
        clusterGm.SetGlobalBuffer((__gm__ int64_t*)cluster + startOffset, this->workPoints);

        // 初始化队列和管道（两个缓冲区进行双缓冲处理）
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->workPoints * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->workPoints * sizeof(float));
        pipe.InitBuffer(outQueueCluster, BUFFER_NUM, this->workPoints * sizeof(int64_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t loopCount = workPoints / BUFFER_POINTS;
        uint32_t lastPoints = workPoints % BUFFER_POINTS;

        // 每次处理128个点
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t batch_offset = i * BUFFER_POINTS;
            CopyIn(batch_offset);
            Compute(batch_offset);
            CopyOut(batch_offset);
        }
        if (lastPoints > 0) {
            uint32_t batch_offset = loopCount * BUFFER_POINTS;
            CopyIn(batch_offset);
            Compute(batch_offset);
            CopyOut(batch_offset);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t batch_offset)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();

        // 使用 DataCopy 加载数据
        AscendC::DataCopy(xLocal, xGm[batch_offset * dim], this->workPoints);
        AscendC::DataCopy(yLocal, yGm[batch_offset * dim], this->workPoints);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t batch_offset)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        AscendC::LocalTensor<int64_t> clusterLocal = outQueueCluster.AllocTensor<int64_t>();

        // 遍历每个 x 点和每个 y 点计算距离
        for (uint32_t i = 0; i < this->workPoints; i++) {
            float dist = 0.f;
            int64_t best_idx = -1;  // 初始化最小距离的索引
            float best_dist = 1e38f;  // 设置一个较大的初始距离

            for (uint32_t n_y = 0; n_y < this->workPoints; n_y++) {
                // 计算每个 x 和 y 的距离
                for (uint32_t j = 0; j < this->dim; j++) {
                    float xi = xLocal.GetValue(i * dim + j);  // 获取 x 点的值
                    float yi = yLocal.GetValue(n_y * dim + j);  // 获取 y 点的值

                    // 计算差值并更新最小距离
                    float diff = xi - yi;
                    dist += diff * diff;
                }

                // 更新最近邻点
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = n_y;
                }
            }

            // 将最近邻的索引存入 cluster
            clusterLocal.SetValue(i, best_idx);  // 使用 SetValue 将最近邻的索引写回全局内存
        }

        outQueueCluster.EnQue(clusterLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t batch_offset)
    {
        AscendC::LocalTensor<int64_t> clusterLocal = outQueueCluster.DeQue<int64_t>();
        AscendC::DataCopy(clusterGm[batch_offset * this->workPoints], clusterLocal, this->workPoints);
        outQueueCluster.FreeTensor(clusterLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueCluster;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<int64_t> ptr_xGm;
    AscendC::GlobalTensor<int64_t> ptr_yGm;
    AscendC::GlobalTensor<int64_t> clusterGm;
    uint32_t dim, workPoints, batchSize;
};

extern "C" __global__ __aicore__ void nearest_neighbor(GM_ADDR x, GM_ADDR y, GM_ADDR ptr_x, GM_ADDR ptr_y, GM_ADDR cluster, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelNearestNeighbor op;
    op.Init(x, y, ptr_x, ptr_y, cluster, &tiling_data);
    op.Process();
}