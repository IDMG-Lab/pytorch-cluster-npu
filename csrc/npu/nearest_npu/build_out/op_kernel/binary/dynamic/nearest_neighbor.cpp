#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t BUFFER_POINTS = 128;

// ==========================================================
// dim=2 特化参数
// ==========================================================
constexpr uint32_t X_TILE_POINTS = 16;
constexpr uint32_t Y_TILE_POINTS = 256;
constexpr uint32_t Y_TILE_FLOATS = Y_TILE_POINTS * 2;

// float 单次 repeat 处理 256B = 64 个 float
constexpr uint32_t FLOAT_ELEMS_PER_REPEAT = 64;

// dim=2 GatherMask Normal 模式按 repeat 处理
constexpr uint32_t GATHER_REPEAT_TIMES =
    (Y_TILE_FLOATS + FLOAT_ELEMS_PER_REPEAT - 1) / FLOAT_ELEMS_PER_REPEAT;

constexpr uint32_t GATHER_SRC_FLOATS =
    GATHER_REPEAT_TIMES * FLOAT_ELEMS_PER_REPEAT;

constexpr uint32_t GATHER_DST_FLOATS =
    GATHER_SRC_FLOATS / 2;

// float 单 repeat = 64 float = 256B = 8 个 datablock
constexpr uint32_t GATHER_SRC0_REPEAT_STRIDE =
    (GATHER_REPEAT_TIMES > 1) ? 8 : 0;

// ReduceMin 临时空间
constexpr uint32_t REDUCE_WORK_FLOATS = 4096;

// ReduceMin 输出：
// reduceOutLocal[0] = min value
// reduceOutLocal[1] = min index
constexpr uint32_t REDUCE_OUT_FLOATS = 8;

// dim=2 y_tile 队列双缓冲
constexpr int32_t NN_Y_BUFFER_NUM = 2;

// ==========================================================
// 任意维通用路径：yBlock 复用 + dim_block 队列双缓冲参数
//
// 说明：
// 1. dim 仍然是运行时变量，支持 dim=32/128/300/...
// 2. dim 方向按 GENERAL_PIPE_DIM_CHUNK 分块
// 3. y 方向向量化
// 4. 一个 yBlock 搬入后，在 x_tile 内复用给多个 x
// ==========================================================
constexpr uint32_t GENERAL_PIPE_X_TILE_POINTS = 4;
constexpr uint32_t GENERAL_PIPE_Y_TILE_POINTS = 128;
constexpr uint32_t GENERAL_PIPE_DIM_CHUNK = 128;

constexpr uint32_t GENERAL_PIPE_YDIM_BLOCK_FLOATS =
    GENERAL_PIPE_Y_TILE_POINTS * GENERAL_PIPE_DIM_CHUNK;

constexpr int32_t NN_GENERAL_Y_BUFFER_NUM = 2;

// 共享向量 buffer 大小：既要满足 dim=2 的 256，也要满足通用路径 Y_TILE
constexpr uint32_t GENERAL_VEC_FLOATS =
    (GENERAL_PIPE_Y_TILE_POINTS > GATHER_DST_FLOATS)
        ? GENERAL_PIPE_Y_TILE_POINTS
        : GATHER_DST_FLOATS;

class KernelNearestNeighbor {
public:
    __aicore__ inline KernelNearestNeighbor() {}

    __aicore__ inline void Init(GM_ADDR x,
                                GM_ADDR y,
                                GM_ADDR ptr_x,
                                GM_ADDR ptr_y,
                                GM_ADDR cluster,
                                const NearestNeighborTilingData* tilingData)
    {
        uint32_t coreIdx = GetBlockIdx();

        this->dim = tilingData->dim;
        this->numPoints = tilingData->numPoints;
        this->blockPoints = tilingData->blockPoints;
        this->tailPoints = tilingData->tailPoints;
        this->batchSize = tilingData->batchSize;

        this->startOffset = coreIdx * this->blockPoints;

        uint32_t endOffset = this->startOffset + this->blockPoints;
        if (endOffset > this->numPoints) {
            endOffset = this->numPoints;
        }

        if (this->startOffset >= this->numPoints) {
            this->workPoints = 0;
        } else {
            this->workPoints = endOffset - this->startOffset;
        }

        xGm.SetGlobalBuffer((__gm__ float*)x);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        ptr_xGm.SetGlobalBuffer((__gm__ int64_t*)ptr_x);
        ptr_yGm.SetGlobalBuffer((__gm__ int64_t*)ptr_y);
        clusterGm.SetGlobalBuffer((__gm__ int64_t*)cluster);

        this->isDim2 = (this->dim == 2);

        pipe.Init();

        if (this->isDim2) {
            pipe.InitBuffer(
                yTileQueue,
                NN_Y_BUFFER_NUM,
                Y_TILE_FLOATS * sizeof(float)
            );
        } else {
            pipe.InitBuffer(
                generalYBlockQueue,
                NN_GENERAL_Y_BUFFER_NUM,
                GENERAL_PIPE_YDIM_BLOCK_FLOATS * sizeof(float)
            );

            // yBlock 复用版本中，需要为 x_tile 内多个 x 分别保存 distLocal
            pipe.InitBuffer(generalDist0Buf, GENERAL_VEC_FLOATS * sizeof(float));
            pipe.InitBuffer(generalDist1Buf, GENERAL_VEC_FLOATS * sizeof(float));
            pipe.InitBuffer(generalDist2Buf, GENERAL_VEC_FLOATS * sizeof(float));
            pipe.InitBuffer(generalDist3Buf, GENERAL_VEC_FLOATS * sizeof(float));
        }

        // ======================================================
        // 共享向量计算 buffer
        // ======================================================
        pipe.InitBuffer(yxBuf, GENERAL_VEC_FLOATS * sizeof(float));
        pipe.InitBuffer(yyBuf, GENERAL_VEC_FLOATS * sizeof(float));

        pipe.InitBuffer(x0VecBuf, GENERAL_VEC_FLOATS * sizeof(float));
        pipe.InitBuffer(x1VecBuf, GENERAL_VEC_FLOATS * sizeof(float));

        pipe.InitBuffer(dxBuf, GENERAL_VEC_FLOATS * sizeof(float));
        pipe.InitBuffer(dyBuf, GENERAL_VEC_FLOATS * sizeof(float));
        pipe.InitBuffer(dx2Buf, GENERAL_VEC_FLOATS * sizeof(float));
        pipe.InitBuffer(dy2Buf, GENERAL_VEC_FLOATS * sizeof(float));
        pipe.InitBuffer(distBuf, GENERAL_VEC_FLOATS * sizeof(float));

        pipe.InitBuffer(reduceOutBuf, REDUCE_OUT_FLOATS * sizeof(float));
        pipe.InitBuffer(reduceWorkBuf, REDUCE_WORK_FLOATS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->workPoints == 0) {
            return;
        }

        uint32_t loopCount = this->workPoints / BUFFER_POINTS;
        uint32_t lastPoints = this->workPoints % BUFFER_POINTS;

        for (uint32_t t = 0; t < loopCount; ++t) {
            uint32_t globalXStart = this->startOffset + t * BUFFER_POINTS;
            Compute(BUFFER_POINTS, globalXStart);
        }

        if (lastPoints > 0) {
            uint32_t globalXStart = this->startOffset + loopCount * BUFFER_POINTS;
            Compute(lastPoints, globalXStart);
        }
    }

private:
    __aicore__ inline void Compute(uint32_t count, uint32_t globalXStart)
    {
        uint32_t blockEnd = globalXStart + count;

        uint32_t batch_idx = FindBatchIdx(globalXStart);
        uint32_t curX = globalXStart;

        while (curX < blockEnd && batch_idx < this->batchSize) {
            int64_t x_batch_start_i64 = ptr_xGm.GetValue(batch_idx);
            int64_t x_batch_end_i64   = ptr_xGm.GetValue(batch_idx + 1);

            uint32_t x_batch_start = static_cast<uint32_t>(x_batch_start_i64);
            uint32_t x_batch_end   = static_cast<uint32_t>(x_batch_end_i64);

            uint32_t segStart = curX;
            if (segStart < x_batch_start) {
                segStart = x_batch_start;
            }

            uint32_t segEnd = blockEnd;
            if (segEnd > x_batch_end) {
                segEnd = x_batch_end;
            }

            int64_t y_start = ptr_yGm.GetValue(batch_idx);
            int64_t y_end   = ptr_yGm.GetValue(batch_idx + 1);

            if (this->dim == 2) {
                ComputeDim2StageReduceMin(segStart, segEnd, y_start, y_end);
            } else {
                ComputeGeneralAnyDimVectorizedReusePipeline(
                    segStart,
                    segEnd,
                    y_start,
                    y_end
                );
            }

            curX = segEnd;
            ++batch_idx;
        }
    }

    // =========================================================
    // dim=2: CopyIn y_tile
    // =========================================================
    __aicore__ inline void CopyInDim2YTile(int64_t y_tile_start)
    {
        LocalTensor<float> yTileLocal = yTileQueue.AllocTensor<float>();

        uint32_t gmOffset =
            static_cast<uint32_t>(y_tile_start) * 2;

        DataCopy(yTileLocal, yGm[gmOffset], Y_TILE_FLOATS);

        yTileQueue.EnQue(yTileLocal);
    }

    // =========================================================
    // dim=2: Compute one queued y_tile
    // =========================================================
    __aicore__ inline void ComputeDim2OneQueuedYTile(
        int64_t y_tile_start,
        uint32_t x_count,
        float x0Reg[X_TILE_POINTS],
        float x1Reg[X_TILE_POINTS],
        float bestDist[X_TILE_POINTS],
        int64_t bestIdx[X_TILE_POINTS])
    {
        LocalTensor<float> yTileLocal = yTileQueue.DeQue<float>();

        LocalTensor<float> yxLocal = yxBuf.Get<float>();
        LocalTensor<float> yyLocal = yyBuf.Get<float>();

        LocalTensor<float> x0VecLocal = x0VecBuf.Get<float>();
        LocalTensor<float> x1VecLocal = x1VecBuf.Get<float>();

        LocalTensor<float> dxLocal = dxBuf.Get<float>();
        LocalTensor<float> dyLocal = dyBuf.Get<float>();
        LocalTensor<float> dx2Local = dx2Buf.Get<float>();
        LocalTensor<float> dy2Local = dy2Buf.Get<float>();
        LocalTensor<float> distLocal = distBuf.Get<float>();

        LocalTensor<float> reduceOutLocal = reduceOutBuf.Get<float>();
        LocalTensor<float> reduceWorkLocal = reduceWorkBuf.Get<float>();

        uint64_t rsvdCntX = 0;
        uint64_t rsvdCntY = 0;

        uint8_t xPattern = 1;
        uint8_t yPattern = 2;

        GatherMask(
            yxLocal,
            yTileLocal,
            xPattern,
            false,
            0,
            {
                1,
                static_cast<uint16_t>(GATHER_REPEAT_TIMES),
                static_cast<uint16_t>(GATHER_SRC0_REPEAT_STRIDE),
                0
            },
            rsvdCntX
        );

        GatherMask(
            yyLocal,
            yTileLocal,
            yPattern,
            false,
            0,
            {
                1,
                static_cast<uint16_t>(GATHER_REPEAT_TIMES),
                static_cast<uint16_t>(GATHER_SRC0_REPEAT_STRIDE),
                0
            },
            rsvdCntY
        );

        PipeBarrier<PIPE_V>();

        yTileQueue.FreeTensor(yTileLocal);

        for (uint32_t xi = 0; xi < x_count; ++xi) {
            Duplicate<float>(x0VecLocal, x0Reg[xi], Y_TILE_POINTS);
            Duplicate<float>(x1VecLocal, x1Reg[xi], Y_TILE_POINTS);

            PipeBarrier<PIPE_V>();

            Sub<float>(dxLocal, x0VecLocal, yxLocal, Y_TILE_POINTS);
            Sub<float>(dyLocal, x1VecLocal, yyLocal, Y_TILE_POINTS);

            PipeBarrier<PIPE_V>();

            Mul<float>(dx2Local, dxLocal, dxLocal, Y_TILE_POINTS);
            Mul<float>(dy2Local, dyLocal, dyLocal, Y_TILE_POINTS);

            PipeBarrier<PIPE_V>();

            Add<float>(distLocal, dx2Local, dy2Local, Y_TILE_POINTS);

            PipeBarrier<PIPE_V>();

            ReduceMin<float>(
                reduceOutLocal,
                distLocal,
                reduceWorkLocal,
                static_cast<int32_t>(Y_TILE_POINTS),
                true
            );

            PipeBarrier<PIPE_V>();

            float tileMinDist = reduceOutLocal.GetValue(0);
            float tileMinIdxAsFloat = reduceOutLocal.GetValue(1);
            uint32_t tileMinIdx =
                *reinterpret_cast<uint32_t*>(&tileMinIdxAsFloat);

            int64_t cur_y_idx =
                y_tile_start + static_cast<int64_t>(tileMinIdx);

            if (tileMinDist < bestDist[xi]) {
                bestDist[xi] = tileMinDist;
                bestIdx[xi] = cur_y_idx;
            }
        }
    }

    // =========================================================
    // dim=2 特化向量化 + y_tile 队列双缓冲路径
    // =========================================================
    __aicore__ inline void ComputeDim2StageReduceMin(uint32_t segStart,
                                                     uint32_t segEnd,
                                                     int64_t y_start,
                                                     int64_t y_end)
    {
        int64_t y_count = y_end - y_start;

        int64_t y_tile_count =
            y_count / static_cast<int64_t>(Y_TILE_POINTS);

        int64_t y_main_end =
            y_start + y_tile_count * static_cast<int64_t>(Y_TILE_POINTS);

        for (uint32_t x_tile_start = segStart;
             x_tile_start < segEnd;
             x_tile_start += X_TILE_POINTS) {

            uint32_t x_count = X_TILE_POINTS;
            if (x_tile_start + x_count > segEnd) {
                x_count = segEnd - x_tile_start;
            }

            float x0Reg[X_TILE_POINTS];
            float x1Reg[X_TILE_POINTS];
            float bestDist[X_TILE_POINTS];
            int64_t bestIdx[X_TILE_POINTS];

            for (uint32_t xi = 0; xi < x_count; ++xi) {
                uint32_t global_x_idx = x_tile_start + xi;
                uint32_t x_base = global_x_idx * 2;

                x0Reg[xi] = xGm.GetValue(x_base);
                x1Reg[xi] = xGm.GetValue(x_base + 1);

                bestDist[xi] = 1e38f;
                bestIdx[xi] = -1;
            }

            if (y_tile_count > 0) {
                CopyInDim2YTile(y_start);

                for (int64_t y_tile_start = y_start;
                     y_tile_start < y_main_end;
                     y_tile_start += static_cast<int64_t>(Y_TILE_POINTS)) {

                    int64_t next_y_tile_start =
                        y_tile_start + static_cast<int64_t>(Y_TILE_POINTS);

                    if (next_y_tile_start < y_main_end) {
                        CopyInDim2YTile(next_y_tile_start);
                    }

                    ComputeDim2OneQueuedYTile(
                        y_tile_start,
                        x_count,
                        x0Reg,
                        x1Reg,
                        bestDist,
                        bestIdx
                    );
                }
            }

            for (uint32_t xi = 0; xi < x_count; ++xi) {
                for (int64_t n_y = y_main_end; n_y < y_end; ++n_y) {
                    uint32_t y_base = static_cast<uint32_t>(n_y) * 2;

                    float y0 = yGm.GetValue(y_base);
                    float y1 = yGm.GetValue(y_base + 1);

                    float diff0 = x0Reg[xi] - y0;
                    float diff1 = x1Reg[xi] - y1;
                    float dist = diff0 * diff0 + diff1 * diff1;

                    if (dist < bestDist[xi]) {
                        bestDist[xi] = dist;
                        bestIdx[xi] = n_y;
                    }
                }
            }

            for (uint32_t xi = 0; xi < x_count; ++xi) {
                clusterGm.SetValue(x_tile_start + xi, bestIdx[xi]);
            }
        }
    }

    // =========================================================
    // 任意维路径：CopyIn 一个 y_tile 的 dim_block
    //
    // 搬运逻辑：
    // y[(y_tile_start + k), dim_start : dim_start + curDimCount]
    //
    // local 布局：
    // yBlockLocal[k * GENERAL_PIPE_DIM_CHUNK + dc]
    //
    // 混合策略：
    // - 满足 32B 对齐时用 DataCopy
    // - 否则用 DataCopyPad
    // =========================================================
    __aicore__ inline void CopyInGeneralAnyDimBlock(int64_t y_tile_start,
                                                    uint32_t curYCount,
                                                    uint32_t dim_start,
                                                    uint32_t curDimCount)
    {
        LocalTensor<float> yBlockLocal =
            generalYBlockQueue.AllocTensor<float>();

        uint32_t copyBytes =
            static_cast<uint32_t>(curDimCount * sizeof(float));

        DataCopyExtParams copyParams{
            1,
            copyBytes,
            0,
            0,
            0
        };

        DataCopyPadExtParams<float> padParams{
            true,
            0,
            0,
            0
        };

        for (uint32_t k = 0; k < curYCount; ++k) {
            uint32_t y_idx =
                static_cast<uint32_t>(y_tile_start) + k;

            uint32_t gmOffset =
                y_idx * this->dim + dim_start;

            uint32_t localOffset =
                k * GENERAL_PIPE_DIM_CHUNK;

            bool gmAligned32 =
                ((gmOffset % 8) == 0);

            bool localAligned32 =
                ((localOffset % 8) == 0);

            bool lenAligned32 =
                ((curDimCount % 8) == 0);

            if (gmAligned32 && localAligned32 && lenAligned32) {
                DataCopy(
                    yBlockLocal[localOffset],
                    yGm[gmOffset],
                    curDimCount
                );
            } else {
                DataCopyPad(
                    yBlockLocal[localOffset],
                    yGm[gmOffset],
                    copyParams,
                    padParams
                );
            }
        }

        generalYBlockQueue.EnQue(yBlockLocal);
    }

    // =========================================================
    // 对单个 x 累加当前维度 dc 的距离
    //
    // distLocal[k] += (x[globalDim] - yDimLocal[k])^2
    // =========================================================
    __aicore__ inline void AccumulateOneXOneDim(
        uint32_t x_base,
        uint32_t globalDim,
        uint32_t curYCount,
        LocalTensor<float>& yDimLocal,
        LocalTensor<float>& distLocal,
        LocalTensor<float>& xDimVecLocal,
        LocalTensor<float>& diffLocal,
        LocalTensor<float>& diff2Local)
    {
        float xVal = xGm.GetValue(x_base + globalDim);

        Duplicate<float>(
            xDimVecLocal,
            xVal,
            curYCount
        );

        PipeBarrier<PIPE_V>();

        Sub<float>(
            diffLocal,
            xDimVecLocal,
            yDimLocal,
            curYCount
        );

        PipeBarrier<PIPE_V>();

        Mul<float>(
            diff2Local,
            diffLocal,
            diffLocal,
            curYCount
        );

        PipeBarrier<PIPE_V>();

        Add<float>(
            distLocal,
            distLocal,
            diff2Local,
            curYCount
        );

        PipeBarrier<PIPE_V>();
    }

    // =========================================================
    // 任意维路径：计算一个 queued yBlock，并复用给 x_tile 内多个 x
    //
    // 该函数是 yBlock 复用的核心：
    // - yBlockLocal 只 DeQue 一次
    // - 每个维度 dc 下先构造 yDimLocal
    // - 然后依次复用 yDimLocal 给 x0/x1/x2/x3 累加距离
    // =========================================================
    __aicore__ inline void ComputeGeneralAnyDimOneQueuedBlockReuse(
        uint32_t x_base0,
        uint32_t x_base1,
        uint32_t x_base2,
        uint32_t x_base3,
        uint32_t x_count,
        uint32_t dim_start,
        uint32_t curDimCount,
        uint32_t curYCount,
        LocalTensor<float>& dist0Local,
        LocalTensor<float>& dist1Local,
        LocalTensor<float>& dist2Local,
        LocalTensor<float>& dist3Local)
    {
        LocalTensor<float> yBlockLocal =
            generalYBlockQueue.DeQue<float>();

        LocalTensor<float> yDimLocal = yxBuf.Get<float>();
        LocalTensor<float> xDimVecLocal = x0VecBuf.Get<float>();
        LocalTensor<float> diffLocal = dxBuf.Get<float>();
        LocalTensor<float> diff2Local = dx2Buf.Get<float>();

        for (uint32_t dc = 0; dc < curDimCount; ++dc) {
            uint32_t globalDim = dim_start + dc;

            // 从 [Y_TILE, DIM_CHUNK] 中提取第 dc 列，得到 y_tile 方向连续向量
            for (uint32_t k = 0; k < curYCount; ++k) {
                uint32_t localOffset =
                    k * GENERAL_PIPE_DIM_CHUNK + dc;

                float yVal = yBlockLocal.GetValue(localOffset);
                yDimLocal.SetValue(k, yVal);
            }

            PipeBarrier<PIPE_V>();

            if (x_count > 0) {
                AccumulateOneXOneDim(
                    x_base0,
                    globalDim,
                    curYCount,
                    yDimLocal,
                    dist0Local,
                    xDimVecLocal,
                    diffLocal,
                    diff2Local
                );
            }

            if (x_count > 1) {
                AccumulateOneXOneDim(
                    x_base1,
                    globalDim,
                    curYCount,
                    yDimLocal,
                    dist1Local,
                    xDimVecLocal,
                    diffLocal,
                    diff2Local
                );
            }

            if (x_count > 2) {
                AccumulateOneXOneDim(
                    x_base2,
                    globalDim,
                    curYCount,
                    yDimLocal,
                    dist2Local,
                    xDimVecLocal,
                    diffLocal,
                    diff2Local
                );
            }

            if (x_count > 3) {
                AccumulateOneXOneDim(
                    x_base3,
                    globalDim,
                    curYCount,
                    yDimLocal,
                    dist3Local,
                    xDimVecLocal,
                    diffLocal,
                    diff2Local
                );
            }
        }

        generalYBlockQueue.FreeTensor(yBlockLocal);
    }

    // =========================================================
    // 任意维通用向量化 + yBlock 复用 + dim_block 队列双缓冲
    //
    // 核心改动：
    // - 一个 y_tile + dim_block 只 CopyIn 一次
    // - 复用给 x_tile 内多个 x
    // - 每个 x 维护独立 distLocal
    // =========================================================
    __aicore__ inline void ComputeGeneralAnyDimVectorizedReusePipeline(
        uint32_t segStart,
        uint32_t segEnd,
        int64_t y_start,
        int64_t y_end)
    {
        LocalTensor<float> dist0Local = generalDist0Buf.Get<float>();
        LocalTensor<float> dist1Local = generalDist1Buf.Get<float>();
        LocalTensor<float> dist2Local = generalDist2Buf.Get<float>();
        LocalTensor<float> dist3Local = generalDist3Buf.Get<float>();

        LocalTensor<float> reduceOutLocal = reduceOutBuf.Get<float>();
        LocalTensor<float> reduceWorkLocal = reduceWorkBuf.Get<float>();

        for (uint32_t x_tile_start = segStart;
             x_tile_start < segEnd;
             x_tile_start += GENERAL_PIPE_X_TILE_POINTS) {

            uint32_t x_count = GENERAL_PIPE_X_TILE_POINTS;
            if (x_tile_start + x_count > segEnd) {
                x_count = segEnd - x_tile_start;
            }

            float bestDist[GENERAL_PIPE_X_TILE_POINTS];
            int64_t bestIdx[GENERAL_PIPE_X_TILE_POINTS];

            for (uint32_t xi = 0; xi < x_count; ++xi) {
                bestDist[xi] = 1e38f;
                bestIdx[xi] = -1;
            }

            uint32_t x_base0 = (x_tile_start + 0) * this->dim;
            uint32_t x_base1 = (x_tile_start + 1) * this->dim;
            uint32_t x_base2 = (x_tile_start + 2) * this->dim;
            uint32_t x_base3 = (x_tile_start + 3) * this->dim;

            for (int64_t y_tile_start = y_start;
                 y_tile_start < y_end;
                 y_tile_start += static_cast<int64_t>(GENERAL_PIPE_Y_TILE_POINTS)) {

                uint32_t curYCount = GENERAL_PIPE_Y_TILE_POINTS;

                if (y_tile_start + static_cast<int64_t>(curYCount) > y_end) {
                    curYCount =
                        static_cast<uint32_t>(y_end - y_tile_start);
                }

                // 每个 y_tile 开始时，为 x_tile 内每个 x 初始化 distLocal
                if (x_count > 0) {
                    Duplicate<float>(dist0Local, 0.0f, curYCount);
                }
                if (x_count > 1) {
                    Duplicate<float>(dist1Local, 0.0f, curYCount);
                }
                if (x_count > 2) {
                    Duplicate<float>(dist2Local, 0.0f, curYCount);
                }
                if (x_count > 3) {
                    Duplicate<float>(dist3Local, 0.0f, curYCount);
                }

                PipeBarrier<PIPE_V>();

                if (this->dim > 0) {
                    uint32_t firstDimCount = GENERAL_PIPE_DIM_CHUNK;
                    if (firstDimCount > this->dim) {
                        firstDimCount = this->dim;
                    }

                    // 先预取第一个 dim_block
                    CopyInGeneralAnyDimBlock(
                        y_tile_start,
                        curYCount,
                        0,
                        firstDimCount
                    );

                    for (uint32_t dim_start = 0;
                         dim_start < this->dim;
                         dim_start += GENERAL_PIPE_DIM_CHUNK) {

                        uint32_t curDimCount = GENERAL_PIPE_DIM_CHUNK;

                        if (dim_start + curDimCount > this->dim) {
                            curDimCount = this->dim - dim_start;
                        }

                        uint32_t nextDimStart =
                            dim_start + GENERAL_PIPE_DIM_CHUNK;

                        if (nextDimStart < this->dim) {
                            uint32_t nextDimCount =
                                GENERAL_PIPE_DIM_CHUNK;

                            if (nextDimStart + nextDimCount > this->dim) {
                                nextDimCount =
                                    this->dim - nextDimStart;
                            }

                            // 预取下一个 dim_block
                            CopyInGeneralAnyDimBlock(
                                y_tile_start,
                                curYCount,
                                nextDimStart,
                                nextDimCount
                            );
                        }

                        // 计算当前 dim_block，并复用给 x_tile 内多个 x
                        ComputeGeneralAnyDimOneQueuedBlockReuse(
                            x_base0,
                            x_base1,
                            x_base2,
                            x_base3,
                            x_count,
                            dim_start,
                            curDimCount,
                            curYCount,
                            dist0Local,
                            dist1Local,
                            dist2Local,
                            dist3Local
                        );
                    }
                }

                // 当前 y_tile 内对每个 x 做 ReduceMin，并更新 best
                if (x_count > 0) {
                    ReduceMin<float>(
                        reduceOutLocal,
                        dist0Local,
                        reduceWorkLocal,
                        static_cast<int32_t>(curYCount),
                        true
                    );

                    PipeBarrier<PIPE_V>();

                    float tileMinDist = reduceOutLocal.GetValue(0);
                    float tileMinIdxAsFloat = reduceOutLocal.GetValue(1);
                    uint32_t tileMinIdx =
                        *reinterpret_cast<uint32_t*>(&tileMinIdxAsFloat);

                    int64_t cur_y_idx =
                        y_tile_start + static_cast<int64_t>(tileMinIdx);

                    if (tileMinDist < bestDist[0]) {
                        bestDist[0] = tileMinDist;
                        bestIdx[0] = cur_y_idx;
                    }
                }

                if (x_count > 1) {
                    ReduceMin<float>(
                        reduceOutLocal,
                        dist1Local,
                        reduceWorkLocal,
                        static_cast<int32_t>(curYCount),
                        true
                    );

                    PipeBarrier<PIPE_V>();

                    float tileMinDist = reduceOutLocal.GetValue(0);
                    float tileMinIdxAsFloat = reduceOutLocal.GetValue(1);
                    uint32_t tileMinIdx =
                        *reinterpret_cast<uint32_t*>(&tileMinIdxAsFloat);

                    int64_t cur_y_idx =
                        y_tile_start + static_cast<int64_t>(tileMinIdx);

                    if (tileMinDist < bestDist[1]) {
                        bestDist[1] = tileMinDist;
                        bestIdx[1] = cur_y_idx;
                    }
                }

                if (x_count > 2) {
                    ReduceMin<float>(
                        reduceOutLocal,
                        dist2Local,
                        reduceWorkLocal,
                        static_cast<int32_t>(curYCount),
                        true
                    );

                    PipeBarrier<PIPE_V>();

                    float tileMinDist = reduceOutLocal.GetValue(0);
                    float tileMinIdxAsFloat = reduceOutLocal.GetValue(1);
                    uint32_t tileMinIdx =
                        *reinterpret_cast<uint32_t*>(&tileMinIdxAsFloat);

                    int64_t cur_y_idx =
                        y_tile_start + static_cast<int64_t>(tileMinIdx);

                    if (tileMinDist < bestDist[2]) {
                        bestDist[2] = tileMinDist;
                        bestIdx[2] = cur_y_idx;
                    }
                }

                if (x_count > 3) {
                    ReduceMin<float>(
                        reduceOutLocal,
                        dist3Local,
                        reduceWorkLocal,
                        static_cast<int32_t>(curYCount),
                        true
                    );

                    PipeBarrier<PIPE_V>();

                    float tileMinDist = reduceOutLocal.GetValue(0);
                    float tileMinIdxAsFloat = reduceOutLocal.GetValue(1);
                    uint32_t tileMinIdx =
                        *reinterpret_cast<uint32_t*>(&tileMinIdxAsFloat);

                    int64_t cur_y_idx =
                        y_tile_start + static_cast<int64_t>(tileMinIdx);

                    if (tileMinDist < bestDist[3]) {
                        bestDist[3] = tileMinDist;
                        bestIdx[3] = cur_y_idx;
                    }
                }
            }

            for (uint32_t xi = 0; xi < x_count; ++xi) {
                clusterGm.SetValue(x_tile_start + xi, bestIdx[xi]);
            }
        }
    }

    __aicore__ inline uint32_t FindBatchIdx(uint32_t global_x_idx)
    {
        uint32_t batch_idx = 0;

        for (uint32_t b = 0; b < this->batchSize; ++b) {
            int64_t x_start = ptr_xGm.GetValue(b);
            int64_t x_end   = ptr_xGm.GetValue(b + 1);

            if (global_x_idx >= static_cast<uint32_t>(x_start) &&
                global_x_idx <  static_cast<uint32_t>(x_end)) {
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

    // dim=2 y_tile 队列双缓冲
    TQue<TPosition::VECIN, NN_Y_BUFFER_NUM> yTileQueue;

    // 任意维通用路径的 y-dim block 队列双缓冲
    TQue<TPosition::VECIN, NN_GENERAL_Y_BUFFER_NUM> generalYBlockQueue;

    // 共享向量 buffer
    TBuf<TPosition::VECCALC> yxBuf;
    TBuf<TPosition::VECCALC> yyBuf;

    TBuf<TPosition::VECCALC> x0VecBuf;
    TBuf<TPosition::VECCALC> x1VecBuf;

    TBuf<TPosition::VECCALC> dxBuf;
    TBuf<TPosition::VECCALC> dyBuf;
    TBuf<TPosition::VECCALC> dx2Buf;
    TBuf<TPosition::VECCALC> dy2Buf;
    TBuf<TPosition::VECCALC> distBuf;

    // yBlock 复用版本中，为 x_tile 内多个 x 分别保存累计距离
    TBuf<TPosition::VECCALC> generalDist0Buf;
    TBuf<TPosition::VECCALC> generalDist1Buf;
    TBuf<TPosition::VECCALC> generalDist2Buf;
    TBuf<TPosition::VECCALC> generalDist3Buf;

    TBuf<TPosition::VECCALC> reduceOutBuf;
    TBuf<TPosition::VECCALC> reduceWorkBuf;

    uint32_t dim = 0;
    uint32_t numPoints = 0;
    uint32_t blockPoints = 0;
    uint32_t tailPoints = 0;
    uint32_t batchSize = 0;
    uint32_t startOffset = 0;
    uint32_t workPoints = 0;
    bool isDim2 = false;
};

extern "C" __global__ __aicore__ void nearest_neighbor(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR ptr_x,
    GM_ADDR ptr_y,
    GM_ADDR cluster,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    KernelNearestNeighbor op;
    op.Init(x, y, ptr_x, ptr_y, cluster, &tiling_data);
    op.Process();
}