#include "nearest_neighbor_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

static inline uint32_t AlignUp(uint32_t a, uint32_t align)
{
    return CeilDiv(a, align) * align;
}

// TilingFunc 函数
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    NearestNeighborTilingData tiling;

    auto x_shape = context->GetInputShape(0)->GetStorageShape();

    uint32_t numPoints = static_cast<uint32_t>(x_shape.GetDim(0));
    uint32_t dim = static_cast<uint32_t>(x_shape.GetDim(1));
    uint32_t batchSize =
        static_cast<uint32_t>(context->GetInputShape(2)->GetStorageShape().GetDim(0) - 1);

    constexpr uint32_t OUT_ELEMS_PER_CACHELINE = 64 / sizeof(int64_t); // 8

    // ============================================================
    // 1. Host 侧获取平台核数
    //
    // 当前 nearest 算子主要使用 Vector 类 API：
    // DataCopy / GatherMask / Duplicate / Sub / Mul / Add / ReduceMin
    // 因此优先使用 AIV 核数。
    //
    // 如果 AIV 核数获取异常为 0，则退回 AIC 核数；
    // 如果仍为 0，则至少使用 1 个 block。
    // ============================================================
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();

    uint32_t maxCoreNum = aivNum;
    if (maxCoreNum == 0) {
        maxCoreNum = aicNum;
    }
    if (maxCoreNum == 0) {
        maxCoreNum = 1;
    }

    // ============================================================
    // 2. 根据 x 点数确定实际使用 block 数
    //
    // 原来这里固定 MAX_BLOCK_NUM = 8。
    // 现在改成：最多使用平台 Vector 核数，但不能超过有意义的输出切分数。
    // 每个 block 至少处理一个 int64 cache line，即 8 个输出。
    // ============================================================
    uint32_t maxUsefulBlocks =
        CeilDiv(numPoints, OUT_ELEMS_PER_CACHELINE);

    uint32_t blockNum = std::min(maxCoreNum, maxUsefulBlocks);

    if (blockNum == 0) {
        blockNum = 1;
    }

    // ============================================================
    // 3. 计算每个 block 处理的点数，并按输出 cache line 对齐
    // ============================================================
    uint32_t pointsPerCore = CeilDiv(numPoints, blockNum);

    pointsPerCore = AlignUp(pointsPerCore, OUT_ELEMS_PER_CACHELINE);

    // ============================================================
    // 4. 对齐后反推真实 blockNum
    //
    // 例如 numPoints 很小时，对齐后可能不需要那么多 block。
    // 所以这里反推一次，避免启动无意义 block。
    // ============================================================
    blockNum = CeilDiv(numPoints, pointsPerCore);

    if (blockNum == 0) {
        blockNum = 1;
    }

    context->SetBlockDim(blockNum);

    // ============================================================
    // 5. 写入 tilingData
    // ============================================================
    tiling.set_numPoints(numPoints);
    tiling.set_dim(dim);
    tiling.set_blockPoints(pointsPerCore);
    tiling.set_tailPoints(0);
    tiling.set_batchSize(batchSize);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );

    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling


namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);

    // 输出形状为 [N]
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, x_shape->GetDim(0));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    // 强制输出类型为 int64
    context->SetOutputDataType(0, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge


namespace ops {

class NearestNeighbor : public OpDef {
public:
    explicit NearestNeighbor(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("ptr_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("ptr_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("cluster")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(NearestNeighbor);

} // namespace ops