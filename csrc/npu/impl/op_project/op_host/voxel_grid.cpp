#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "voxel_grid_tiling.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    VoxelGridTilingData tiling;

    // 1. 获取输入形状
    // SoA布局: pos: [D, N]
    auto pos_shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t dim = pos_shape.GetDim(0);
    uint32_t numPoints = pos_shape.GetDim(1);

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    uint32_t coreNum = std::max(1U,ascendcPlatform.GetCoreNumAiv());
    uint32_t blockNum = std::min(coreNum,std::max(1U,(numPoints+4095)/4096));
    // Keep inter-core output boundaries at complete int64 cache lines.
    uint32_t pointsPerCore = blockNum == 1 ? numPoints : (numPoints/blockNum/8)*8;
    if(pointsPerCore==0) {blockNum=1;pointsPerCore=numPoints;}
    uint32_t tailPoints = numPoints-pointsPerCore*blockNum;
    context->SetBlockDim(blockNum);

    // 4. 设置 Tiling 数据
    tiling.set_numPoints(numPoints);
    tiling.set_dim(dim);
    tiling.set_blockPoints(pointsPerCore);
    tiling.set_tailPoints(tailPoints);
    tiling.set_reserved(0U);
    tiling.set_usePrecomputed(dim <= VOXEL_GRID_MAX_TILING_DIM ? 1U : 0U);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);  // [D, N]
    gert::Shape* y_shape = context->GetOutputShape(0);

    y_shape->SetDimNum(1);
    // output: [N]
    y_shape->SetDim(0, x1_shape->GetDim(1));

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    // 强制设置输出为 int64，不要跟随输入的 float
    context->SetOutputDataType(0, ge::DT_INT64);

    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class VoxelGrid : public OpDef {
public:
    explicit VoxelGrid(const char* name) : OpDef(name) {
        this->Input("pos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("size")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        // Host 已保证传入
        this->Input("start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        // Host 已保证传入
        this->Input("end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("cluster")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(VoxelGrid);

}  // namespace ops
