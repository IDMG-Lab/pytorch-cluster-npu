
#include "register/op_def_registry.h"
#include "voxel_grid_tiling.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    VoxelGridTilingData tiling;

    // 1. 获取输入形状 pos: [N, D]
    auto pos_shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t numPoints = pos_shape.GetDim(0);  // 点的数量 N
    uint32_t dim = pos_shape.GetDim(1);        // 坐标维度 D

    // 2. 确定核数 (BlockDim)
    uint32_t blockNum = 8;  // 或者通过 context->GetPlatformInfo() 动态获取
    context->SetBlockDim(blockNum);

    // 3. 计算切分 (按点切分，不打散单个点的 D 维坐标)
    uint32_t pointsPerCore = numPoints / blockNum;
    uint32_t tailPoints = numPoints % blockNum;

    // 4. 设置 Tiling 数据
    tiling.set_numPoints(numPoints);
    tiling.set_dim(dim);
    tiling.set_blockPoints(pointsPerCore);
    tiling.set_tailPoints(tailPoints);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);  // [N, D]
    gert::Shape* y_shape = context->GetOutputShape(0);

    // 输出应该是 [N]
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, x1_shape->GetDim(0));

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
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("size")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("start")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("end")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
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
