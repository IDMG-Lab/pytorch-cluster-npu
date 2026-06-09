#include "nearest_neighbor_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

// TilingFunc 函数
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    NearestNeighborTilingData tiling;

    // 1. 获取输入形状
    auto x_shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t numPoints = x_shape.GetDim(0);  // 点的数量
    uint32_t dim = x_shape.GetDim(1);        // 点的维度

     // 获取 ptr_x 的批次数量
    uint32_t batchSize = context->GetInputShape(2)->GetStorageShape().GetDim(0); 

    // 2. 设置核的维度
    uint32_t blockNum = 8; // 设置为8个block
    context->SetBlockDim(blockNum);

    // 3. 计算每个核处理的点数
    uint32_t pointsPerCore = numPoints / blockNum;
    uint32_t tailPoints = numPoints % blockNum;

    // 4. 保存Tiling数据
    tiling.set_numPoints(numPoints);
    tiling.set_dim(dim);
    tiling.set_blockPoints(pointsPerCore);
    tiling.set_tailPoints(tailPoints);
    tiling.set_batchSize(batchSize);  // 保存批次大小信息

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);  // 输入形状 [N, D]
    gert::Shape* y_shape = context->GetOutputShape(0);

    // 输出形状为 [N]
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, x_shape->GetDim(0));  // 输出维度为 N

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
            .DataType({ge::DT_INT64, ge::DT_INT64}) // 修改：指定两个类型
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})  // 修改：指定两个格式
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});  // 修改：确保有两个格式定义      
        this->Input("ptr_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})  // 修改：指定两个类型
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})  // 修改：指定两个格式
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});  // 修改：确保有两个格式定义       
        this->Output("cluster")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})  // 修改：指定两个类型
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})  // 修改：指定两个格式
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});  // 修改：确保有两个格式定义

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
    }
};

OP_ADD(NearestNeighbor);

} // namespace ops