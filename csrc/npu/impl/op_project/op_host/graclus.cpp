#include "graclus_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  GraclusTilingData data;
  auto rowptr = context->GetInputShape(0)->GetStorageShape();
  auto col = context->GetInputShape(1)->GetStorageShape();
  data.set_numNodes(static_cast<uint32_t>(rowptr.GetDim(0) - 1));
  data.set_numEdges(static_cast<uint32_t>(col.GetDim(0)));
  data.set_weighted(*context->GetAttrs()->GetBool(0) ? 1U : 0U);
  context->SetBlockDim(1);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  auto rowptr = context->GetInputShape(0);
  auto out = context->GetOutputShape(0);
  out->SetDimNum(1);
  out->SetDim(0, rowptr->GetDim(0) - 1);
  return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Graclus : public OpDef {
public:
  explicit Graclus(const char* name) : OpDef(name) {
    for (const char* input : {"rowptr", "col"})
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT64})
          .Format({ge::FORMAT_ND});
    this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    this->Input("permutation").ParamType(REQUIRED).DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND});
    this->Output("cluster").ParamType(REQUIRED).DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND});
    this->Attr("weighted").Bool();
    this->SetInferShape(ge::InferShape);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(Graclus);
}
