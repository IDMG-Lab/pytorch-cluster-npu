#include "knn_search_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  KNNSearchTilingData data;
  auto x = context->GetInputShape(0)->GetStorageShape();
  auto y = context->GetInputShape(1)->GetStorageShape();
  auto ptr = context->GetInputShape(2)->GetStorageShape();
  uint32_t n = static_cast<uint32_t>(x.GetDim(0));
  uint32_t m = static_cast<uint32_t>(y.GetDim(0));
  uint32_t d = static_cast<uint32_t>(x.GetDim(1));
  uint32_t groups = (m + 7) / 8;
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t cores = platform.GetCoreNumAiv();
  if (cores == 0) cores = 1;
  uint32_t blocks = std::min(cores, std::max(1U, groups));
  data.set_numReference(n);
  data.set_numQuery(m);
  data.set_dimension(d);
  data.set_numBatches(static_cast<uint32_t>(ptr.GetDim(0) - 1));
  data.set_k(static_cast<uint32_t>(*context->GetAttrs()->GetInt(0)));
  data.set_cosine(*context->GetAttrs()->GetBool(1) ? 1U : 0U);
  data.set_queriesPerCore(((groups + blocks - 1) / blocks) * 8);
  context->SetBlockDim(blocks);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const auto m = context->GetInputShape(1)->GetDim(0);
  const auto k = *context->GetAttrs()->GetInt(0);
  auto out = context->GetOutputShape(0);
  out->SetDimNum(2);
  out->SetDim(0, 2);
  out->SetDim(1, m * k);
  return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class KNNSearch : public OpDef {
public:
  explicit KNNSearch(const char* name) : OpDef(name) {
    for (const char* input : {"x", "y"})
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_FLOAT})
          .Format({ge::FORMAT_ND});
    for (const char* input : {"ptr_x", "ptr_y"})
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT64})
          .Format({ge::FORMAT_ND});
    this->Output("edge_index").ParamType(REQUIRED).DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND});
    this->Attr("k").Int();
    this->Attr("cosine").Bool();
    this->SetInferShape(ge::InferShape);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(KNNSearch);
}
