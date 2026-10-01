#include "cluster_subgraph_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  ClusterSubgraphTilingData data;
  const auto rowptr = context->GetInputShape(0)->GetStorageShape();
  const auto col = context->GetInputShape(1)->GetStorageShape();
  const auto nodes = context->GetInputShape(2)->GetStorageShape();
  data.set_numNodes(static_cast<uint32_t>(rowptr.GetDim(0) - 1));
  data.set_numEdges(static_cast<uint32_t>(col.GetDim(0)));
  data.set_subsetSize(static_cast<uint32_t>(nodes.GetDim(0)));
  const auto phase = static_cast<uint32_t>(*context->GetAttrs()->GetInt(0));
  data.set_phase(phase);
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const uint32_t cores = std::max(1U, platform.GetCoreNumAiv());
  uint32_t blocks = 1;
  if (phase == 0 && data.get_numNodes() >= 65536) blocks = cores;
  if (phase == 1) blocks = std::min(cores, std::max(1U, (data.get_subsetSize() + 7) / 8));
  if (phase == 3) blocks = cores;
  context->SetBlockDim(blocks);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const auto e = context->GetInputShape(1)->GetDim(0);
  const auto q = context->GetInputShape(2)->GetDim(0);
  auto rowptr = context->GetOutputShape(0);
  auto col = context->GetOutputShape(1);
  auto eid = context->GetOutputShape(2);
  auto count = context->GetOutputShape(3);
  rowptr->SetDimNum(1);
  rowptr->SetDim(0, q + 1);
  col->SetDimNum(1);
  col->SetDim(0, e);
  eid->SetDimNum(1);
  eid->SetDim(0, e);
  count->SetDimNum(1);
  count->SetDim(0, 1);
  return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class ClusterSubgraph : public OpDef {
public:
  explicit ClusterSubgraph(const char* name) : OpDef(name) {
    for (const char* input : {"rowptr", "col", "nodes"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT64})
          .Format({ge::FORMAT_ND});
    }
    this->Input("mapping").ParamType(REQUIRED).DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND});
    for (const char* output : {"out_rowptr", "out_col", "out_eid", "out_count"}) {
      this->Output(output).ParamType(REQUIRED).DataType({ge::DT_INT64})
          .Format({ge::FORMAT_ND});
    }
    this->Attr("phase").Int();
    this->SetInferShape(ge::InferShape);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(ClusterSubgraph);
}
