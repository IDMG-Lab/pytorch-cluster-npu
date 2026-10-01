#include "random_walk_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  RandomWalkTilingData data;
  auto rp = context->GetInputShape(0)->GetStorageShape();
  auto col = context->GetInputShape(1)->GetStorageShape();
  auto start = context->GetInputShape(2)->GetStorageShape();
  uint32_t n = static_cast<uint32_t>(rp.GetDim(0) - 1);
  uint32_t e = static_cast<uint32_t>(col.GetDim(0));
  uint32_t s = static_cast<uint32_t>(start.GetDim(0));
  uint32_t len = static_cast<uint32_t>(*context->GetAttrs()->GetInt(0));
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t cores = platform.GetCoreNumAiv();
  if (cores == 0) cores = 1;
  // Each physical output row is 64-byte aligned, so whole walks can be
  // distributed individually without sharing writable DCache lines.
  uint32_t blocks = std::min(cores, std::max(1U, s));
  uint32_t walksPerCore = (s + blocks - 1) / blocks;
  const uint32_t phase = static_cast<uint32_t>(*context->GetAttrs()->GetInt(3));
  const uint32_t checkBlocks = std::min(40U, cores);
  if (phase == 0) blocks = checkBlocks;
  data.set_numNodes(n);
  data.set_numEdges(e);
  data.set_numWalks(s);
  data.set_walkLength(len);
  data.set_nodesPitch((len + 1 + 7) / 8 * 8);
  data.set_edgesPitch((len + 7) / 8 * 8);
  data.set_walksPerCore(walksPerCore);
  data.set_phase(phase);
  data.set_checkBlocks(checkBlocks);
  data.set_p(*context->GetAttrs()->GetFloat(1));
  data.set_q(*context->GetAttrs()->GetFloat(2));
  context->SetBlockDim(blocks);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  auto s = context->GetInputShape(2)->GetDim(0);
  auto len = *context->GetAttrs()->GetInt(0);
  auto nodes = context->GetOutputShape(0);
  auto edges = context->GetOutputShape(1);
  nodes->SetDimNum(2);
  nodes->SetDim(0, s);
  nodes->SetDim(1, (len + 1 + 7) / 8 * 8);
  edges->SetDimNum(2);
  edges->SetDim(0, s);
  edges->SetDim(1, (len + 7) / 8 * 8);
  auto flags = context->GetOutputShape(2);
  flags->SetDimNum(1);
  flags->SetDim(0, 640);
  return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class RandomWalk : public OpDef {
public:
  explicit RandomWalk(const char* name) : OpDef(name) {
    for (const char* input : {"rowptr", "col", "start"})
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT64})
          .Format({ge::FORMAT_ND});
    this->Input("random").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    for (const char* output : {"walks", "e_ids"})
      this->Output(output).ParamType(REQUIRED).DataType({ge::DT_INT64})
          .Format({ge::FORMAT_ND});
    this->Output("graph_flags").ParamType(REQUIRED).DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND});
    this->Attr("walk_length").Int();
    this->Attr("p").Float();
    this->Attr("q").Float();
    this->Attr("phase").Int(1);
    this->SetInferShape(ge::InferShape);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(RandomWalk);
}
