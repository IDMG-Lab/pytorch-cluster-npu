#include "graclus_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  GraclusTilingData data;
  const auto rp = context->GetInputShape(0)->GetStorageShape();
  const auto ci = context->GetInputShape(1)->GetStorageShape();
  const uint32_t n = static_cast<uint32_t>(rp.GetDim(0) - 1);
  const uint32_t e = static_cast<uint32_t>(ci.GetDim(0));
  const uint32_t phase = *context->GetAttrs()->GetInt(0);
  const uint32_t weighted = *context->GetAttrs()->GetBool(1);
  const uint32_t kind = *context->GetAttrs()->GetInt(2);
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  uint64_t ub = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
  const uint32_t available = std::min(64U, std::max(1U, platform.GetCoreNumAiv()));
  // One output owner per 64-byte unit; min work is an engineering constant.
  const uint64_t work = phase == 0 ? uint64_t(n) + e : uint64_t(n) + (phase == 1 ? e : 0);
  uint32_t cores = std::min(available, std::max(1U, uint32_t((work + 2047) / 2048)));
  cores = std::min(cores, std::max(1U, (n + 15) / 16));
  if (n*uint64_t(4)>96*1024 && ((uint64_t(n)+255)/256)*32<=128*1024)
    cores=std::min(cores,std::max(1U,(n+511)/512));
  if (phase == 4) cores = 1;
  const uint64_t bitmapBytes=((uint64_t(n)+255)/256)*32;
  const uint64_t stateBytes = ((uint64_t(n) * 4 + 31) / 32) * 32;
  const bool resident = stateBytes <= 96 * 1024 && stateBytes + 64 * 1024 <= ub;
  const bool bitmap=!resident && bitmapBytes<=128*1024 && bitmapBytes+48*1024<=ub;
  data.set_numNodes(n); data.set_numEdges(e); data.set_phase(phase);
  data.set_weighted(weighted); data.set_weightKind(kind);
  const auto align32=[](uint64_t bytes){return (bytes+31)/32*32;};
  const uint64_t fusedBytes=align32((uint64_t(n)+1)*8)+align32(uint64_t(e)*8)+
    align32(uint64_t(e)*4)+2*align32(uint64_t(n)*4);
  const bool fused=phase==0 && kind==0 && e>0 && fusedBytes<=96*1024 && fusedBytes+68*1024<=ub;
  const uint32_t mode=fused?3:resident?1:bitmap?2:0;
  data.set_cores(fused?1:cores); data.set_tile(bitmap && !fused?512:1024); data.set_resident(mode);
  const uint64_t key = phase == 0 ? (fused?7:0) : phase == 1 ? (!weighted ? 2 : kind == 0 ? 1 : 3)+mode*10
                                  : phase == 2 ? 4 : phase == 3 ? 5 : 6;
  context->SetTilingKey(key); context->SetBlockDim(fused?1:cores);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const auto n = context->GetInputShape(0)->GetDim(0) - 1;
  for (int i : {0, 1, 3}) {
    auto shape = context->GetOutputShape(i);
    shape->SetDimNum(1); shape->SetDim(0, n);
  }
  auto counts = context->GetOutputShape(2);
  counts->SetDimNum(2); counts->SetDim(0, 64); counts->SetDim(1, 16);
  return ge::GRAPH_SUCCESS;
}
}
namespace ops {
class Graclus : public OpDef {
public:
  explicit Graclus(const char* name) : OpDef(name) {
    for (const char* input : {"rowptr", "col"})
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
    for (const char* input : {"weight", "state", "candidate"})
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
    for (const char* output : {"next_state", "next_candidate", "counts"})
      this->Output(output).ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
    this->Output("cluster").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
    this->Attr("phase").Int(); this->Attr("weighted").Bool(); this->Attr("weight_kind").Int();
    this->SetInferShape(ge::InferShape);
    this->AICore().SetTiling(optiling::TilingFunc); this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(Graclus);
}
