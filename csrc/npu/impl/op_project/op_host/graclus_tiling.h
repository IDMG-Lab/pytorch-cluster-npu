#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(GraclusTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numNodes);
  TILING_DATA_FIELD_DEF(uint32_t, numEdges);
  TILING_DATA_FIELD_DEF(uint32_t, weighted);
  TILING_DATA_FIELD_DEF(uint32_t, phase);
  TILING_DATA_FIELD_DEF(uint32_t, weightKind);
  TILING_DATA_FIELD_DEF(uint32_t, cores);
  TILING_DATA_FIELD_DEF(uint32_t, tile);
  TILING_DATA_FIELD_DEF(uint32_t, resident);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Graclus, GraclusTilingData)
}
