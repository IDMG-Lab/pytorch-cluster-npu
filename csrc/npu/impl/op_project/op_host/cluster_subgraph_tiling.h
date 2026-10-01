#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ClusterSubgraphTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numNodes);
  TILING_DATA_FIELD_DEF(uint32_t, numEdges);
  TILING_DATA_FIELD_DEF(uint32_t, subsetSize);
  TILING_DATA_FIELD_DEF(uint32_t, phase);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ClusterSubgraph, ClusterSubgraphTilingData)
}
