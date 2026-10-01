#pragma once
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RandomWalkTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numNodes);
  TILING_DATA_FIELD_DEF(uint32_t, numEdges);
  TILING_DATA_FIELD_DEF(uint32_t, numWalks);
  TILING_DATA_FIELD_DEF(uint32_t, walkLength);
  TILING_DATA_FIELD_DEF(uint32_t, nodesPitch);
  TILING_DATA_FIELD_DEF(uint32_t, edgesPitch);
  TILING_DATA_FIELD_DEF(uint32_t, walksPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, phase);
  TILING_DATA_FIELD_DEF(uint32_t, checkBlocks);
  TILING_DATA_FIELD_DEF(float, p);
  TILING_DATA_FIELD_DEF(float, q);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RandomWalk, RandomWalkTilingData)
}
