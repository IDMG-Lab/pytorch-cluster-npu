#pragma once
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(KNNSearchTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numReference);
  TILING_DATA_FIELD_DEF(uint32_t, numQuery);
  TILING_DATA_FIELD_DEF(uint32_t, dimension);
  TILING_DATA_FIELD_DEF(uint32_t, numBatches);
  TILING_DATA_FIELD_DEF(uint32_t, k);
  TILING_DATA_FIELD_DEF(uint32_t, cosine);
  TILING_DATA_FIELD_DEF(uint32_t, queriesPerCore);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(KNNSearch, KNNSearchTilingData)
}
