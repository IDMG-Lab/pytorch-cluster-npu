#ifndef FARTHEST_POINT_SAMPLING_TILING_H
#define FARTHEST_POINT_SAMPLING_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FarthestPointSamplingTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, dim);          // src shape: [dim, padded_N]
    TILING_DATA_FIELD_DEF(uint32_t, totalPoints);  // padded_N
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);    // batch 数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FarthestPointSampling, FarthestPointSamplingTilingData)

}

#endif