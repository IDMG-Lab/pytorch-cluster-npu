#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint32_t VOXEL_GRID_MAX_TILING_DIM = 16;

BEGIN_TILING_DATA_DEF(VoxelGridTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numPoints);
TILING_DATA_FIELD_DEF(uint32_t, dim);
TILING_DATA_FIELD_DEF(uint32_t, blockPoints);
TILING_DATA_FIELD_DEF(uint32_t, tailPoints);
TILING_DATA_FIELD_DEF(uint32_t, usePrecomputed);
TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(VoxelGrid, VoxelGridTilingData)
}  // namespace optiling
