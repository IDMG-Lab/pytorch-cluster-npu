#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(VoxelGridTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numPoints);    // 总点数 N
TILING_DATA_FIELD_DEF(uint32_t, dim);          // 坐标维度 D (如 3)
TILING_DATA_FIELD_DEF(uint32_t, blockPoints);  // 每个 Core 处理的点数
TILING_DATA_FIELD_DEF(uint32_t, tailPoints);   // 最后一个 Core 处理的点数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(VoxelGrid, VoxelGridTilingData)
}  // namespace optiling