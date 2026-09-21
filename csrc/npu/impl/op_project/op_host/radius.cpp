// /**
//  * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
//  * This file is a part of the CANN Open Software.
//  * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// /**
//  * @file radius.cpp
//  */
#include "radius_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdio>

namespace optiling
{
  static ge::graphStatus TilingFunc(gert::TilingContext *context)
  {
    RadiusTilingData tiling;
    const gert::StorageShape *x_shape = context->GetInputShape(0);
    const gert::StorageShape *y_shape = context->GetInputShape(1);
    const gert::StorageShape *ptr_x_shape = context->GetOptionalInputShape(2);
    const gert::StorageShape *ptr_y_shape = context->GetOptionalInputShape(3);
    // 输入改为 SoA 布局：x=[F, N]，y=[F, M]（由中间层 transpose 得到），便于向量化分块
    uint32_t itemLength = x_shape->GetStorageShape().GetDim(0);
    uint32_t xSize = x_shape->GetStorageShape().GetDim(1);
    uint32_t ySize = y_shape->GetStorageShape().GetDim(1);

    auto attrs = context->GetAttrs();
    float r = *attrs->GetFloat(0);
    uint32_t max_num_neighbors = *attrs->GetInt(1);
    uint32_t ignore_same_index = *attrs->GetBool(2);
    uint32_t ptrXLen = 0;
    uint32_t ptrYLen = 0;
    if (ptr_x_shape != nullptr)
    {
      ptrXLen = ptr_x_shape->GetStorageShape().GetShapeSize();
    }
    if (ptr_y_shape != nullptr)
    {
      ptrYLen = ptr_y_shape->GetStorageShape().GetShapeSize();
    }
    // context->SetBlockDim(1);
    // 多核实现
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t workItems = (ptrYLen > 1) ? (ptrYLen - 1) : ySize; // 无batch按y切，有batch按batch切
    uint32_t blockDim = (workItems == 0) ? 1 : ((workItems < aivNum) ? workItems : aivNum);
    // context->SetBlockDim(blockDim);
    context->SetBlockDim(aivNum);
    tiling.set_ptrXLen(ptrXLen);
    tiling.set_ptrYLen(ptrYLen);
    tiling.set_xSize(xSize);
    tiling.set_ySize(ySize);
    tiling.set_itemLength(itemLength);
    tiling.set_r(r);
    tiling.set_max_num_neighbors(max_num_neighbors);
    tiling.set_ignore_same_index(ignore_same_index);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
  }
}

namespace ge
{
  static ge::graphStatus InferShape(gert::InferShapeContext *context)
  {
    return GRAPH_SUCCESS;
  }
  static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
  {
    const ge::DataType x1_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x1_dtype);
    return GRAPH_SUCCESS;
  }
}

namespace ops
{
  class Radius : public OpDef
  {
  public:
    explicit Radius(const char *name) : OpDef(name)
    {
      this->Input("x")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("y")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("ptr_x")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      this->Input("ptr_y")
          .ParamType(OPTIONAL)
          .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      // ---- hash-grid 内部输入（由中间层构造，不对外暴露 torch.ops 接口）----
      // 各段内按 cell 排序后的 x（SoA，段间按 ptr_x 连续拼接）
      this->Input("sorted_x")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      // 排序位置 -> 全局原 x 下标
      this->Input("order")
          .ParamType(REQUIRED)
          .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      // 各段 cell 起止（段间拼接）
      this->Input("cell_start")
          .ParamType(REQUIRED)
          .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      // 各段在 cell_start 中的偏移 [B+1]
      this->Input("cell_start_off")
          .ParamType(REQUIRED)
          .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      // 各段各维网格原点 [B*F]
      this->Input("grid_min")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      // 各段各维 cell 数 [B*F]
      this->Input("grid_g")
          .ParamType(REQUIRED)
          .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      // 1=走 hash-grid，0=回退暴力
      this->Input("use_grid")
          .ParamType(REQUIRED)
          .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      this->Output("out")
          .ParamType(REQUIRED)
          .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
      this->Attr("r").Float();
      this->Attr("max_num_neighbors").Int();
      this->Attr("ignore_same_index").Bool();

      this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

      this->AICore()
          .SetTiling(optiling::TilingFunc);
      this->AICore().AddConfig("ascend910b");
    }
  };

  OP_ADD(Radius);
}

// 单核
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// /**
//  * @file radius.cpp
//  */
// #include "radius_tiling.h"
// #include "register/op_def_registry.h"
// #include <cstdio>

// namespace optiling
// {
//   static ge::graphStatus TilingFunc(gert::TilingContext *context)
//   {
//     RadiusTilingData tiling;
//     const gert::StorageShape *x_shape = context->GetInputShape(0);
//     const gert::StorageShape *y_shape = context->GetInputShape(1);
//     const gert::StorageShape *ptr_x_shape = context->GetOptionalInputShape(2);
//     const gert::StorageShape *ptr_y_shape = context->GetOptionalInputShape(3);
//     uint32_t xSize = x_shape->GetStorageShape().GetDim(0);
//     uint32_t itemLength = x_shape->GetStorageShape().GetDim(1);
//     uint32_t ySize = y_shape->GetStorageShape().GetDim(0);

//     auto attrs = context->GetAttrs();
//     float r = *attrs->GetFloat(0);
//     uint32_t max_num_neighbors = *attrs->GetInt(1);
//     uint32_t ignore_same_index = *attrs->GetBool(2);
//     uint32_t ptrXLen = 0;
//     uint32_t ptrYLen = 0;
//     if (ptr_x_shape != nullptr)
//     {
//       ptrXLen = ptr_x_shape->GetStorageShape().GetShapeSize();
//     }
//     if (ptr_y_shape != nullptr)
//     {
//       ptrYLen = ptr_y_shape->GetStorageShape().GetShapeSize();
//     }
//     context->SetBlockDim(1);
//     tiling.set_ptrXLen(ptrXLen);
//     tiling.set_ptrYLen(ptrYLen);
//     tiling.set_xSize(xSize);
//     tiling.set_ySize(ySize);
//     tiling.set_itemLength(itemLength);
//     tiling.set_r(r);
//     tiling.set_max_num_neighbors(max_num_neighbors);
//     tiling.set_ignore_same_index(ignore_same_index);
//     tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
//     context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

//     return ge::GRAPH_SUCCESS;
//   }
// }

// namespace ge
// {
//   static ge::graphStatus InferShape(gert::InferShapeContext *context)
//   {
//     return GRAPH_SUCCESS;
//   }
//   static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
//   {
//     const ge::DataType x1_dtype = context->GetInputDataType(0);
//     context->SetOutputDataType(0, x1_dtype);
//     return GRAPH_SUCCESS;
//   }
// }

// namespace ops
// {
//   class Radius : public OpDef
//   {
//   public:
//     explicit Radius(const char *name) : OpDef(name)
//     {
//       this->Input("x")
//           .ParamType(REQUIRED)
//           .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
//           .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
//           .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
//       this->Input("y")
//           .ParamType(REQUIRED)
//           .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
//           .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
//           .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
//       this->Input("ptr_x")
//           .ParamType(OPTIONAL)
//           .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
//           .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
//           .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
//       this->Input("ptr_y")
//           .ParamType(OPTIONAL)
//           .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
//           .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
//           .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
//       this->Output("out")
//           .ParamType(REQUIRED)
//           .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
//           .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
//           .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
//       this->Attr("r").Float();
//       this->Attr("max_num_neighbors").Int();
//       this->Attr("ignore_same_index").Bool();

//       this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

//       this->AICore()
//           .SetTiling(optiling::TilingFunc);
//       this->AICore().AddConfig("ascend910b");
//     }
//   };

//   OP_ADD(Radius);
// }