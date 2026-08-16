#include "farthest_point_sampling_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {

constexpr uint32_t FPS_FIXED_BLOCK_DIM = 40;

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    FarthestPointSamplingTilingData tiling;

    // src 已经是 padded SoA: [dim, padded_N]
    auto srcShape = context->GetInputShape(0)->GetStorageShape();

    uint32_t dim = static_cast<uint32_t>(srcShape.GetDim(0));
    uint32_t totalPoints = static_cast<uint32_t>(srcShape.GetDim(1));

    auto ptrShape = context->GetInputShape(1)->GetStorageShape();
    uint32_t batchSize = static_cast<uint32_t>(ptrShape.GetDim(0) - 1);

    tiling.set_dim(dim);
    tiling.set_totalPoints(totalPoints);
    tiling.set_batchSize(batchSize);

    // 统一固定 40 block。
    //
    // mode=0 batch-parallel:
    //   block_id, block_id+40, ... grid-stride 处理 batch。
    //
    // mode=1 inner-40:
    //   40 个 block 协同处理每个 batch。
    context->SetBlockDim(FPS_FIXED_BLOCK_DIM);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class FarthestPointSampling : public OpDef {
public:
    explicit FarthestPointSampling(const char* name) : OpDef(name) {
        this->Input("src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->Input("ptr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});

        this->Input("padded_point_ptr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});

        this->Input("out_ptr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});

        this->Input("padded_out_ptr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});

        this->Input("start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});

        // int32[1], 0=batch-parallel, 1=inner-40
        this->Input("mode")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});

        // GM workspace: [padded_N]
        this->Input("dist")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        // [40 * 8] float, 每核一个 32B slot。
        this->Input("local_max_val")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        // [40 * 4] int64, 每核一个 32B slot。
        this->Input("local_max_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});

        // [4] int64, 32B chosen slot。
        this->Input("chosen_local")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});

        // [40 * 8] int32, SyncAll soft workspace。
        this->Input("sync_workspace")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});

        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(FarthestPointSampling);

}