#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(FarthestPointSampling)
    .INPUT(src, ge::TensorType::ALL())
    .INPUT(ptr, ge::TensorType::ALL())
    .INPUT(padded_point_ptr, ge::TensorType::ALL())
    .INPUT(out_ptr, ge::TensorType::ALL())
    .INPUT(padded_out_ptr, ge::TensorType::ALL())
    .INPUT(start, ge::TensorType::ALL())
    .INPUT(mode, ge::TensorType::ALL())
    .INPUT(dist, ge::TensorType::ALL())
    .INPUT(local_max_val, ge::TensorType::ALL())
    .INPUT(local_max_idx, ge::TensorType::ALL())
    .INPUT(chosen_local, ge::TensorType::ALL())
    .INPUT(sync_workspace, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(FarthestPointSampling);

}

#endif
