#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(VoxelGrid)
    .INPUT(pos, ge::TensorType::ALL())
    .INPUT(size, ge::TensorType::ALL())
    .OPTIONAL_INPUT(start, ge::TensorType::ALL())
    .OPTIONAL_INPUT(end, ge::TensorType::ALL())
    .OUTPUT(cluster, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(VoxelGrid);

}

#endif
