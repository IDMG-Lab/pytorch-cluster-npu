#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(NearestNeighbor)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(y, ge::TensorType::ALL())
    .INPUT(ptr_x, ge::TensorType::ALL())
    .INPUT(ptr_y, ge::TensorType::ALL())
    .OUTPUT(cluster, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(NearestNeighbor);

}

#endif
