#pragma once
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <optional>

namespace torch_cluster_npu {
// Integer index outputs cannot represent an undefined NaN/Inf geometry.
// Validate on device and only read one Boolean; coordinate data stays on NPU.
inline void check_finite_coordinates(
    const at::Tensor& x,const std::optional<at::Tensor>& y=std::nullopt) {
  if(!x.is_floating_point())return;
  at::NoGradGuard guard;
  auto values=x.reshape({-1});
  if(y && !(x.data_ptr()==y->data_ptr() && x.sizes()==y->sizes() && x.strides()==y->strides()))
    values=at::cat({values,y->reshape({-1})});
  TORCH_CHECK(at::isfinite(values).all().item<bool>(),
              "NPU geometry coordinates must be finite");
}
} // namespace torch_cluster_npu
