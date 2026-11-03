#pragma once
#include <ATen/ATen.h>

namespace torch_cluster_npu {
// Tensor::to(Double) in torch_npu 2.7 may silently choose Float32. Allocate
// the requested dtype explicitly and copy on the same device instead.
inline at::Tensor as_double(const at::Tensor& source) {
  if (source.scalar_type() == at::kDouble) return source;
  auto result = at::empty_like(source, source.options().dtype(at::kDouble));
  result.copy_(source);
  TORCH_CHECK(result.scalar_type() == at::kDouble && result.device() == source.device(),
              "NPU precise computation requires an actual device Double tensor");
  return result;
}
} // namespace torch_cluster_npu
