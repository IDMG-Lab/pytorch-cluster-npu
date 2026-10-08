#include "graclus_npu.h"
#include "include/pytorch_npu_helper.hpp"
#include <climits>
#include <array>
#include <vector>

namespace {
thread_local std::array<int64_t,5> phaseCounts{};
void CheckWeight(torch::Tensor rowptr,int64_t e,std::optional<torch::Tensor> weight) {
  if(!weight.has_value()) return;
  const auto w=weight.value();const auto type=w.scalar_type();
  TORCH_CHECK(w.device()==rowptr.device() && w.dim()==1 && w.numel()==e,
              "graclus_npu: weight must be a matching one-dimensional NPU tensor");
  TORCH_CHECK(type==at::kHalf || type==at::kBFloat16 || type==at::kFloat ||
              type==at::kDouble || type==at::kInt || type==at::kLong,
              "graclus_npu: unsupported weight dtype");
}
at::Tensor PackWeight(torch::Tensor rowptr, int64_t e,
                      std::optional<torch::Tensor> optional_weight, int64_t& kind) {
  bool weighted = optional_weight.has_value();
  CheckWeight(rowptr,e,optional_weight);
  kind = 0;
  at::Tensor raw;
  if (weighted) {
    auto w = optional_weight.value();
    switch (w.scalar_type()) {
      case at::kHalf: case at::kBFloat16: case at::kFloat:
        raw = w.to(at::kFloat).contiguous().view(at::kInt); break;
      case at::kInt: case at::kLong:
        kind = 1; raw = w.to(at::kLong).contiguous().view(at::kInt); break;
      case at::kDouble:
        kind = 2; raw = w.contiguous().view(at::kInt); break;
      default: TORCH_CHECK(false, "graclus_npu: unsupported weight dtype");
    }
  } else {
    raw = at::empty({1}, rowptr.options().dtype(at::kInt));
  }
  return raw;
}
}

torch::Tensor graclus_npu(torch::Tensor rowptr, torch::Tensor col,
                         std::optional<torch::Tensor> optional_weight) {
  phaseCounts={0,0,0,0,0};
  TORCH_CHECK(rowptr.device().type() == c10::DeviceType::PrivateUse1 && col.device() == rowptr.device(),
              "graclus_npu: CSR inputs must be on one NPU device");
  TORCH_CHECK(rowptr.scalar_type() == at::kLong && col.scalar_type() == at::kLong &&
              rowptr.dim() == 1 && col.dim() == 1 && rowptr.numel() >= 1,
              "graclus_npu: expected one-dimensional int64 CSR inputs");
  const int64_t n = rowptr.numel() - 1, e = col.numel();
  TORCH_CHECK(n <= INT32_MAX && e <= INT32_MAX && (n > 0 || e == 0), "graclus_npu: invalid shape or tiling range");
  auto out = at::empty({n}, rowptr.options());
  if (!n) {
    CheckWeight(rowptr,e,optional_weight);
    TORCH_CHECK(rowptr.item<int64_t>()==0,"graclus_npu: invalid CSR offsets or column index");
    return out;
  }
  bool weighted = optional_weight.has_value();
  int64_t kind = 0;
  auto raw = PackWeight(rowptr,e,optional_weight,kind);
  auto rp = rowptr.contiguous(), ci = col.contiguous();
  auto options = rowptr.options().dtype(at::kInt);
  auto state = at::empty({n}, options), next = at::empty({n}, options);
  auto candidate = at::empty({n}, options), unused = at::empty({n}, options);
  auto counts = at::empty({64,16}, options);
  int64_t phase = 0;
  EXEC_NPU_CMD(aclnnGraclus, rp, ci, raw, next, unused, phase, weighted, kind,
               state, unused, counts, out);
  phaseCounts[0]++;
  auto initial = counts.cpu();
  auto initial_data = initial.data_ptr<int32_t>();
  for (int i = 0; i < 64; ++i)
    TORCH_CHECK(initial_data[i * 16 + 2] == 0, "graclus_npu: invalid CSR offsets or column index");
  if(!e || initial_data[3]==1) return out; // Empty or complete UB-resident fused result.
  bool finished = false;
  for (int iteration = 0; iteration < 64; ++iteration) {
    phase = 1;
    EXEC_NPU_CMD(aclnnGraclus, rp, ci, raw, state, unused, phase, weighted, kind,
                 next, candidate, counts, out);
    phaseCounts[1]++;
    phase = 2;
    EXEC_NPU_CMD(aclnnGraclus, rp, ci, raw, state, candidate, phase, weighted, kind,
                 next, unused, counts, out);
    phaseCounts[2]++;
    auto progress = counts.cpu();
    const auto data = progress.data_ptr<int32_t>();
    int64_t remaining = 0, changed = 0;
    for (int i = 0; i < 64; ++i) { remaining += data[i*16]; changed += data[i*16+1]; }
    std::swap(state, next);
    if (!remaining) { finished = true; break; }
    if (!changed) break;
  }
  if (!finished) {
    phase = 4;
    EXEC_NPU_CMD(aclnnGraclus, rp, ci, raw, state, unused, phase, weighted, kind,
                 next, candidate, counts, out);
    phaseCounts[3]++;
    std::swap(state, next);
  }
  phase = 3;
  EXEC_NPU_CMD(aclnnGraclus, rp, ci, raw, state, candidate, phase, weighted, kind,
               next, unused, counts, out);
  phaseCounts[4]++;
  return out;
}

// Private engineering hook: inspect the actual Select and Commit tensors.
static std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> GraclusPhaseDetails(
    at::Tensor rowptr,at::Tensor col,std::optional<at::Tensor> weight,at::Tensor state) {
  const int64_t n=rowptr.numel()-1;
  TORCH_CHECK(n>0 && rowptr.scalar_type()==at::kLong && col.scalar_type()==at::kLong &&
              state.numel()==n && state.device()==rowptr.device() && col.device()==rowptr.device(),
              "graclus phase: invalid test inputs");
  int64_t kind=0,phase=1;
  bool weighted=weight.has_value();
  auto raw=PackWeight(rowptr,col.numel(),weight,kind);
  auto rp=rowptr.contiguous(),ci=col.contiguous(),old=state.to(at::kInt).contiguous();
  auto options=rowptr.options().dtype(at::kInt);
  auto next=at::empty({n},options),candidate=at::empty({n},options),unused=at::empty({n},options);
  auto counts=at::empty({64,16},options),out=at::empty({n},rowptr.options());
  if(n*4>96*1024 && n<=128*1024*8) {
    auto hostState=state.cpu().to(at::kInt);
    auto bitmap=at::zeros({(n+31)/32},at::TensorOptions().dtype(at::kInt));
    auto words=bitmap.data_ptr<int32_t>(); const auto flags=hostState.data_ptr<int32_t>();
    for(int64_t u=0;u<n;u++) if(flags[u]>=0) words[u/32]|=int32_t(1U<<(u%32));
    unused.slice(0,0,bitmap.numel()).copy_(bitmap.to(state.device()));
  }
  EXEC_NPU_CMD(aclnnGraclus,rp,ci,raw,old,unused,phase,weighted,kind,next,candidate,counts,out);
  phase=2;
  EXEC_NPU_CMD(aclnnGraclus,rp,ci,raw,old,candidate,phase,weighted,kind,next,unused,counts,out);
  return {candidate,next,counts,unused};
}
static std::tuple<at::Tensor,at::Tensor,at::Tensor> GraclusPhase(
    at::Tensor rowptr,at::Tensor col,std::optional<at::Tensor> weight,at::Tensor state) {
  auto result=GraclusPhaseDetails(rowptr,col,weight,state);
  return {std::get<0>(result),std::get<1>(result),std::get<2>(result)};
}
static auto phase_registry=torch::RegisterOperators().op("torch_cluster::_graclus_phase",&GraclusPhase);
static auto bitmap_registry=torch::RegisterOperators().op("torch_cluster::_graclus_phase_bitmap",&GraclusPhaseDetails);
static std::vector<int64_t> GraclusPhaseCounts() {
  return std::vector<int64_t>(phaseCounts.begin(),phaseCounts.end());
}
static auto counts_registry=torch::RegisterOperators().op("torch_cluster::_graclus_phase_counts",&GraclusPhaseCounts);
