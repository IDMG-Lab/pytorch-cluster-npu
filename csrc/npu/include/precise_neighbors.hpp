#pragma once
#include "precise_dtype.hpp"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <limits>
#include <vector>

namespace torch_cluster_npu {

// Squared Euclidean distances are nonnegative IEEE doubles, so their signed
// int64 bit patterns have the same order. Sorting the bits avoids the backend's
// unsupported Double argsort without narrowing the scores. Candidate IDs are
// sorted first, which makes equal-distance selection deterministic.
inline at::Tensor score_order(const at::Tensor& squared) {
  return at::argsort(squared.contiguous().view(at::kLong), true, -1, false);
}

inline at::Tensor exact_neighbor_rows(const at::Tensor& references,
                                     const at::Tensor& queries, int64_t k) {
  const auto n=references.size(0), d=references.size(1), m=queries.size(0);
  auto result=at::empty({m,k},queries.options().dtype(at::kLong));
  // Direct subtraction avoids cancellation in difficult shortlist rows.
  const int64_t query_tile=std::max<int64_t>(1,std::min<int64_t>(32,1048576/std::max<int64_t>(1,n*d)));
  for(int64_t first=0;first<m;first+=query_tile) {
    const auto end=std::min(m,first+query_tile);
    auto delta=references.unsqueeze(0)-queries.slice(0,first,end).unsqueeze(1);
    auto scores=delta.square().sum(2);
    result.slice(0,first,end).copy_(score_order(scores).slice(1,0,k));
  }
  return result;
}

// Cube only proposes a shortlist. Double direct-subtraction scores decide its
// order; a conservative Float32 error bound sends ambiguous rows through the
// full Double search above. No coordinate data or golden result moves to CPU.
inline at::Tensor precise_neighbors(const at::Tensor& refs,const at::Tensor& query,int64_t k) {
  at::NoGradGuard guard;
  const int64_t n=refs.size(0),m=query.size(0),d=refs.size(1);
  TORCH_CHECK(n>=k && k>0,"precise_neighbors: insufficient reference points");
  TORCH_CHECK(at::isfinite(refs).all().item<bool>() && at::isfinite(query).all().item<bool>(),
              "precise_neighbors: coordinates must be finite");
  auto r=as_double(refs),q=as_double(query);
  auto output=at::empty({m,k},query.options().dtype(at::kLong));
  if(!m)return output;
  if(n<=k+32)return exact_neighbor_rows(r,q,k);
  auto origin=r.slice(0,0,1);
  auto rf=(r-origin).to(at::kFloat),qf=(q-origin).to(at::kFloat);
  auto rn=rf.square().sum(1),qn=qf.square().sum(1);
  auto rt=rf.t().contiguous();
  auto largest_norm=rn.amax();
  const int64_t shortlist=std::min<int64_t>(n,k+32);
  const int64_t tile=std::max<int64_t>(1,std::min<int64_t>({1024,8388608/n,
                                                        8388608/(shortlist*std::max<int64_t>(1,d))}));
  for(int64_t first=0;first<m;first+=tile) {
    const int64_t end=std::min(m,first+tile),count=end-first;
    auto approx=at::mm(qf.slice(0,first,end),rt)*-2;
    approx=approx+rn.unsqueeze(0)+qn.slice(0,first,end).unsqueeze(1);
    auto top=approx.topk(shortlist,1,false,true);
    auto boundary=std::get<0>(top).select(1,shortlist-1);
    auto candidates=std::get<0>(std::get<1>(top).sort(1));
    auto delta=r.index_select(0,candidates.reshape({-1})).view({count,shortlist,d})-
                 q.slice(0,first,end).unsqueeze(1);
    auto scores=delta.square().sum(2);
    auto order=score_order(scores);
    auto chosen=candidates.gather(1,order.slice(1,0,k));
    auto kth=scores.gather(1,order.slice(1,k-1,k)).reshape({-1});
    // Covers conversion, centering, norm sums and the FP32 dot product with
    // a generous gamma_d bound. Overflow/nonfinite approximations are unsafe.
    auto bound=(16.0*(d+3)*std::numeric_limits<float>::epsilon())*
               (1+largest_norm+qn.slice(0,first,end));
    auto unsafe=at::logical_or(at::logical_not(at::isfinite(boundary)),
                  at::logical_or(at::logical_not(at::isfinite(bound)),kth>=boundary-bound));
    auto ids=unsafe.nonzero().reshape({-1});
    if(ids.numel())chosen.index_copy_(0,ids,exact_neighbor_rows(r,q.slice(0,first,end).index_select(0,ids),k));
    output.slice(0,first,end).copy_(chosen);
  }
  return output;
}
} // namespace torch_cluster_npu
