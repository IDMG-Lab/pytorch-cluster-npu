#include "kernel_operator.h"
using namespace AscendC;

// Single-core correctness baseline reproduces the CPU permutation and greedy
// matching semantics. The fixed permutation makes CPU/NPU labels comparable.
class KernelGraclus {
public:
  __aicore__ inline void Init(GM_ADDR rowptr, GM_ADDR col, GM_ADDR weight,
                              GM_ADDR permutation, GM_ADDR out,
                              uint32_t n, uint32_t e, uint32_t weighted) {
    this->n = n;
    this->e = e;
    this->weighted = weighted != 0;
    rp.SetGlobalBuffer((__gm__ int64_t*)rowptr, n + 1);
    ci.SetGlobalBuffer((__gm__ int64_t*)col, e);
    weights.SetGlobalBuffer((__gm__ float*)weight, e);
    perm.SetGlobalBuffer((__gm__ int64_t*)permutation, n);
    cluster.SetGlobalBuffer((__gm__ int64_t*)out, n);
    clusterWords.SetGlobalBuffer((__gm__ int32_t*)out,
                                 static_cast<uint64_t>(n) * 2);
    pipe.InitBuffer(fillBuffer, 512 * sizeof(int32_t));
    pipe.InitBuffer(localBuffer, 16384 * sizeof(int32_t));
    useLocal = n >= 2048 && n <= 16384;
  }

  template <bool Local>
  __aicore__ inline int64_t ReadCluster(int64_t u) {
    return Local ? localCluster.GetValue(u) : cluster.GetValue(u);
  }

  template <bool Local>
  __aicore__ inline void WriteCluster(int64_t u, int64_t label) {
    if (Local) localCluster.SetValue(u, static_cast<int32_t>(label));
    else cluster.SetValue(u, label);
  }

  template <bool Local, bool Weighted>
  __aicore__ inline void ProcessMatching() {
    if (Local) {
      localCluster = localBuffer.Get<int32_t>();
      for (uint32_t u = 0; u < n; ++u) localCluster.SetValue(u, -1);
    } else if (n < 4096) {
      for (uint32_t u = 0; u < n; ++u) cluster.SetValue(u, -1);
    } else {
      auto fill = fillBuffer.Get<int32_t>();
      Duplicate(fill, -1, 512);
      PipeBarrier<PIPE_ALL>();
      for (uint32_t off = 0; off < n; off += 256) {
        const uint32_t len = n - off < 256 ? n - off : 256;
        DataCopyExtParams params{1, static_cast<uint32_t>(2 * len * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(clusterWords[static_cast<uint64_t>(off) * 2], fill, params);
      }
      PipeBarrier<PIPE_ALL>();
    }
    for (uint32_t position = 0; position < n; ++position) {
      const int64_t u = perm.GetValue(position);
      if (u < 0 || u >= n || ReadCluster<Local>(u) >= 0) continue;
      const int64_t first = rp.GetValue(u);
      const int64_t last = rp.GetValue(u + 1);
      if (Weighted) {
        int64_t chosen = u;
        float best = 0.0f;
        for (int64_t edge = first; edge < last; ++edge) {
          // A lower weight cannot change the greedy choice. Test it
          // before the random destination/cluster reads; keep >= ties.
          const float value = weights.GetValue(edge);
          if (!(value >= best)) continue;
          const int64_t v = ci.GetValue(edge);
          if (v < 0 || v >= n || ReadCluster<Local>(v) >= 0) continue;
          best = value;
          chosen = v;
        }
        const int64_t label = u < chosen ? u : chosen;
        WriteCluster<Local>(u, label);
        WriteCluster<Local>(chosen, label);
      } else {
        WriteCluster<Local>(u, u);
        for (int64_t edge = first; edge < last; ++edge) {
          const int64_t v = ci.GetValue(edge);
          if (v < 0 || v >= n || ReadCluster<Local>(v) >= 0) continue;
          const int64_t label = u < v ? u : v;
          WriteCluster<Local>(u, label);
          WriteCluster<Local>(v, label);
          break;
        }
      }
    }
    if (Local) {
      for (uint32_t u = 0; u < n; ++u)
        cluster.SetValue(u, localCluster.GetValue(u));
    }
    DataCacheCleanAndInvalid<int64_t, CacheLine::ENTIRE_DATA_CACHE>(cluster);
  }

  __aicore__ inline void Process() {
    if (useLocal) {
      if (weighted) ProcessMatching<true, true>();
      else ProcessMatching<true, false>();
    } else {
      if (weighted) ProcessMatching<false, true>();
      else ProcessMatching<false, false>();
    }
  }

private:
  TPipe pipe;
  TBuf<QuePosition::VECCALC> fillBuffer;
  TBuf<QuePosition::VECCALC> localBuffer;
  LocalTensor<int32_t> localCluster;
  GlobalTensor<int32_t> clusterWords;
  GlobalTensor<int64_t> rp, ci, perm, cluster;
  GlobalTensor<float> weights;
  uint32_t n, e;
  bool weighted;
  bool useLocal;
};

extern "C" __global__ __aicore__ void graclus(
    GM_ADDR rowptr, GM_ADDR col, GM_ADDR weight, GM_ADDR permutation,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(data, tiling);
  KernelGraclus op;
  op.Init(rowptr, col, weight, permutation, out,
          data.numNodes, data.numEdges, data.weighted);
  op.Process();
}
