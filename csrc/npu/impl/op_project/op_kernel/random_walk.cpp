#include "kernel_operator.h"
using namespace AscendC;

class KernelRandomWalk {
public:
  __aicore__ inline void Init(GM_ADDR rowptr, GM_ADDR col, GM_ADDR start,
                              GM_ADDR random, GM_ADDR walks, GM_ADDR edgeIds,
                              GM_ADDR flags,
                              uint32_t n, uint32_t e, uint32_t s, uint32_t len,
                              uint32_t nodesPitch, uint32_t edgesPitch,
                              uint32_t walksPerCore, uint32_t phase,
                              uint32_t checkBlocks, float p, float q) {
    this->n = n;
    this->e = e;
    this->s = s;
    this->len = len;
    this->phase = phase;
    this->checkBlocks = checkBlocks;
    this->degreeOne = false;
    this->nodesPitch = nodesPitch;
    this->edgesPitch = edgesPitch;
    this->first = GetBlockIdx() * walksPerCore;
    this->last = first + walksPerCore < s ? first + walksPerCore : s;
    this->uniform = p == 1.0f && q == 1.0f;
    this->localGraph = n <= 4096 && e <= 8192 && len >= 8;
    this->returnWeight = 1.0f / p;
    this->farWeight = 1.0f / q;
    rp.SetGlobalBuffer((__gm__ int64_t*)rowptr, n + 1);
    ci.SetGlobalBuffer((__gm__ int64_t*)col, e);
    starts.SetGlobalBuffer((__gm__ int64_t*)start, s);
    rnd.SetGlobalBuffer((__gm__ float*)random, static_cast<uint64_t>(s) * len);
    walkGm.SetGlobalBuffer((__gm__ int64_t*)walks,
                           static_cast<uint64_t>(s) * nodesPitch);
    edgeGm.SetGlobalBuffer((__gm__ int64_t*)edgeIds,
                           static_cast<uint64_t>(s) * edgesPitch);
    graphFlags.SetGlobalBuffer((__gm__ int32_t*)flags, 640);
    if (localGraph) {
      pipe.InitBuffer(rowBuffer, ((n + 4) / 4 * 4) * sizeof(int64_t));
      pipe.InitBuffer(colBuffer, ((e + 3) / 4 * 4 + 4) * sizeof(int64_t));
    }
  }

  template <bool Local>
  __aicore__ inline int64_t Rowptr(int64_t u) {
    return Local ? localRows.GetValue(u) : rp.GetValue(u);
  }

  template <bool Local>
  __aicore__ inline int64_t Column(int64_t edge) {
    return Local ? localCols.GetValue(edge) : ci.GetValue(edge);
  }

  template <bool Local>
  __aicore__ inline bool HasEdge(int64_t u, int64_t v) {
    for (int64_t i = Rowptr<Local>(u); i < Rowptr<Local>(u + 1); ++i)
      if (Column<Local>(i) == v) return true;
    return false;
  }

  template <bool Local>
  __aicore__ inline float Weight(int64_t previous, int64_t candidate) {
    if (candidate == previous) return returnWeight;
    return HasEdge<Local>(previous, candidate) ? 1.0f : farWeight;
  }

  template <bool Uniform, bool Local, bool DegreeOne = false>
  __aicore__ inline void ProcessWalks() {
    for (uint32_t walk = first; walk < last; ++walk) {
      int64_t current = starts.GetValue(walk);
      int64_t previous = -1;
      const uint64_t nodeBase = static_cast<uint64_t>(walk) * nodesPitch;
      const uint64_t edgeBase = static_cast<uint64_t>(walk) * edgesPitch;
      walkGm.SetValue(nodeBase, current);
      for (uint32_t step = 0; step < len; ++step) {
        int64_t edge = -1;
        if (current >= 0 && current < n) {
          if (DegreeOne) {
            edge = current;
          } else {
          const int64_t begin = Rowptr<Local>(current);
          const int64_t end = Rowptr<Local>(current + 1);
          const int64_t degree = end - begin;
          if (degree > 0) {
            if (degree == 1) {
              edge = begin;
            } else {
              float u = rnd.GetValue(static_cast<uint64_t>(walk) * len + step);
              if (u < 0.0f) u = 0.0f;
              if (u >= 1.0f) u = 0.99999994f;
              if (Uniform || previous < 0) {
                int64_t offset = static_cast<int64_t>(u * degree);
                if (offset >= degree) offset = degree - 1;
                edge = begin + offset;
              } else {
                float total = 0.0f;
                for (int64_t i = begin; i < end; ++i)
                  total += Weight<Local>(previous, Column<Local>(i));
                const float target = u * total;
                float cumulative = 0.0f;
                edge = end - 1;
                for (int64_t i = begin; i < end; ++i) {
                  cumulative += Weight<Local>(previous, Column<Local>(i));
                  if (target < cumulative) { edge = i; break; }
                }
              }
            }
          }
          }
        }
        const int64_t next = edge < 0 ? current : Column<Local>(edge);
        edgeGm.SetValue(edgeBase + step, edge);
        walkGm.SetValue(nodeBase + step + 1, next);
        if (!Uniform) previous = current;
        current = next;
      }
    }
    DataCacheCleanAndInvalid<int64_t, CacheLine::ENTIRE_DATA_CACHE>(walkGm);
  }

  template <bool Local>
  __aicore__ inline void DispatchWalks() {
    if (degreeOne) ProcessWalks<true, Local, true>();
    else if (uniform) ProcessWalks<true, Local>();
    else ProcessWalks<false, Local>();
  }

  __aicore__ inline void Process() {
    if (phase == 0) {
      // Each validator owns a complete 64-byte flag line. All n+1 row
      // pointers must equal their positions; e==n is checked by the adapter.
      const uint64_t begin = (static_cast<uint64_t>(n) + 1) * GetBlockIdx() / checkBlocks;
      const uint64_t end = (static_cast<uint64_t>(n) + 1) * (GetBlockIdx() + 1) / checkBlocks;
      bool valid = true;
      for (uint64_t i = begin; i < end; ++i) {
        if (rp.GetValue(i) != static_cast<int64_t>(i)) { valid = false; break; }
      }
      graphFlags.SetValue(GetBlockIdx() * 16, valid ? 1 : 0);
      DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(graphFlags);
      return;
    }
    if (phase == 2) {
      degreeOne = true;
      for (uint32_t i = 0; i < checkBlocks; ++i) {
        if (graphFlags.GetValue(i * 16) != 1) { degreeOne = false; break; }
      }
    }
    if (localGraph) {
      localRows = rowBuffer.Get<int64_t>();
      localCols = colBuffer.Get<int64_t>();
      DataCopyExtParams rowCopy{1, static_cast<uint32_t>((n + 1) * sizeof(int64_t)), 0, 0, 0};
      DataCopyPad(localRows, rp, rowCopy, DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
      if (e) {
        DataCopyExtParams colCopy{1, static_cast<uint32_t>(e * sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(localCols, ci, colCopy, DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
      }
      PipeBarrier<PIPE_ALL>();
      DispatchWalks<true>();
    } else {
      DispatchWalks<false>();
    }
  }

private:
  TPipe pipe;
  TBuf<QuePosition::VECCALC> rowBuffer, colBuffer;
  LocalTensor<int64_t> localRows, localCols;
  GlobalTensor<int64_t> rp, ci, starts, walkGm, edgeGm;
  GlobalTensor<float> rnd;
  GlobalTensor<int32_t> graphFlags;
  uint32_t n, e, s, len, nodesPitch, edgesPitch, first, last;
  float returnWeight, farWeight;
  uint32_t phase, checkBlocks;
  bool uniform, localGraph, degreeOne;
};

extern "C" __global__ __aicore__ void random_walk(
    GM_ADDR rowptr, GM_ADDR col, GM_ADDR start, GM_ADDR random,
    GM_ADDR walks, GM_ADDR edgeIds, GM_ADDR flags, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(data, tiling);
  KernelRandomWalk op;
  op.Init(rowptr, col, start, random, walks, edgeIds, flags,
          data.numNodes, data.numEdges, data.numWalks, data.walkLength,
          data.nodesPitch, data.edgesPitch, data.walksPerCore,
          data.phase, data.checkBlocks, data.p, data.q);
  op.Process();
}
