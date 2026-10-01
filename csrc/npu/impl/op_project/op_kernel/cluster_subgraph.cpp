#include "kernel_operator.h"

using namespace AscendC;

// Large CSR subgraphs run map/count/prefix/scatter as ordered launches.
// Count partitions and output partitions own complete 64-byte cache lines.
// Small subgraphs retain the lower-overhead single-launch implementation.
class KernelClusterSubgraph {
public:
  __aicore__ inline void Init(GM_ADDR rowptr, GM_ADDR col, GM_ADDR nodes,
                              GM_ADDR mapping, GM_ADDR outRowptr,
                              GM_ADDR outCol, GM_ADDR outEid, GM_ADDR outCount,
                              uint32_t n, uint32_t e, uint32_t q, uint32_t phase) {
    this->n = n;
    this->e = e;
    this->q = q;
    this->phase = phase;
    rp.SetGlobalBuffer((__gm__ int64_t*)rowptr, n + 1);
    ci.SetGlobalBuffer((__gm__ int64_t*)col, e);
    nd.SetGlobalBuffer((__gm__ int64_t*)nodes, q);
    map.SetGlobalBuffer((__gm__ int32_t*)mapping, n);
    outRp.SetGlobalBuffer((__gm__ int64_t*)outRowptr, q + 1);
    outCi.SetGlobalBuffer((__gm__ int64_t*)outCol, e);
    outEi.SetGlobalBuffer((__gm__ int64_t*)outEid, e);
    countGm.SetGlobalBuffer((__gm__ int64_t*)outCount, 1);
    pipe.InitBuffer(fillBuffer, 4096 * sizeof(int32_t));
    pipe.InitBuffer(localMapBuffer, 16384 * sizeof(int32_t));
    useLocal = phase == 4 && n >= 4096 && n <= 16384;
  }

  __aicore__ inline int32_t Lookup(int64_t node) {
    return useLocal ? localMap.GetValue(node) : map.GetValue(node);
  }

  __aicore__ inline void BuildMap() {
    uint32_t mapBegin = 0, mapEnd = n;
    if (phase == 0 && GetBlockNum() > 1) {
      // One owner per complete 64-byte line avoids both initialization
      // versus scatter races and dirty DCache-line sharing between cores.
      const uint64_t groups = (static_cast<uint64_t>(n) + 15) / 16;
      mapBegin = 16 * (groups * GetBlockIdx() / GetBlockNum());
      const uint64_t boundary = 16 * (groups * (GetBlockIdx() + 1) / GetBlockNum());
      mapEnd = boundary < n ? boundary : n;
    }
    if (useLocal) {
      localMap = localMapBuffer.Get<int32_t>();
      Duplicate(localMap, -1, (n + 7) / 8 * 8);
      PipeBarrier<PIPE_ALL>();
    } else if (n < 4096) {
      for (uint32_t i = mapBegin; i < mapEnd; ++i) map.SetValue(i, -1);
    } else {
      auto fill = fillBuffer.Get<int32_t>();
      Duplicate(fill, -1, 4096);
      PipeBarrier<PIPE_ALL>();
      for (uint32_t off = mapBegin; off < mapEnd; off += 4096) {
        const uint32_t len = mapEnd - off < 4096 ? mapEnd - off : 4096;
        DataCopyExtParams params{1, static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(map[off], fill, params);
      }
      PipeBarrier<PIPE_ALL>();
    }
    for (uint32_t i = 0; i < q; ++i) {
      const int64_t u = nd.GetValue(i);
      if (u >= mapBegin && u < mapEnd) {
        if (useLocal) localMap.SetValue(u, i);
        else map.SetValue(u, i);
      }
    }
  }

  __aicore__ inline void CountRows() {
    const uint32_t groups = (q + 7) / 8;
    const uint32_t first = 8 * (static_cast<uint64_t>(groups) * GetBlockIdx() / GetBlockNum());
    const uint32_t last = 8 * (static_cast<uint64_t>(groups) * (GetBlockIdx() + 1) / GetBlockNum());
    for (uint32_t i = first; i < last && i < q; ++i) {
      int64_t count = 0;
      const int64_t u = nd.GetValue(i);
      if (u >= 0 && u < n) {
        int64_t begin = rp.GetValue(u);
        int64_t end = rp.GetValue(u + 1);
        if (begin < 0) begin = 0;
        if (end > e) end = e;
        for (int64_t edge = begin; edge < end; ++edge) {
          const int64_t v = ci.GetValue(edge);
          if (v >= 0 && v < n && map.GetValue(v) >= 0) ++count;
        }
      }
      outRp.SetValue(i, count);
    }
  }

  __aicore__ inline void PrefixRows() {
    int64_t total = 0;
    for (uint32_t i = 0; i < q; ++i) {
      const int64_t count = outRp.GetValue(i);
      outRp.SetValue(i, total);
      total += count;
    }
    outRp.SetValue(q, total);
    countGm.SetValue(0, total);
  }

  __aicore__ inline void ScatterOutputs() {
    const uint64_t total = countGm.GetValue(0);
    const uint64_t groups = (total + 7) / 8;
    const uint64_t first = 8 * (groups * GetBlockIdx() / GetBlockNum());
    const uint64_t boundary = 8 * (groups * (GetBlockIdx() + 1) / GetBlockNum());
    const uint64_t last = boundary < total ? boundary : total;
    if (first >= last) return;
    // Locate the first contributing CSR row for this output partition.
    uint32_t lo = 0, hi = q;
    while (lo < hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (static_cast<uint64_t>(outRp.GetValue(mid + 1)) <= first) lo = mid + 1;
      else hi = mid;
    }
    for (uint32_t row = lo; row < q; ++row) {
      uint64_t position = outRp.GetValue(row);
      if (position >= last) break;
      if (static_cast<uint64_t>(outRp.GetValue(row + 1)) <= first) continue;
      const int64_t u = nd.GetValue(row);
      if (u < 0 || u >= n) continue;
      int64_t begin = rp.GetValue(u);
      int64_t end = rp.GetValue(u + 1);
      if (begin < 0) begin = 0;
      if (end > e) end = e;
      for (int64_t edge = begin; edge < end && position < last; ++edge) {
        const int64_t v = ci.GetValue(edge);
        if (v < 0 || v >= n) continue;
        const int32_t mapped = map.GetValue(v);
        if (mapped < 0) continue;
        if (position >= first) {
          outCi.SetValue(position, mapped);
          outEi.SetValue(position, edge);
        }
        ++position;
      }
    }
  }

  __aicore__ inline void SinglePass() {
    BuildMap();
    int64_t count = 0;
    outRp.SetValue(0, 0);
    for (uint32_t i = 0; i < q; ++i) {
      const int64_t u = nd.GetValue(i);
      if (u >= 0 && u < n) {
        int64_t begin = rp.GetValue(u);
        int64_t end = rp.GetValue(u + 1);
        if (begin < 0) begin = 0;
        if (end > e) end = e;
        for (int64_t j = begin; j < end; ++j) {
          const int64_t v = ci.GetValue(j);
          if (v >= 0 && v < n) {
            const int64_t mapped = Lookup(v);
            if (mapped >= 0) {
              outCi.SetValue(count, mapped);
              outEi.SetValue(count, j);
              ++count;
            }
          }
        }
      }
      outRp.SetValue(i + 1, count);
    }
    countGm.SetValue(0, count);
  }

  __aicore__ inline void Process() {
    DataCacheCleanAndInvalid<int64_t, CacheLine::ENTIRE_DATA_CACHE>(outRp);
    if (phase == 0) BuildMap();
    else if (phase == 1) CountRows();
    else if (phase == 2) PrefixRows();
    else if (phase == 3) ScatterOutputs();
    else SinglePass();
    DataCacheCleanAndInvalid<int64_t, CacheLine::ENTIRE_DATA_CACHE>(outRp);
  }

private:
  TPipe pipe;
  TBuf<QuePosition::VECCALC> fillBuffer;
  TBuf<QuePosition::VECCALC> localMapBuffer;
  LocalTensor<int32_t> localMap;
  GlobalTensor<int64_t> rp, ci, nd, outRp, outCi, outEi, countGm;
  GlobalTensor<int32_t> map;
  uint32_t n, e, q, phase;
  bool useLocal;
};

extern "C" __global__ __aicore__ void cluster_subgraph(
    GM_ADDR rowptr, GM_ADDR col, GM_ADDR nodes, GM_ADDR mapping,
    GM_ADDR outRowptr, GM_ADDR outCol, GM_ADDR outEid, GM_ADDR outCount,
    GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(data, tiling);
  KernelClusterSubgraph op;
  op.Init(rowptr, col, nodes, mapping, outRowptr, outCol, outEid,
          outCount, data.numNodes, data.numEdges, data.subsetSize, data.phase);
  op.Process();
}
