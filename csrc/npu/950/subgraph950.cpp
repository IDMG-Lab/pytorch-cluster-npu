#include "kernel_operator.h"
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"

using namespace AscendC;

// Each launch is ordered on PyTorch's current stream. The map and the two
// levels of exclusive scan are rebuilt on every invocation (no input cache).
template<class I>
__simt_vf__ __aicore__ void SubgraphStage(
    __gm__ I* rp, __gm__ I* col, __gm__ I* nodes, __gm__ int32_t* map,
    __gm__ I* outRp, __gm__ I* outCol, __gm__ I* eid,
    __gm__ I* count, __gm__ I* sums, uint32_t n, uint32_t q,
    uint32_t stage, uint32_t core, uint32_t cores) {
  uint32_t tid = core * Simt::GetThreadNum() + Simt::GetThreadIdx();
  uint32_t stride = cores * Simt::GetThreadNum();
  const bool threadRows = stage >= 8;
  stage &= 7;
  if (stage == 0) {
    for (uint32_t i = tid; i < n; i += stride) map[i] = -1;
  } else if (stage == 1) {
    for (uint32_t i = tid; i < q; i += stride) map[nodes[i]] = i;
  } else if (stage == 2) {
    if (threadRows) {
      const uint32_t lane = tid % 32;
      for (uint32_t group = tid - lane; group < q; group += stride) {
        const uint32_t i = group + lane;
        I begin = 0, end = 0;
        if (i < q) { I u = nodes[i]; begin = rp[u]; end = rp[u + 1]; }
        const bool heavy = end - begin > 16;
        if (i < q && !heavy) {
          I total = 0;
          for (I j = begin; j < end; ++j) total += map[col[j]] >= 0;
          outRp[i] = total;
        }
        // A few hubs must not serialize millions of edges on one thread.
        uint32_t mask = Simt::WarpBallotSync(heavy);
        while (mask) {
          uint32_t owner = 0;
          while (!(mask & (1u << owner))) ++owner;
          I first = Simt::WarpShflSync(begin, owner);
          I last = Simt::WarpShflSync(end, owner);
          I total = 0;
          for (I base = first; base < last; base += 32) {
            I j = base + lane;
            total += Simt::Popc(Simt::WarpBallotSync(j < last && map[col[j]] >= 0));
          }
          if (lane == 0) outRp[group + owner] = total;
          mask &= mask - 1;
        }
      }
      return;
    }
    const uint32_t lane = tid % 32;
    for (uint32_t i = tid / 32; i < q; i += stride / 32) {
      I total = 0, u = nodes[i];
      for (I base = rp[u]; base < rp[u + 1]; base += 32) {
        I j = base + lane;
        int keep = j < rp[u + 1] && map[col[j]] >= 0;
        total += Simt::Popc(Simt::WarpBallotSync(keep));
      }
      if (lane == 0) outRp[i] = total;
    }
  } else if (stage == 3) {
    const uint32_t lane = tid % 32;
    for (uint32_t chunk = tid / 32; chunk < (q + 255) / 256; chunk += stride / 32) {
      I values[8];
      I total = 0;
      for (uint32_t j = 0; j < 8; ++j) {
        uint32_t i = chunk * 256 + lane * 8 + j;
        values[j] = total;
        if (i < q) total += outRp[i];
      }
      I inclusive = total;
      for (uint32_t offset = 1; offset < 32; offset *= 2) {
        I prev = Simt::WarpShflUpSync(inclusive, offset);
        if (lane >= offset) inclusive += prev;
      }
      I prefix = inclusive - total;
      for (uint32_t j = 0; j < 8; ++j) {
        uint32_t i = chunk * 256 + lane * 8 + j;
        if (i < q) outRp[i] = prefix + values[j];
      }
      if (lane == 31) sums[chunk] = inclusive;
    }
  } else if (stage == 4) {
    // Scan chunk totals in a warp, not a serial GM load/store chain.
    if (tid < 32) {
      I total = 0;
      const uint32_t chunks = (q + 255) / 256;
      for (uint32_t base = 0; base < chunks; base += 32) {
        const uint32_t i = base + tid;
        I value = i < chunks ? sums[i] : 0;
        I inclusive = value;
        for (uint32_t offset = 1; offset < 32; offset *= 2) {
          I prev = Simt::WarpShflUpSync(inclusive, offset);
          if (tid >= offset) inclusive += prev;
        }
        if (i < chunks) sums[i] = total + inclusive - value;
        total += Simt::WarpShflSync(inclusive, 31);
      }
      if (tid == 0) { count[0] = total; outRp[q] = total; }
    }
  } else {
    if (threadRows) {
      const uint32_t lane = tid % 32;
      for (uint32_t group = tid - lane; group < q; group += stride) {
        const uint32_t i = group + lane;
        I pos = 0, begin = 0, end = 0;
        if (i < q) {
          pos = outRp[i] + sums[i / 256]; outRp[i] = pos;
          I u = nodes[i]; begin = rp[u]; end = rp[u + 1];
        }
        const bool heavy = end - begin > 16;
        if (i < q && !heavy) {
          for (I j = begin; j < end; ++j) {
            int32_t mapped = map[col[j]];
            if (mapped >= 0) { outCol[pos] = mapped; eid[pos] = j; ++pos; }
          }
        }
        uint32_t heavyMask = Simt::WarpBallotSync(heavy);
        while (heavyMask) {
          uint32_t owner = 0;
          while (!(heavyMask & (1u << owner))) ++owner;
          I first = Simt::WarpShflSync(begin, owner);
          I last = Simt::WarpShflSync(end, owner);
          I destination = Simt::WarpShflSync(pos, owner);
          for (I base = first; base < last; base += 32) {
            I j = base + lane;
            int32_t mapped = j < last ? map[col[j]] : -1;
            uint32_t mask = Simt::WarpBallotSync(mapped >= 0);
            uint32_t rank = Simt::Popc(mask & ((1u << lane) - 1));
            if (mapped >= 0) { outCol[destination + rank] = mapped; eid[destination + rank] = j; }
            destination += Simt::Popc(mask);
          }
          heavyMask &= heavyMask - 1;
        }
      }
      return;
    }
    const uint32_t lane = tid % 32;
    for (uint32_t i = tid / 32; i < q; i += stride / 32) {
      I pos = 0;
      if (lane == 0) { pos = outRp[i] + sums[i / 256]; outRp[i] = pos; }
      pos = Simt::WarpShflSync(pos, 0);
      I u = nodes[i];
      for (I base = rp[u]; base < rp[u + 1]; base += 32) {
        I j = base + lane;
        int32_t mapped = j < rp[u + 1] ? map[col[j]] : -1;
        uint32_t mask = Simt::WarpBallotSync(mapped >= 0);
        uint32_t rank = Simt::Popc(mask & ((1u << lane) - 1));
        if (mapped >= 0) { outCol[pos + rank] = mapped; eid[pos + rank] = j; }
        pos += Simt::Popc(mask);
      }
    }
  }
}

template<class I>
__attribute__((aiv)) __global__ __aicore__ void Subgraph950(
    GM_ADDR rp, GM_ADDR col, GM_ADDR nodes, GM_ADDR map, GM_ADDR outRp,
    GM_ADDR outCol, GM_ADDR eid, GM_ADDR count, GM_ADDR sums,
    uint32_t n, uint32_t q, uint32_t stage) {
  Simt::VF_CALL<SubgraphStage<I>>(Simt::Dim3(256),
      (__gm__ I*)rp, (__gm__ I*)col, (__gm__ I*)nodes, (__gm__ int32_t*)map,
      (__gm__ I*)outRp, (__gm__ I*)outCol, (__gm__ I*)eid,
      (__gm__ I*)count, (__gm__ I*)sums, n, q, stage, GetBlockIdx(), GetBlockNum());
}

extern "C" void launch_subgraph950(void* rp, void* col, void* nodes, void* map,
    void* outRp, void* outCol, void* eid, void* count, void* sums,
    uint32_t n, uint32_t q, bool wide, bool threadRows, void* stream) {
  // Platform discovery is host work and can take longer than the kernels.
  // Cache only device metadata, never graph data or outputs.
  static thread_local int32_t cachedDevice = -1;
  static thread_local uint32_t cores = 0;
  int32_t device = -1;
  aclrtGetDevice(&device);
  if (!cores || device != cachedDevice) {
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    cores = platform->GetCoreNumAiv();
    if (!cores) cores = 1;
    cachedDevice = device;
  }
  for (uint32_t stage = 0; stage < 6; ++stage) {
    uint32_t blocks = stage == 4 ? 1 : cores;
    uint32_t encodedStage = stage | (threadRows ? 8 : 0);
    if (wide) Subgraph950<int64_t><<<blocks, nullptr, stream>>>(
        (GM_ADDR)rp, (GM_ADDR)col, (GM_ADDR)nodes, (GM_ADDR)map,
        (GM_ADDR)outRp, (GM_ADDR)outCol, (GM_ADDR)eid, (GM_ADDR)count,
        (GM_ADDR)sums, n, q, encodedStage);
    else Subgraph950<int32_t><<<blocks, nullptr, stream>>>(
        (GM_ADDR)rp, (GM_ADDR)col, (GM_ADDR)nodes, (GM_ADDR)map,
        (GM_ADDR)outRp, (GM_ADDR)outCol, (GM_ADDR)eid, (GM_ADDR)count,
        (GM_ADDR)sums, n, q, encodedStage);
  }
}
