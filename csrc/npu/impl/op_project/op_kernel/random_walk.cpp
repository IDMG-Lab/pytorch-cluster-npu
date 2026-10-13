#include "kernel_operator.h"
using namespace AscendC;

template <typename IndexT>
class KernelRandomWalk {
public:
  __aicore__ inline void Init(GM_ADDR rowptr, GM_ADDR col, GM_ADDR start,
                              GM_ADDR random, GM_ADDR walks, GM_ADDR edgeIds,
                              GM_ADDR flags,
                              uint32_t n, uint32_t e, uint32_t s, uint32_t len,
                              uint32_t nodesPitch, uint32_t edgesPitch,
                              uint32_t walksPerCore, uint32_t phase,
                              uint32_t checkBlocks, float p, float q,
                              uint32_t outputMode, GM_ADDR workspace) {
    this->n = n;
    this->e = e;
    this->s = s;
    this->len = len;
    this->phase = phase;
    this->checkBlocks = checkBlocks;
    this->outputMode = outputMode;
    this->walkSpan = walksPerCore;
    this->shardWorkspace = workspace;
    this->degreeOne = false;
    this->regularDegree = 0;
    this->trustedGraph = false;
    this->nodesPitch = nodesPitch;
    this->edgesPitch = edgesPitch;
    this->first = GetBlockIdx() * walksPerCore;
    this->last = first + walksPerCore < s ? first + walksPerCore : s;
    this->uniform = p == 1.0f && q == 1.0f;
    this->localGraph = n <= 4096 && e <= 8192 && len >= 8;
    this->returnWeight = 1.0f / p;
    this->farWeight = 1.0f / q;
    rp.SetGlobalBuffer((__gm__ IndexT*)rowptr, n + 1);
    ci.SetGlobalBuffer((__gm__ IndexT*)col, e);
    ciCache.SetGlobalBuffer((__gm__ uint64_t*)col);
    starts.SetGlobalBuffer((__gm__ IndexT*)start, s);
    rnd.SetGlobalBuffer((__gm__ float*)random, static_cast<uint64_t>(s) * len);
    walkGm.SetGlobalBuffer((__gm__ IndexT*)walks,
                           static_cast<uint64_t>(outputMode ? len + 1 : s) * nodesPitch);
    edgeGm.SetGlobalBuffer((__gm__ IndexT*)edgeIds,
                           static_cast<uint64_t>(outputMode ? len : s) * edgesPitch);
    graphFlags.SetGlobalBuffer((__gm__ int32_t*)flags, 640);
    if (phase == 4) {
      // Private exact Philox probe, independent from public sampling. Input
      // rowptr carries seed_lo/seed_hi/offset_lo/offset_hi. Output graph_flags
      // aliases a float32 [len,s] tensor. Separate queues overlap output DMA.
      pipe.InitBuffer(rngStateBuffer, 10 * kRngLanes * 4);
      pipe.InitBuffer(rngOutQueue, 2, 4 * kRngLanes * 4);
      if (!outputMode) pipe.InitBuffer(rngIndexBuffer, 4 * kRngLanes * 4);
      return;
    }
    if (outputMode && phase != 0) {
      // All three cooperative phases reuse one arena. No old sampler UB is
      // live alongside the 128 KiB column shard.
      pipe.InitBuffer(shardBuffer, 163840);
      return;
    }
    if (phase == 3) {
      pipe.InitBuffer(streamBuffer, 16384 * sizeof(IndexT));
      return;
    }
    if (phase == 0) {
      pipe.InitBuffer(checkBuffer, 512 * sizeof(IndexT));
      if constexpr (sizeof(IndexT) == 4) {
        pipe.InitBuffer(boundsFloatBuffer, 512 * sizeof(float));
        pipe.InitBuffer(boundsWorkBuffer, 512 * sizeof(float));
        pipe.InitBuffer(boundsResultBuffer, 8 * sizeof(float));
      }
    }
    if (uniform) {
      pipe.InitBuffer(randomBuffer, kWalkBatch * kStepTile * sizeof(float));
      pipe.InitBuffer(nodeBuffer, kWalkBatch * kNodeTile * sizeof(IndexT));
      pipe.InitBuffer(edgeBuffer, kWalkBatch * kStepTile * sizeof(IndexT));
      if constexpr (sizeof(IndexT) == 4)
        pipe.InitBuffer(offsetBuffer, kWalkBatch * kStepTile * sizeof(int32_t));
    }
    if constexpr (sizeof(IndexT) == 4) {
      if (uniform && phase == 2 && !localGraph) {
        pipe.InitBuffer(hybridRandomBuffer, kHybridBatch * kHybridSteps * 4);
        pipe.InitBuffer(hybridOffsetBuffer, kHybridBatch * kHybridSteps * 4);
        pipe.InitBuffer(hybridNodesBuffer, kHybridBatch * kHybridNodePitch * 4);
        pipe.InitBuffer(hybridEdgesBuffer, kHybridBatch * kHybridSteps * 4);
        pipe.InitBuffer(hybridOutputBuffer, kHybridBatch * kHybridNodePitch * 4);
        pipe.InitBuffer(hybridIndicesBuffer, kHybridBatch * (kHybridSteps + kHybridNodePitch) * 4);
        pipe.InitBuffer(hybridStateBuffer, kHybridBatch * 4 * 4);
      }
    }
    if (localGraph) {
      pipe.InitBuffer(rowBuffer, ((n + 1 + 31 / sizeof(IndexT)) / (32 / sizeof(IndexT)) * (32 / sizeof(IndexT))) * sizeof(IndexT));
      pipe.InitBuffer(colBuffer, ((e + 31 / sizeof(IndexT)) / (32 / sizeof(IndexT)) * (32 / sizeof(IndexT)) + 32 / sizeof(IndexT)) * sizeof(IndexT));
    }
  }

  __aicore__ inline void RngXor(LocalTensor<int32_t> dst,
                               LocalTensor<int32_t> left,
                               LocalTensor<int32_t> right,
                               LocalTensor<int32_t> tmp, uint32_t count) {
    // C220 exposes 16-bit bitwise operations. OR - AND is XOR on the full
    // word (the bit fields are disjoint); integer Sub wraps modulo 2**32.
    Or(tmp.ReinterpretCast<uint16_t>(), left.ReinterpretCast<uint16_t>(),
       right.ReinterpretCast<uint16_t>(), count * 2);
    And(dst.ReinterpretCast<uint16_t>(), left.ReinterpretCast<uint16_t>(),
        right.ReinterpretCast<uint16_t>(), count * 2);
    PipeBarrier<PIPE_V>();
    Sub(dst, tmp, dst, count);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void RngMultiply(LocalTensor<int32_t> low,
                                    LocalTensor<int32_t> high,
                                    LocalTensor<int32_t> input,
                                    uint32_t multiplier, uint32_t count) {
    auto state = rngStateBuffer.Get<int32_t>();
    auto lo = state[4 * kRngLanes];
    auto hi = state[5 * kRngLanes];
    auto mid = state[6 * kRngLanes];
    auto carry = state[7 * kRngLanes];
    auto mask = state[8 * kRngLanes];
    const int32_t ml = static_cast<int32_t>(multiplier & 65535U);
    const int32_t mh = static_cast<int32_t>(multiplier >> 16);
    And(lo.ReinterpretCast<uint16_t>(), input.ReinterpretCast<uint16_t>(),
        mask.ReinterpretCast<uint16_t>(), count * 2);
    ShiftRight(hi.ReinterpretCast<uint32_t>(), input.ReinterpretCast<uint32_t>(), 16U, count);
    PipeBarrier<PIPE_V>();
    Muls(carry, lo, ml, count);
    Muls(mid, lo, mh, count);
    PipeBarrier<PIPE_V>();
    ShiftRight(carry.ReinterpretCast<uint32_t>(), carry.ReinterpretCast<uint32_t>(), 16U, count);
    PipeBarrier<PIPE_V>();
    Add(mid, mid, carry, count);
    PipeBarrier<PIPE_V>();
    ShiftRight(high.ReinterpretCast<uint32_t>(), mid.ReinterpretCast<uint32_t>(), 16U, count);
    And(carry.ReinterpretCast<uint16_t>(), mid.ReinterpretCast<uint16_t>(),
        mask.ReinterpretCast<uint16_t>(), count * 2);
    PipeBarrier<PIPE_V>();
    Muls(mid, hi, ml, count);
    PipeBarrier<PIPE_V>();
    Add(mid, mid, carry, count);
    Muls(carry, hi, mh, count);
    PipeBarrier<PIPE_V>();
    ShiftRight(mid.ReinterpretCast<uint32_t>(), mid.ReinterpretCast<uint32_t>(), 16U, count);
    PipeBarrier<PIPE_V>();
    Add(high, high, mid, count);
    PipeBarrier<PIPE_V>();
    Add(high, high, carry, count);
    Muls(low, input, static_cast<int32_t>(multiplier), count);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void ProcessRandom() {
    const uint32_t key0 = static_cast<uint32_t>(rp.GetValue(0));
    const uint32_t key1 = static_cast<uint32_t>(rp.GetValue(1));
    const uint32_t offset0 = static_cast<uint32_t>(rp.GetValue(2));
    const uint32_t offset1 = static_cast<uint32_t>(rp.GetValue(3));
    // The first round's c2 is the uniform offset; its product is constant.
    // This makes c0 of round 1 constant too. Hoist both full products out of
    // the vector loop while preserving every Philox round and output bit.
    const uint64_t constantB = static_cast<uint64_t>(offset0) * 0xcd9e8d57U;
    const uint32_t nextC0 = static_cast<uint32_t>(constantB >> 32) ^ key0;
    const uint64_t constantA = static_cast<uint64_t>(nextC0) * 0xd2511f53U;
    const uint32_t round0Key = offset1 ^ key1;
    const uint32_t round1Key0 = static_cast<uint32_t>(constantB) ^ (key0 + 0x9e3779b9U);
    const uint32_t round1Key1 = static_cast<uint32_t>(constantA >> 32) ^ (key1 + 0xbb67ae85U);
    auto state = rngStateBuffer.Get<int32_t>();
    LocalTensor<uint32_t> packedIndices;
    if (!outputMode) {
      packedIndices = rngIndexBuffer.Get<uint32_t>();
      // Philox produces four word planes. Convert their flat output order
      // once, before the round scratch becomes live, for contiguous DMA.
      auto row = state;
      auto mask = state[4 * kRngLanes];
      CreateVecIndex(packedIndices.ReinterpretCast<int32_t>(), int32_t(0), 4 * kRngLanes);
      Duplicate(mask, int32_t(3), 4 * kRngLanes);
      PipeBarrier<PIPE_V>();
      ShiftRight(row.ReinterpretCast<uint32_t>(), packedIndices, 2U, 4 * kRngLanes);
      And(packedIndices.ReinterpretCast<uint16_t>(), packedIndices.ReinterpretCast<uint16_t>(),
          mask.ReinterpretCast<uint16_t>(), 8 * kRngLanes);
      PipeBarrier<PIPE_V>();
      Muls(row, row, int32_t(4), 4 * kRngLanes);
      Muls(packedIndices.ReinterpretCast<int32_t>(), packedIndices.ReinterpretCast<int32_t>(),
           int32_t(kRngLanes * 4), 4 * kRngLanes);
      PipeBarrier<PIPE_V>();
      Add(packedIndices.ReinterpretCast<int32_t>(), packedIndices.ReinterpretCast<int32_t>(),
          row, 4 * kRngLanes);
      PipeBarrier<PIPE_ALL>();
    }
    auto maskLow = state[8 * kRngLanes];
    auto maskFloat = state[9 * kRngLanes];
    Duplicate(maskLow, 65535, kRngLanes);
    Duplicate(maskFloat, 0x7fffff, kRngLanes);
    GlobalTensor<float> output;
    output.SetGlobalBuffer((__gm__ float*)graphFlags.GetPhyAddr(), static_cast<uint64_t>(s) * len);
    const uint32_t groups = static_cast<uint64_t>(s) * len / 4;
    // A flat counter group emits four adjacent values. Align ownership to
    // cache lines so distinct cores never write the same DMA destination.
    const uint32_t flatSpan = ((groups + GetBlockNum() - 1) / GetBlockNum() + 15) / 16 * 16;
    const uint32_t begin = outputMode ? first : GetBlockIdx() * flatSpan;
    const uint32_t end = outputMode ? last : (begin + flatSpan < groups ? begin + flatSpan : groups);
    for (uint32_t walk = begin; walk < end; walk += kRngLanes) {
      const uint32_t count = end - walk < kRngLanes ? end - walk : kRngLanes;
      const uint32_t calc = (count + 63) / 64 * 64;
      for (uint32_t group = 0; group < (outputMode ? len / 4 : 1); ++group) {
        auto out = rngOutQueue.AllocTensor<float>();
        auto initial = out.ReinterpretCast<int32_t>();
        CreateVecIndex(initial, static_cast<int32_t>(walk), calc);
        PipeBarrier<PIPE_V>();
        if (outputMode) {
          Muls(initial, initial, static_cast<int32_t>(len / 4), calc);
          PipeBarrier<PIPE_V>();
          Adds(initial, initial, static_cast<int32_t>(group), calc);
          PipeBarrier<PIPE_V>();
        }
        auto tmp = state[4 * kRngLanes];
        auto key = state[5 * kRngLanes];
        RngMultiply(state[3 * kRngLanes], state[2 * kRngLanes], initial, 0xd2511f53U, calc);
        Duplicate(key, static_cast<int32_t>(round0Key), calc); PipeBarrier<PIPE_V>();
        RngXor(state[2 * kRngLanes], state[2 * kRngLanes], key, tmp, calc);
        RngMultiply(initial[kRngLanes], initial, state[2 * kRngLanes], 0xcd9e8d57U, calc);
        Duplicate(key, static_cast<int32_t>(round1Key0), calc); PipeBarrier<PIPE_V>();
        RngXor(initial, initial, key, tmp, calc);
        Duplicate(key, static_cast<int32_t>(round1Key1), calc); PipeBarrier<PIPE_V>();
        RngXor(initial[2 * kRngLanes], state[3 * kRngLanes], key, tmp, calc);
        Duplicate(initial[3 * kRngLanes], static_cast<int32_t>(constantA), calc);
        PipeBarrier<PIPE_V>();
        uint32_t k0 = key0 + 2U * 0x9e3779b9U, k1 = key1 + 2U * 0xbb67ae85U;
        for (uint32_t round = 2; round < 10; ++round) {
          auto src = (round & 1) ? state : initial;
          auto dst = (round & 1) ? initial : state;
          RngMultiply(dst[3 * kRngLanes], dst[2 * kRngLanes], src, 0xd2511f53U, calc);
          RngMultiply(dst[kRngLanes], dst, src[2 * kRngLanes], 0xcd9e8d57U, calc);
          RngXor(dst, dst, src[kRngLanes], tmp, calc);
          Duplicate(key, static_cast<int32_t>(k0), calc); PipeBarrier<PIPE_V>();
          RngXor(dst, dst, key, tmp, calc);
          RngXor(dst[2 * kRngLanes], dst[2 * kRngLanes], src[3 * kRngLanes], tmp, calc);
          Duplicate(key, static_cast<int32_t>(k1), calc); PipeBarrier<PIPE_V>();
          RngXor(dst[2 * kRngLanes], dst[2 * kRngLanes], key, tmp, calc);
          k0 += 0x9e3779b9U;
          k1 += 0xbb67ae85U;
        }
        for (uint32_t word = 0; word < 4; ++word) {
          auto values = initial[word * kRngLanes];
          And(values.ReinterpretCast<uint16_t>(), values.ReinterpretCast<uint16_t>(),
              maskFloat.ReinterpretCast<uint16_t>(), calc * 2);
          PipeBarrier<PIPE_V>();
          Adds(values, values, 0x3f800000, calc); PipeBarrier<PIPE_V>();
          Adds(values.ReinterpretCast<float>(), values.ReinterpretCast<float>(), -1.0f, calc);
        }
        if (outputMode) {
          rngOutQueue.EnQue(out);
          auto ready = rngOutQueue.DeQue<float>();
          DataCopyExtParams cp{4, count * 4, (kRngLanes - (count + 7) / 8 * 8) / 8,
                               (s - count) * 4, 0};
          DataCopyPad(output[static_cast<uint64_t>(group * 4) * s + walk], ready, cp);
          rngOutQueue.FreeTensor(ready);
        } else {
          PipeBarrier<PIPE_V>();
          auto packed = rngOutQueue.AllocTensor<float>();
          Gather(packed, out, packedIndices, 0, count * 4);
          rngOutQueue.EnQue(packed);
          auto ready = rngOutQueue.DeQue<float>();
          DataCopyExtParams cp{1, count * 16, 0, 0, 0};
          DataCopyPad(output[static_cast<uint64_t>(walk) * 4], ready, cp);
          rngOutQueue.FreeTensor(ready);
          rngOutQueue.FreeTensor(out);
        }
      }
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
      const uint64_t nodeBase = outputMode ? walk : static_cast<uint64_t>(walk) * nodesPitch;
      const uint64_t edgeBase = outputMode ? walk : static_cast<uint64_t>(walk) * edgesPitch;
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
              float u = rnd.GetValue(outputMode ? static_cast<uint64_t>(step) * s + walk : static_cast<uint64_t>(walk) * len + step);
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
        edgeGm.SetValue(edgeBase + (outputMode ? static_cast<uint64_t>(step) * edgesPitch : step), edge);
        walkGm.SetValue(nodeBase + (outputMode ? static_cast<uint64_t>(step + 1) * nodesPitch : step + 1), next);
        if (!Uniform) previous = current;
        current = next;
      }
    }
    DataCacheCleanAndInvalid<IndexT, CacheLine::ENTIRE_DATA_CACHE>(walkGm);
  }

  template <bool Local>
  __aicore__ inline void ProcessUniformBatch() {
    // Walks are independent. Keep their state and output in UB; DMA handles
    // contiguous RNG/output tiles instead of one scalar GM access per value.
    auto randomTile = randomBuffer.Get<float>();
    auto nodeTile = nodeBuffer.Get<IndexT>();
    auto edgeTile = edgeBuffer.Get<IndexT>();
    for (uint32_t base = first; base < last; base += kWalkBatch) {
      const uint32_t batch = last - base < kWalkBatch ? last - base : kWalkBatch;
      int64_t current[kWalkBatch];
      int64_t selected[kWalkBatch];
      for (uint32_t b = 0; b < batch; ++b) current[b] = starts.GetValue(base + b);
      for (uint32_t tile = 0; tile < len; tile += kStepTile) {
        const uint32_t steps = len - tile < kStepTile ? len - tile : kStepTile;
        for (uint32_t b = 0; b < batch; ++b) {
          DataCopyExtParams cp{1, static_cast<uint32_t>(steps * sizeof(float)), 0, 0, 0};
          DataCopyPad(randomTile[b * kStepTile], rnd[static_cast<uint64_t>(base + b) * len + tile],
                      cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
          if (tile == 0) nodeTile.SetValue(b * kNodeTile, current[b]);
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t step = 0; step < steps; ++step) {
          for (uint32_t b = 0; b < batch; ++b) {
            const int64_t u = current[b];
            int64_t edge = -1;
            if (u >= 0 && u < n) {
              if (regularDegree) {
                const float draw = randomTile.GetValue(b * kStepTile + step);
                int64_t offset = static_cast<int64_t>(draw * regularDegree);
                if (offset >= regularDegree) offset = regularDegree - 1;
                edge = u * regularDegree + offset;
              }
              else if (degreeOne) edge = u;
              else {
                const int64_t begin = Rowptr<Local>(u);
                const int64_t end = Rowptr<Local>(u + 1);
                if (end > begin) {
                  float randomValue = randomTile.GetValue(b * kStepTile + step);
                  randomValue = randomValue < 0 ? 0 : randomValue;
                  randomValue = randomValue >= 1 ? 0.99999994f : randomValue;
                  int64_t offset = static_cast<int64_t>(randomValue * (end - begin));
                  if (offset >= end - begin) offset = end - begin - 1;
                  edge = begin + offset;
                }
              }
            }
            selected[b] = edge;
            edgeTile.SetValue(b * kStepTile + step, edge);
            if constexpr (!Local) {
              if (edge >= 0) DataCachePreload(ciCache[(edge * sizeof(IndexT) / 64) * 8], static_cast<int16_t>(0));
            }
          }
          for (uint32_t b = 0; b < batch; ++b) {
            const int64_t next = selected[b] >= 0 ? Column<Local>(selected[b]) : current[b];
            nodeTile.SetValue(b * kNodeTile + step + (tile == 0 ? 1 : 0), next);
            current[b] = next;
          }
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t b = 0; b < batch; ++b) {
          const uint32_t nodeCount = steps + (tile == 0 ? 1 : 0);
          DataCopyExtParams nodesCopy{1, static_cast<uint32_t>(nodeCount * sizeof(IndexT)), 0, 0, 0};
          DataCopyPad(walkGm[static_cast<uint64_t>(base + b) * nodesPitch + tile + (tile == 0 ? 0 : 1)],
                      nodeTile[b * kNodeTile], nodesCopy);
          DataCopyExtParams edgesCopy{1, static_cast<uint32_t>(steps * sizeof(IndexT)), 0, 0, 0};
          DataCopyPad(edgeGm[static_cast<uint64_t>(base + b) * edgesPitch + tile],
                      edgeTile[b * kStepTile], edgesCopy);
        }
        PipeBarrier<PIPE_ALL>();
      }
    }
  }

  template <bool Local, bool Trusted = false, int32_t ConstantDegree = 0, bool FullBatch = false>
  __aicore__ inline void ProcessUniformRaw() {
    auto randomTile = randomBuffer.Get<float>();
    auto offsets = offsetBuffer.Get<int32_t>();
    auto nodes = nodeBuffer.Get<int32_t>();
    auto edges = edgeBuffer.Get<int32_t>();
    __ubuf__ int32_t* offsetPtr = (__ubuf__ int32_t*)offsets.GetPhyAddr();
    __ubuf__ int32_t* nodePtr = (__ubuf__ int32_t*)nodes.GetPhyAddr();
    __ubuf__ int32_t* edgePtr = (__ubuf__ int32_t*)edges.GetPhyAddr();
    const __gm__ IndexT* columnPtr = ci.GetPhyAddr();
    __ubuf__ IndexT* localColumnPtr = nullptr;
    if constexpr (Local) localColumnPtr = (__ubuf__ IndexT*)localCols.GetPhyAddr();
    const int32_t degree = ConstantDegree ? ConstantDegree : static_cast<int32_t>(regularDegree);
    for (uint32_t base = first; base < last; base += kWalkBatch) {
      const uint32_t batch = FullBatch ? kWalkBatch : (last - base < kWalkBatch ? last - base : kWalkBatch);
      int32_t current[kWalkBatch];
      int32_t chosen[kWalkBatch];
      for (uint32_t b = 0; b < batch; ++b) current[b] = starts.GetValue(base + b);
      for (uint32_t tile = 0; tile < len; tile += kStepTile) {
        const uint32_t steps = len - tile < kStepTile ? len - tile : kStepTile;
        for (uint32_t b = 0; b < batch; ++b) {
          DataCopyExtParams cp{1, steps * 4, 0, 0, 0};
          DataCopyPad(randomTile[b * kStepTile], rnd[static_cast<uint64_t>(base + b) * len + tile],
                      cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
          if (tile == 0) nodePtr[b * kNodeTile] = current[b];
        }
        PipeBarrier<PIPE_ALL>();
        Muls(randomTile, randomTile, static_cast<float>(degree), kWalkBatch * kStepTile);
        PipeBarrier<PIPE_V>();
        Cast(offsets, randomTile, RoundMode::CAST_FLOOR, kWalkBatch * kStepTile);
        PipeBarrier<PIPE_V>();
        Mins(offsets, offsets, degree - 1, kWalkBatch * kStepTile);
        PipeBarrier<PIPE_ALL>();
        if constexpr (Trusted) {
          #pragma unroll
          for (uint32_t b = 0; b < kWalkBatch; ++b) {
            if (b < batch) {
              chosen[b] = current[b] * degree + offsetPtr[b * kStepTile];
              if constexpr (!Local)
                DataCachePreload(ciCache, static_cast<int64_t>(chosen[b]) * 4);
            }
          }
          for (uint32_t step = 0; step < steps; ++step) {
            #pragma unroll
            for (uint32_t b = 0; b < kWalkBatch; ++b) {
              if (b < batch) {
                const int32_t edge = chosen[b];
                int32_t next;
                if constexpr (Local) next = localColumnPtr[edge];
                else next = columnPtr[edge];
                edgePtr[b * kStepTile + step] = edge;
                nodePtr[b * kNodeTile + step + (tile == 0 ? 1 : 0)] = next;
                if (step + 1 < steps) {
                  chosen[b] = next * degree + offsetPtr[b * kStepTile + step + 1];
                  if constexpr (!Local)
                    DataCachePreload(ciCache, static_cast<int64_t>(chosen[b]) * 4);
                }
              }
            }
          }
          #pragma unroll
          for (uint32_t b = 0; b < kWalkBatch; ++b)
            if (b < batch) current[b] = nodePtr[b * kNodeTile + steps - 1 + (tile == 0 ? 1 : 0)];
        } else for (uint32_t step = 0; step < steps; ++step) {
          #pragma unroll
          for (uint32_t b = 0; b < kWalkBatch; ++b) {
            if (b < batch) {
              const int32_t u = current[b];
              const int32_t edge = static_cast<uint32_t>(u) < n
                  ? u * static_cast<int32_t>(regularDegree) + offsetPtr[b * kStepTile + step] : -1;
              chosen[b] = edge;
              if constexpr (!Local) {
                if (edge >= 0) DataCachePreload(ciCache, static_cast<int64_t>(edge) * 4);
              }
            }
          }
          #pragma unroll
          for (uint32_t b = 0; b < kWalkBatch; ++b) {
            if (b < batch) {
              const int32_t edge = chosen[b];
              int32_t next = current[b];
              if (edge >= 0) {
                if constexpr (Local) next = localColumnPtr[edge];
                else next = columnPtr[edge];
              }
              edgePtr[b * kStepTile + step] = edge;
              nodePtr[b * kNodeTile + step + (tile == 0 ? 1 : 0)] = next;
              current[b] = next;
            }
          }
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t b = 0; b < batch; ++b) {
          DataCopyExtParams nodesCopy{1, (steps + (tile == 0 ? 1U : 0U)) * 4, 0, 0, 0};
          DataCopyPad(walkGm[static_cast<uint64_t>(base + b) * nodesPitch + tile + (tile == 0 ? 0 : 1)],
                      nodes[b * kNodeTile], nodesCopy);
          DataCopyExtParams edgesCopy{1, steps * 4, 0, 0, 0};
          DataCopyPad(edgeGm[static_cast<uint64_t>(base + b) * edgesPitch + tile], edges[b * kStepTile], edgesCopy);
        }
        PipeBarrier<PIPE_ALL>();
      }
    }
  }

  __aicore__ inline void ProcessUniformHybrid() {
    auto draws = hybridRandomBuffer.Get<float>();
    auto offsets = hybridOffsetBuffer.Get<int32_t>();
    auto nodes = hybridNodesBuffer.Get<int32_t>();
    auto edges = hybridEdgesBuffer.Get<int32_t>();
    auto output = hybridOutputBuffer.Get<int32_t>();
    auto indices = hybridIndicesBuffer.Get<uint32_t>();
    auto initialIndices = indices[kHybridBatch * kHybridSteps];
    auto current = hybridStateBuffer.Get<int32_t>();
    auto drawOffsets = current[kHybridBatch];
    auto rngIndices = current[kHybridBatch * 2].ReinterpretCast<uint32_t>();
    auto columnIndices = current[kHybridBatch * 3].ReinterpretCast<uint32_t>();
    __ubuf__ uint32_t* indexPtr = (__ubuf__ uint32_t*)indices.GetPhyAddr();
    __ubuf__ uint32_t* initialPtr = (__ubuf__ uint32_t*)initialIndices.GetPhyAddr();
    for (uint32_t b = 0; b < kHybridBatch; ++b) {
      for (uint32_t t = 0; t < kHybridSteps; ++t)
        indexPtr[b * kHybridSteps + t] = (t * kHybridBatch + b) * 4;
      for (uint32_t t = 0; t < kHybridNodePitch; ++t)
        initialPtr[b * kHybridNodePitch + t] = ((t < kHybridSteps + 1 ? t : 0) * kHybridBatch + b) * 4;
    }
    CreateVecIndex(rngIndices.ReinterpretCast<int32_t>(), int32_t(0), kHybridBatch);
    CreateVecIndex(columnIndices.ReinterpretCast<int32_t>(), int32_t(0), kHybridBatch);
    PipeBarrier<PIPE_V>();
    Muls(rngIndices.ReinterpretCast<int32_t>(), rngIndices.ReinterpretCast<int32_t>(), int32_t(kHybridSteps * 4), kHybridBatch);
    Muls(columnIndices.ReinterpretCast<int32_t>(), columnIndices.ReinterpretCast<int32_t>(), int32_t(32), kHybridBatch);
    PipeBarrier<PIPE_ALL>();
    const __gm__ IndexT* column = ci.GetPhyAddr();
    for (uint32_t base = first; base < last; base += kHybridBatch) {
      const uint32_t batch = last - base < kHybridBatch ? last - base : kHybridBatch;
      DataCopyExtParams startCopy{1, batch * 4, 0, 0, 0};
      DataCopyPad(current, starts[base], startCopy, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
      PipeBarrier<PIPE_ALL>();
      for (uint32_t tile = 0; tile < len; tile += kHybridSteps) {
        const uint32_t steps = len - tile < kHybridSteps ? len - tile : kHybridSteps;
        // A 2-D transfer issues all walk rows with one MTE descriptor.
        DataCopyExtParams randomCopy{static_cast<uint16_t>(batch), steps * 4, (len - steps) * 4,
            (kHybridSteps - (steps + 7) / 8 * 8) / 8, 0};
        DataCopyPad(draws, rnd[static_cast<uint64_t>(base) * len + tile], randomCopy,
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Muls(draws, draws, static_cast<float>(static_cast<int32_t>(regularDegree)), kHybridBatch * kHybridSteps);
        PipeBarrier<PIPE_V>();
        Cast(offsets, draws, RoundMode::CAST_FLOOR, kHybridBatch * kHybridSteps);
        PipeBarrier<PIPE_V>();
        Mins(offsets, offsets, static_cast<int32_t>(regularDegree - 1), kHybridBatch * kHybridSteps);
        PipeBarrier<PIPE_V>();
        const uint32_t initial = tile == 0 ? 1 : 0;
        if (initial) { Adds(nodes, current, int32_t(0), batch); PipeBarrier<PIPE_V>(); }
        for (uint32_t step = 0; step < steps; ++step) {
          auto state = step ? nodes[(step - 1 + initial) * kHybridBatch] : current;
          auto next = nodes[(step + initial) * kHybridBatch];
          auto selected = edges[step * kHybridBatch];
          Gather(drawOffsets, offsets, rngIndices, step * 4, batch);
          Muls(selected, state, static_cast<int32_t>(regularDegree), batch);
          PipeBarrier<PIPE_V>();
          Add(selected, selected, drawOffsets, batch);
          PipeBarrier<PIPE_ALL>();
          const __ubuf__ int32_t* __restrict__ selectedPtr = (__ubuf__ int32_t*)selected.GetPhyAddr();
          __ubuf__ int32_t* __restrict__ copyDest = (__ubuf__ int32_t*)output.GetPhyAddr();
          __gm__ IndexT* __restrict__ copySource = (__gm__ IndexT*)ci.GetPhyAddr();
          if ASCEND_IS_AIV {
            uint32_t b = 0;
            for (; b + 8 <= batch; b += 8) {
              #pragma unroll
              for (uint32_t k = 0; k < 8; ++k)
                copy_gm_to_ubuf_align_b32(copyDest + (b + k) * 8, copySource + selectedPtr[b + k], 0, 1, 4, 0, 0, 0, 0);
            }
            for (; b < batch; ++b)
              copy_gm_to_ubuf_align_b32(copyDest + b * 8, copySource + selectedPtr[b], 0, 1, 4, 0, 0, 0, 0);
          }
          PipeBarrier<PIPE_ALL>();
          Gather(next, output, columnIndices, 0, batch);
          PipeBarrier<PIPE_V>();
        }
        Adds(current, nodes[(steps - 1 + initial) * kHybridBatch], int32_t(0), batch);
        PipeBarrier<PIPE_V>();
        // Transpose whole tiles in Vector; DMA all output rows together.
        const uint32_t nodeCount = steps + initial;
        auto nodeIndices = initial ? initialIndices : indices;
        const uint32_t localPitch = initial ? kHybridNodePitch : kHybridSteps;
        Gather(output, nodes, nodeIndices, 0, batch * localPitch);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams nodeCopy{static_cast<uint16_t>(batch), nodeCount * 4,
            (localPitch - (nodeCount + 7) / 8 * 8) / 8, (nodesPitch - nodeCount) * 4, 0};
        DataCopyPad(walkGm[static_cast<uint64_t>(base) * nodesPitch + tile + (initial ? 0 : 1)], output, nodeCopy);
        PipeBarrier<PIPE_ALL>();
        Gather(output, edges, indices, 0, batch * kHybridSteps);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams edgeCopy{static_cast<uint16_t>(batch), steps * 4,
            (kHybridSteps - (steps + 7) / 8 * 8) / 8, (edgesPitch - steps) * 4, 0};
        DataCopyPad(edgeGm[static_cast<uint64_t>(base) * edgesPitch + tile], output, edgeCopy);
        PipeBarrier<PIPE_ALL>();
      }
    }
  }

  template <bool Local>
  __aicore__ inline void DispatchWalks() {
    if (uniform) {
      if constexpr (sizeof(IndexT) == 4) {
        if (regularDegree) {
          if constexpr (!Local) {
            if (trustedGraph) { ProcessUniformHybrid(); return; }
          }
          if (trustedGraph && regularDegree == 8) {
            const uint32_t savedFirst = first, savedLast = last;
            const uint32_t split = first + (last - first) / kWalkBatch * kWalkBatch;
            last = split;
            ProcessUniformRaw<Local, true, 8, true>();
            first = split;
            last = savedLast;
            ProcessUniformRaw<Local, true, 8, false>();
            first = savedFirst;
          }
          else if (trustedGraph) ProcessUniformRaw<Local, true>();
          else ProcessUniformRaw<Local, false>();
          return;
        }
      }
      ProcessUniformBatch<Local>();
    }
    else if (degreeOne) ProcessWalks<true, Local, true>();
    else ProcessWalks<false, Local>();
  }

  __aicore__ inline void RestoreRecords(__ubuf__ uint64_t* records, uint32_t count) {
    // Replies carry the exact negative FP32 UB address as their merge key.
    // Every UB address is below 2^24. Large queues still restore all entries.
    const __ubuf__ float* keys = (const __ubuf__ float*)records;
    uint32_t i = 0;
    for (; i + 8 <= count; i += 8) {
      const uint64_t a = records[i], b = records[i + 1], c = records[i + 2], d = records[i + 3];
      const uint64_t e = records[i + 4], f = records[i + 5], g = records[i + 6], h = records[i + 7];
      *((__ubuf__ int32_t*)static_cast<uint64_t>(static_cast<int32_t>(-keys[i * 2]))) = static_cast<int32_t>(a >> 32);
      *((__ubuf__ int32_t*)static_cast<uint64_t>(static_cast<int32_t>(-keys[(i + 1) * 2]))) = static_cast<int32_t>(b >> 32);
      *((__ubuf__ int32_t*)static_cast<uint64_t>(static_cast<int32_t>(-keys[(i + 2) * 2]))) = static_cast<int32_t>(c >> 32);
      *((__ubuf__ int32_t*)static_cast<uint64_t>(static_cast<int32_t>(-keys[(i + 3) * 2]))) = static_cast<int32_t>(d >> 32);
      *((__ubuf__ int32_t*)static_cast<uint64_t>(static_cast<int32_t>(-keys[(i + 4) * 2]))) = static_cast<int32_t>(e >> 32);
      *((__ubuf__ int32_t*)static_cast<uint64_t>(static_cast<int32_t>(-keys[(i + 5) * 2]))) = static_cast<int32_t>(f >> 32);
      *((__ubuf__ int32_t*)static_cast<uint64_t>(static_cast<int32_t>(-keys[(i + 6) * 2]))) = static_cast<int32_t>(g >> 32);
      *((__ubuf__ int32_t*)static_cast<uint64_t>(static_cast<int32_t>(-keys[(i + 7) * 2]))) = static_cast<int32_t>(h >> 32);
    }
    for (; i < count; ++i) {
      const uint64_t record = records[i];
      *((__ubuf__ int32_t*)static_cast<uint64_t>(static_cast<int32_t>(-keys[i * 2]))) = static_cast<int32_t>(record >> 32);
    }
  }

  __aicore__ inline void MergeFixedReplies(LocalTensor<float> sorted,
                                          LocalTensor<float> scratch,
                                          uint32_t total, uint32_t capacity) {
    auto src = sorted, dst = scratch;
    bool inScratch = false;
    for (uint32_t run = capacity; run < total; run *= 4) {
      const uint32_t groups = total / (run * 4);
      uint32_t consumed[4];
      if (groups) {
        const uint16_t sizes[4] = {static_cast<uint16_t>(run), static_cast<uint16_t>(run),
                                   static_cast<uint16_t>(run), static_cast<uint16_t>(run)};
        MrgSort(dst,
                MrgSortSrcList<float>{src, src[run * 2], src[run * 4], src[run * 6]},
                sizes, consumed, uint16_t(15), groups);
      }
      const uint32_t base = groups * run * 4;
      const uint32_t tail = total - base;
      if (tail) {
        if (tail <= run) DataCopy(dst[base * 2], src[base * 2], tail * 2);
        else {
          uint16_t sizes[4] = {0, 0, 0, 0};
          uint16_t valid = 0;
          for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t start = i * run;
            if (start < tail) {
              sizes[i] = static_cast<uint16_t>(tail - start < run ? tail - start : run);
              valid |= static_cast<uint16_t>(1U << i);
            }
          }
          MrgSort(dst[base * 2], MrgSortSrcList<float>{src[base * 2],
                  src[(base + run) * 2], src[(base + run * 2) * 2], src[(base + run * 3) * 2]},
                  sizes, consumed, valid, 1);
        }
      }
      PipeBarrier<PIPE_V>();
      auto previous = src;
      src = dst; dst = previous;
      inScratch = !inScratch;
    }
    if (inScratch) { DataCopy(sorted, src, total * 2); PipeBarrier<PIPE_V>(); }
  }

  __aicore__ inline void StoreShardBucket(const LocalTensor<int32_t>& selected,
                                         const LocalTensor<int32_t>& fixed,
                                         const LocalTensor<int32_t>& meta,
                                         uint32_t chosen, uint32_t part,
                                         uint32_t capacity, uint32_t& cursor,
                                         uint64_t forwardBase, uint32_t queuePitch) {
    meta.SetValue(part * 2, chosen);
    meta.SetValue(part * 2 + 1, cursor);
    const uint32_t fast = chosen < capacity ? chosen : capacity;
    if (fast) { Adds(fixed[part * capacity], selected, int32_t(0), fast); PipeBarrier<PIPE_V>(); }
    if (chosen > capacity) {
      PipeBarrier<PIPE_ALL>();
      DataCopyPad(graphFlags[forwardBase + GetBlockIdx() * queuePitch + cursor], selected[capacity],
                  DataCopyExtParams{1, (chosen - capacity) * 4, 0, 0, 0}); PipeBarrier<PIPE_ALL>();
      cursor += (chosen - capacity + 15) / 16 * 16;
    }
  }

  __aicore__ inline void ProcessSharded() {
    if ASCEND_IS_AIV {
      constexpr uint32_t quota = 1024, shardEdges = 32768;
      const uint32_t core = GetBlockIdx(), producers = GetBlockNum();
      const uint32_t parts = e / shardEdges, count = last - first;
      const uint32_t capacity = parts == 128 ? (walkSpan <= 4096 ? 32 : 48) : 64;
      const uint32_t padded = (count + 63) / 64 * 64;
      const uint32_t queuePitch = (walkSpan + parts * 16 + 15) / 16 * 16;
      const uint32_t queueCapacity = nodesPitch + 640 * (parts + 1);
      const uint64_t stateBase = 640, metaBase = stateBase + nodesPitch;
      const uint64_t forwardBase = metaBase + 80 * parts;
      const uint64_t replyBase = forwardBase + queueCapacity;
      uint32_t idBits = 0;
      while ((1U << idBits) <= walkSpan) ++idBits;
      auto arena = shardBuffer.Get<int32_t>();
      auto nodeDst = arena[24576];
      const uint32_t nodeAddress = static_cast<uint32_t>(reinterpret_cast<uint64_t>(nodeDst.GetPhyAddr()));
      auto pack = arena[35840].ReinterpretCast<uint32_t>();
      auto table = arena[39936];
      // TPipe allocation is identical on every cooperative core; its UB
      // address is local to each AIV. Replies target the common arena offset.
      GlobalTensor<uint64_t> replies;
      replies.SetGlobalBuffer((__gm__ uint64_t*)(graphFlags.GetPhyAddr() + replyBase), queueCapacity);
      const uint32_t fixedCount = 40 * parts * capacity;
      GlobalTensor<int32_t> fixedForward;
      GlobalTensor<uint64_t> fixedReplies;
      fixedForward.SetGlobalBuffer((__gm__ int32_t*)shardWorkspace, fixedCount);
      fixedReplies.SetGlobalBuffer((__gm__ uint64_t*)((__gm__ int32_t*)shardWorkspace + fixedCount), fixedCount);
      constexpr bool traceStages = false;  // Private diagnostic builds enable this.
      uint64_t elapsed[5] = {0, 0, 0, 0, 0};
      for (uint32_t step = 0; step < len; ++step) {
        uint64_t stamp = traceStages ? GetSystemCycle() : 0;
        auto nodes = arena;
        auto temp = arena[8192];
        auto payload = arena[16384];
        auto buckets = arena[24576].ReinterpretCast<float>();
        auto mask = arena[38912].ReinterpretCast<uint8_t>();
        auto meta = arena[39424];
        DataCopyExtParams stateCopy{1, count * 4, 0, 0, 0};
        DataCopyPad(nodes, step == 0 ? starts[first] : graphFlags[stateBase + first],
                    stateCopy, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        DataCopyPad(temp.ReinterpretCast<float>(), rnd[static_cast<uint64_t>(step) * s + first],
                    stateCopy, DataCopyPadExtParams<float>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        if (step == 0) {
          DataCopyPad(walkGm[first], nodes, stateCopy); PipeBarrier<PIPE_ALL>();
        }
        Muls(temp.ReinterpretCast<float>(), temp.ReinterpretCast<float>(), static_cast<float>(static_cast<int32_t>(regularDegree)), count);
        PipeBarrier<PIPE_V>();
        Cast(payload, temp.ReinterpretCast<float>(), RoundMode::CAST_FLOOR, count);
        Muls(nodes, nodes, static_cast<int32_t>(regularDegree), count); PipeBarrier<PIPE_V>();
        Mins(payload, payload, static_cast<int32_t>(regularDegree - 1), count); PipeBarrier<PIPE_V>();
        Add(nodes, nodes, payload, count); PipeBarrier<PIPE_ALL>();
        DataCopyPad(edgeGm[static_cast<uint64_t>(step) * edgesPitch + first], nodes, stateCopy);
        PipeBarrier<PIPE_ALL>();
        // One forward word holds local edge and producer-local walk ID.
        const bool radix = parts >= 32;
        const uint32_t packShift = radix ? 19 : 15;
        ShiftRight(temp.ReinterpretCast<uint32_t>(), nodes.ReinterpretCast<uint32_t>(), packShift, count);
        Duplicate(buckets, float(-1), padded); PipeBarrier<PIPE_V>();
        Cast(buckets, temp, RoundMode::CAST_NONE, count);
        Muls(temp, temp, static_cast<int32_t>(1U << packShift), count); PipeBarrier<PIPE_V>();
        Sub(payload, nodes, temp, count); PipeBarrier<PIPE_V>();
        Muls(payload, payload, static_cast<int32_t>(1U << idBits), count);
        CreateVecIndex(temp, int32_t(0), count); PipeBarrier<PIPE_V>();
        Add(payload, payload, temp, count); PipeBarrier<PIPE_ALL>();
        auto fixed = arena[8192];
        // The unused ID is larger than every real ID, including power-of-two
        // spans. Its replies sort after all valid walks without extra masks.
        Duplicate(fixed, static_cast<int32_t>((1U << idBits) - 1), parts * capacity); PipeBarrier<PIPE_V>();
        Duplicate(meta, int32_t(0), parts * 2); PipeBarrier<PIPE_V>();
        uint32_t cursor = 0;
        if (radix) {
          // Save all coarse masks before reusing their value buffer. The
          // second level scans only its own group, instead of every walk.
          const uint32_t groups = parts / 16;
          const uint32_t maskStride = (padded + 255) / 256 * 32;
          auto coarseMasks = arena[32768].ReinterpretCast<uint8_t>();
          auto chosenNodes = arena[34816];
          for (uint32_t group = 0; group < groups; ++group) {
            CompareScalar(coarseMasks[group * maskStride], buckets,
                          static_cast<float>(static_cast<int32_t>(group)), CMPMODE::EQ, padded);
          }
          PipeBarrier<PIPE_ALL>();
          for (uint32_t group = 0; group < groups; ++group) {
            uint64_t groupCount64 = 0;
            GatherMask(nodes, payload, coarseMasks[group * maskStride].ReinterpretCast<uint32_t>(),
                       true, count, GatherMaskParams{1, 1, 8, 0}, groupCount64); PipeBarrier<PIPE_ALL>();
            const uint32_t groupCount = static_cast<uint32_t>(groupCount64);
            if (!groupCount) continue;
            if (groupCount <= 4096) {
              const uint32_t groupPadded = (groupCount + 63) / 64 * 64;
              ShiftRight(chosenNodes.ReinterpretCast<uint32_t>(), nodes.ReinterpretCast<uint32_t>(), idBits + 15, groupCount);
              Duplicate(buckets, float(-1), groupPadded); PipeBarrier<PIPE_V>();
              Cast(buckets, chosenNodes, RoundMode::CAST_NONE, groupCount); PipeBarrier<PIPE_V>();
              for (uint32_t inner = 0; inner < 16; ++inner) {
                CompareScalar(mask, buckets, static_cast<float>(static_cast<int32_t>(inner)), CMPMODE::EQ, groupPadded);
                PipeBarrier<PIPE_V>();
                uint64_t chosen64 = 0;
                GatherMask(chosenNodes, nodes, mask.ReinterpretCast<uint32_t>(), true, groupCount,
                           GatherMaskParams{1, 1, 8, 0}, chosen64); PipeBarrier<PIPE_V>();
                const uint32_t chosen = static_cast<uint32_t>(chosen64);
                if (chosen) Adds(chosenNodes, chosenNodes,
                    static_cast<int32_t>(0U - ((inner * shardEdges) << idBits)), chosen);
                PipeBarrier<PIPE_ALL>();
                StoreShardBucket(chosenNodes, fixed, meta, chosen, group * 16 + inner,
                                 capacity, cursor, forwardBase, queuePitch);
              }
            } else {
              // A concentrated coarse group can exceed the small temporary
              // buffer. Scan the original full payload with its coarse mask;
              // the full-sized node buffer holds every selected item.
              ShiftRight(nodes.ReinterpretCast<uint32_t>(), payload.ReinterpretCast<uint32_t>(), idBits + 15, count);
              Duplicate(buckets, float(-1), padded); PipeBarrier<PIPE_V>();
              Cast(buckets, nodes, RoundMode::CAST_NONE, count); PipeBarrier<PIPE_V>();
              for (uint32_t inner = 0; inner < 16; ++inner) {
                CompareScalar(mask, buckets, static_cast<float>(static_cast<int32_t>(inner)), CMPMODE::EQ, padded);
                PipeBarrier<PIPE_V>();
                And(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(),
                    coarseMasks[group * maskStride].ReinterpretCast<uint16_t>(), maskStride / 2);
                PipeBarrier<PIPE_V>();
                uint64_t chosen64 = 0;
                GatherMask(nodes, payload, mask.ReinterpretCast<uint32_t>(), true, count,
                           GatherMaskParams{1, 1, 8, 0}, chosen64); PipeBarrier<PIPE_V>();
                const uint32_t chosen = static_cast<uint32_t>(chosen64);
                if (chosen) Adds(nodes, nodes,
                    static_cast<int32_t>(0U - ((inner * shardEdges) << idBits)), chosen);
                PipeBarrier<PIPE_ALL>();
                StoreShardBucket(nodes, fixed, meta, chosen, group * 16 + inner,
                                 capacity, cursor, forwardBase, queuePitch);
              }
            }
          }
        } else {
          for (uint32_t part = 0; part < parts; ++part) {
            CompareScalar(mask, buckets, static_cast<float>(static_cast<int32_t>(part)), CMPMODE::EQ, padded);
            PipeBarrier<PIPE_V>();
            uint64_t chosen = 0;
            GatherMask(nodes, payload, mask.ReinterpretCast<uint32_t>(), true, count,
                       GatherMaskParams{1, 1, 8, 0}, chosen); PipeBarrier<PIPE_ALL>();
            StoreShardBucket(nodes, fixed, meta, static_cast<uint32_t>(chosen), part,
                             capacity, cursor, forwardBase, queuePitch);
          }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(fixedForward[core * parts * capacity], fixed,
                    DataCopyExtParams{1, parts * capacity * 4, 0, 0, 0});
        DataCopyPad(graphFlags[metaBase + core * parts * 2], meta,
                    DataCopyExtParams{1, parts * 8, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        if (traceStages) { const uint64_t now = GetSystemCycle(); elapsed[0] += now - stamp; stamp = now; }
        SyncAll<true>();
        if (traceStages) { const uint64_t now = GetSystemCycle(); elapsed[1] += now - stamp; stamp = now; }
        // Producer scratch reuses this area. Rebuild the interleave indices
        // with Vector rather than issuing 2048 scalar UB stores per step.
        auto packScratch = arena[37888].ReinterpretCast<uint32_t>();
        CreateVecIndex(pack.ReinterpretCast<int32_t>(), int32_t(0), quota * 2); PipeBarrier<PIPE_V>();
        ShiftLeft(packScratch, pack, uint32_t(31), quota * 2); PipeBarrier<PIPE_V>();
        ShiftRight(packScratch, packScratch, uint32_t(31), quota * 2);
        ShiftRight(pack, pack, uint32_t(1), quota * 2); PipeBarrier<PIPE_V>();
        Muls(pack.ReinterpretCast<int32_t>(), pack.ReinterpretCast<int32_t>(), int32_t(4), quota * 2);
        Muls(packScratch.ReinterpretCast<int32_t>(), packScratch.ReinterpretCast<int32_t>(), int32_t(quota * 4), quota * 2); PipeBarrier<PIPE_V>();
        Add(pack.ReinterpretCast<int32_t>(), pack.ReinterpretCast<int32_t>(), packScratch.ReinterpretCast<int32_t>(), quota * 2); PipeBarrier<PIPE_ALL>();
        auto column = arena;
        auto requests = arena[32768];
        auto next = arena[33792];
        auto offsets = arena[34816].ReinterpretCast<uint32_t>();
        auto records = arena[37888].ReinterpretCast<uint64_t>();
        for (uint32_t part = core; part < parts; part += producers) {
          DataCopyPad(column, ci[static_cast<uint64_t>(part) * shardEdges],
                      DataCopyExtParams{1, shardEdges * 4, 0, 0, 0},
                      DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
          DataCopyPad(table, graphFlags[metaBase + part * 2],
                      DataCopyExtParams{static_cast<uint16_t>(producers), 8, (parts * 2 - 2) * 4, 0, 0},
                      DataCopyPadExtParams<int32_t>{false, 0, 0, 0}); PipeBarrier<PIPE_ALL>();
          // Fetch and return sixteen producer rows with one descriptor each.
          // Padded rows are initialized; only their true counts are restored.
          for (uint32_t producer = 0; producer < producers; producer += quota / capacity) {
            const uint32_t batch = producers - producer < quota / capacity ? producers - producer : quota / capacity;
            const uint32_t filled = batch * capacity;
            DataCopyPad(requests, fixedForward[(producer * parts + part) * capacity],
                        DataCopyExtParams{static_cast<uint16_t>(batch), capacity * 4, (parts - 1) * capacity * 4, 0, 0},
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0}); PipeBarrier<PIPE_ALL>();
            ShiftRight(offsets, requests.ReinterpretCast<uint32_t>(), idBits, filled); PipeBarrier<PIPE_V>();
            Muls(offsets.ReinterpretCast<int32_t>(), offsets.ReinterpretCast<int32_t>(), int32_t(4), filled); PipeBarrier<PIPE_V>();
            Gather(next, column, offsets, 0, filled); PipeBarrier<PIPE_V>();
            ShiftLeft(requests.ReinterpretCast<uint32_t>(), requests.ReinterpretCast<uint32_t>(), uint32_t(32 - idBits), filled); PipeBarrier<PIPE_V>();
            ShiftRight(requests.ReinterpretCast<uint32_t>(), requests.ReinterpretCast<uint32_t>(), uint32_t(32 - idBits), filled); PipeBarrier<PIPE_V>();
            Muls(requests, requests, int32_t(4), filled); PipeBarrier<PIPE_V>();
            Adds(requests, requests, static_cast<int32_t>(nodeAddress), filled); PipeBarrier<PIPE_V>();
            Cast(requests.ReinterpretCast<float>(), requests, RoundMode::CAST_NONE, filled); PipeBarrier<PIPE_V>();
            Muls(requests.ReinterpretCast<float>(), requests.ReinterpretCast<float>(), float(-1), filled); PipeBarrier<PIPE_V>();
            Gather(records.ReinterpretCast<int32_t>(), requests, pack, 0, filled * 2); PipeBarrier<PIPE_ALL>();
            DataCopyPad(fixedReplies[(producer * parts + part) * capacity], records,
                        DataCopyExtParams{static_cast<uint16_t>(batch), capacity * 8, 0, (parts - 1) * capacity * 8, 0});
            PipeBarrier<PIPE_ALL>();
          }
          uint32_t producer = 0, consumed = 0;
          while (producer < producers) {
            uint32_t filled = 0, pieces = 0;
            uint32_t pieceProducer[40], pieceGm[40], pieceUb[40], pieceCount[40];
            while (producer < producers && filled < quota) {
              const uint32_t selected = static_cast<uint32_t>(table.GetValue(producer * 8));
              const uint32_t available = selected > capacity ? selected - capacity : 0;
              if (consumed == available) { ++producer; consumed = 0; continue; }
              const uint32_t take = available - consumed < quota - filled ? available - consumed : quota - filled;
              const uint32_t gmOffset = producer * queuePitch +
                  static_cast<uint32_t>(table.GetValue(producer * 8 + 1)) + consumed;
              pieceProducer[pieces] = producer;
              pieceGm[pieces] = gmOffset;
              pieceUb[pieces] = filled;
              pieceCount[pieces] = take;
              ++pieces;
              DataCopyPad(requests[filled], graphFlags[forwardBase + gmOffset],
                          DataCopyExtParams{1, take * 4, 0, 0, 0},
                          DataCopyPadExtParams<int32_t>{true, 0, 0, 0});
              filled += (take + 7) / 8 * 8;
              consumed += take;
            }
            if (!filled) continue;
            PipeBarrier<PIPE_ALL>();
            ShiftRight(offsets, requests.ReinterpretCast<uint32_t>(), idBits, filled);
            PipeBarrier<PIPE_V>();
            Muls(offsets.ReinterpretCast<int32_t>(), offsets.ReinterpretCast<int32_t>(), int32_t(4), filled);
            PipeBarrier<PIPE_V>();
            Gather(next, column, offsets, 0, filled); PipeBarrier<PIPE_V>();
            ShiftLeft(requests.ReinterpretCast<uint32_t>(), requests.ReinterpretCast<uint32_t>(), uint32_t(32 - idBits), filled);
            PipeBarrier<PIPE_V>();
            ShiftRight(requests.ReinterpretCast<uint32_t>(), requests.ReinterpretCast<uint32_t>(), uint32_t(32 - idBits), filled);
            PipeBarrier<PIPE_V>();
            Muls(requests, requests, int32_t(4), filled); PipeBarrier<PIPE_V>();
            Adds(requests, requests, static_cast<int32_t>(nodeAddress), filled);
            PipeBarrier<PIPE_V>();
            Cast(requests.ReinterpretCast<float>(), requests, RoundMode::CAST_NONE, filled); PipeBarrier<PIPE_V>();
            Muls(requests.ReinterpretCast<float>(), requests.ReinterpretCast<float>(), float(-1), filled); PipeBarrier<PIPE_V>();
            // requests and next are adjacent: interleave address/node words.
            Gather(records.ReinterpretCast<int32_t>(), requests, pack, 0, filled * 2);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t piece = 0; piece < pieces; ++piece)
              DataCopyPad(replies[pieceGm[piece]], records[pieceUb[piece]],
                          DataCopyExtParams{1, pieceCount[piece] * 8, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
          }
        }
        if (traceStages) { const uint64_t now = GetSystemCycle(); elapsed[2] += now - stamp; stamp = now; }
        SyncAll<true>();
        if (traceStages) { const uint64_t now = GetSystemCycle(); elapsed[3] += now - stamp; stamp = now; }
        // Restore all valid replies, ignoring alignment padding only. There
        // is no bucket capacity limit even if every request targets one shard.
        DataCopyPad(meta, graphFlags[metaBase + core * parts * 2],
                    DataCopyExtParams{1, parts * 8, 0, 0, 0},
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        DataCopyPad(arena.ReinterpretCast<uint64_t>(), fixedReplies[core * parts * capacity],
                    DataCopyExtParams{1, parts * capacity * 8, 0, 0, 0},
                    DataCopyPadExtParams<uint64_t>{false, 0, 0, 0}); PipeBarrier<PIPE_ALL>();
        __ubuf__ uint64_t* returned = (__ubuf__ uint64_t*)arena.GetPhyAddr();
        const uint32_t totalReplies = parts * capacity + cursor;
        const uint32_t sortCount = (totalReplies + 31) / 32 * 32;
        if (sortCount <= 8160 && cursor <= 2016) {
          if (cursor) {
            DataCopyPad(arena.ReinterpretCast<uint64_t>()[parts * capacity], replies[core * queuePitch],
                        DataCopyExtParams{1, cursor * 8, 0, 0, 0},
                        DataCopyPadExtParams<uint64_t>{false, 0, 0, 0}); PipeBarrier<PIPE_ALL>();
          }
          auto sortIndices = arena[39936].ReinterpretCast<uint32_t>();
          CreateVecIndex(sortIndices.ReinterpretCast<int32_t>(), int32_t(0), quota); PipeBarrier<PIPE_V>();
          Muls(sortIndices.ReinterpretCast<int32_t>(), sortIndices.ReinterpretCast<int32_t>(), int32_t(8), quota); PipeBarrier<PIPE_V>();
          auto sorted = arena.ReinterpretCast<float>();
          auto scratch = arena[16384].ReinterpretCast<float>();
          const uint32_t fixedRows = parts * capacity;
          if (cursor) {
            auto keys = arena[28672].ReinterpretCast<float>();
            auto values = arena[30720].ReinterpretCast<uint32_t>();
            for (uint32_t base = 0; base < cursor; base += quota) {
              const uint32_t take = cursor - base < quota ? cursor - base : quota;
              Gather(keys[base], sorted, sortIndices, (fixedRows + base) * 8, take);
              Gather(values[base], arena.ReinterpretCast<uint32_t>(), sortIndices, (fixedRows + base) * 8 + 4, take);
            }
            PipeBarrier<PIPE_ALL>();
            for (uint32_t part = 0; part < parts; ++part) {
              const uint32_t selected = meta.GetValue(part * 2);
              if (selected > capacity) {
                const uint32_t valid = selected - capacity;
                const uint32_t base = meta.GetValue(part * 2 + 1);
                for (uint32_t i = valid; i < (valid + 15) / 16 * 16; ++i)
                  keys.SetValue(base + i, float(-1.e30));
              }
            }
            const uint32_t overflowCount = (cursor + 31) / 32 * 32;
            for (uint32_t i = cursor; i < overflowCount; ++i) keys.SetValue(i, float(-1.e30));
            PipeBarrier<PIPE_ALL>();
            auto overflow = arena[12288].ReinterpretCast<float>();
            auto overflowWork = arena[32768].ReinterpretCast<float>();
            keys.SetSize(overflowCount); values.SetSize(overflowCount);
            overflow.SetSize(overflowCount * 2); overflowWork.SetSize(overflowCount * 2);
            Sort<float, true>(overflow, keys, values, overflowWork, overflowCount / 32);
            PipeBarrier<PIPE_ALL>();
          }
          MergeFixedReplies(sorted, scratch, fixedRows, capacity);
          if (cursor) {
            const uint32_t overflowCount = (cursor + 31) / 32 * 32;
            const uint16_t sizes[4] = {static_cast<uint16_t>(fixedRows), static_cast<uint16_t>(overflowCount), 0, 0};
            uint32_t consumed[4];
            auto overflow = arena[12288].ReinterpretCast<float>();
            MrgSort(scratch, MrgSortSrcList<float>{sorted, overflow, sorted, sorted},
                    sizes, consumed, uint16_t(3), 1);
            PipeBarrier<PIPE_V>();
            DataCopy(sorted, scratch, (fixedRows + overflowCount) * 2);
          }
          PipeBarrier<PIPE_ALL>();
          for (uint32_t base = 0; base < count; base += quota) {
            const uint32_t take = count - base < quota ? count - base : quota;
            Gather(nodeDst[base], sorted.ReinterpretCast<int32_t>(), sortIndices, base * 8 + 4, take);
          }
        } else {
          // Large or concentrated queues retain the complete scalar restore.
          for (uint32_t part = 0; part < parts; ++part) {
            const uint32_t selected = meta.GetValue(part * 2);
            RestoreRecords(returned + part * capacity, selected < capacity ? selected : capacity);
          }
          PipeBarrier<PIPE_ALL>();
          if (cursor) {
            DataCopyPad(arena.ReinterpretCast<uint64_t>(), replies[core * queuePitch],
                        DataCopyExtParams{1, cursor * 8, 0, 0, 0},
                        DataCopyPadExtParams<uint64_t>{false, 0, 0, 0}); PipeBarrier<PIPE_ALL>();
            for (uint32_t part = 0; part < parts; ++part) {
              const uint32_t selected = meta.GetValue(part * 2);
              if (selected > capacity)
                RestoreRecords(returned + meta.GetValue(part * 2 + 1), selected - capacity);
            }
          }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(graphFlags[stateBase + first], nodeDst, stateCopy);
        DataCopyPad(walkGm[static_cast<uint64_t>(step + 1) * nodesPitch + first], nodeDst, stateCopy);
        PipeBarrier<PIPE_ALL>();
        if (traceStages) elapsed[4] += static_cast<uint64_t>(GetSystemCycle()) - stamp;
      }
      if (traceStages) {
        GlobalTensor<uint64_t> trace;
        trace.SetGlobalBuffer((__gm__ uint64_t*)graphFlags.GetPhyAddr(), 320);
        for (uint32_t stage = 0; stage < 5; ++stage) trace.SetValue(core * 8 + stage + 1, elapsed[stage]);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE>(trace);
      }
    }
  }

  __aicore__ inline void Process() {
    if (phase == 4) {
      if constexpr (sizeof(IndexT) == 4) ProcessRandom();
      return;
    }
    if (phase == 3) {
      // Private bandwidth probe: no public dispatch selects this phase.
      auto tile = streamBuffer.Get<IndexT>();
      for (uint32_t repeat = 0; repeat < len; ++repeat) {
        for (uint32_t offset = 0; offset < e; offset += 16384) {
          const uint32_t count = e - offset < 16384 ? e - offset : 16384;
          DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(IndexT)), 0, 0, 0};
          DataCopyPad(tile, ci[offset], cp, DataCopyPadExtParams<IndexT>{false, 0, 0, 0});
          PipeBarrier<PIPE_ALL>();
        }
      }
      graphFlags.SetValue(GetBlockIdx() * 16, static_cast<int32_t>(tile.GetValue(0)));
      DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(graphFlags);
      return;
    }
    if (phase == 0) {
      // Check every row pointer before using arithmetic addressing. Shapes
      // alone cannot select this specialization, even when E is divisible by N.
      const uint64_t begin = (static_cast<uint64_t>(n) + 1) * GetBlockIdx() / checkBlocks;
      const uint64_t end = (static_cast<uint64_t>(n) + 1) * (GetBlockIdx() + 1) / checkBlocks;
      bool valid = true;
      const uint32_t degree = n ? e / n : 0;
      auto rows = checkBuffer.Get<IndexT>();
      for (uint64_t tile = begin; tile < end && valid; tile += 512) {
        const uint32_t count = end - tile < 512 ? end - tile : 512;
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(IndexT)), 0, 0, 0};
        DataCopyPad(rows, rp[tile], cp, DataCopyPadExtParams<IndexT>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        if constexpr (sizeof(IndexT) == 4) {
          // Every expected integer below 2^24 is exactly representable. An
          // equality check via FP32 reduction therefore proves all rowptr
          // values, including the final pointer; wider ranges stay scalar.
          if (e < 16777216) {
            auto values = boundsFloatBuffer.Get<float>();
            auto expected = boundsWorkBuffer.Get<float>();
            auto result = boundsResultBuffer.Get<float>();
            Cast(values, rows, RoundMode::CAST_NONE, count);
            CreateVecIndex(expected, static_cast<float>(static_cast<int32_t>(tile)), count);
            PipeBarrier<PIPE_V>();
            Muls(expected, expected, static_cast<float>(static_cast<int32_t>(degree)), count); PipeBarrier<PIPE_V>();
            Sub(values, values, expected, count); PipeBarrier<PIPE_V>();
            Abs(values, values, count); PipeBarrier<PIPE_V>();
            ReduceMax(result, values, expected, count, false); PipeBarrier<PIPE_ALL>();
            valid = result.GetValue(0) == 0.0f;
          } else {
            for (uint32_t i = 0; i < count; ++i)
              if (rows.GetValue(i) != static_cast<int64_t>(tile + i) * degree) { valid = false; break; }
          }
        } else {
          for (uint32_t i = 0; i < count; ++i)
            if (rows.GetValue(i) != static_cast<int64_t>(tile + i) * degree) { valid = false; break; }
        }
      }
      bool boundsValid = false;
      if constexpr (sizeof(IndexT) == 4) {
        if (valid && uniform && n <= 16777216)
          boundsValid = CheckIndexBounds(ci, e) && CheckIndexBounds(starts, s);
      }
      graphFlags.SetValue(GetBlockIdx() * 16, valid ? 1 : 0);
      graphFlags.SetValue(GetBlockIdx() * 16 + 1, boundsValid ? 1 : 0);
      DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(graphFlags);
      return;
    }
    if (phase == 2) {
      degreeOne = true;
      trustedGraph = true;
      for (uint32_t i = 0; i < checkBlocks; ++i) {
        if (graphFlags.GetValue(i * 16) != 1) degreeOne = false;
        if (graphFlags.GetValue(i * 16 + 1) != 1) trustedGraph = false;
      }
      if (degreeOne) regularDegree = n ? e / n : 0;
      degreeOne = regularDegree == 1;
    }
    if (outputMode) {
      if constexpr (sizeof(IndexT) == 4) {
        if (trustedGraph && (regularDegree == 4 || regularDegree == 8) && walkSpan <= 8192) {
          ProcessSharded();
          return;
        }
      }
      ProcessWalks<true, false>();
      return;
    }
    if (localGraph) {
      localRows = rowBuffer.Get<IndexT>();
      localCols = colBuffer.Get<IndexT>();
      DataCopyExtParams rowCopy{1, static_cast<uint32_t>((n + 1) * sizeof(IndexT)), 0, 0, 0};
      DataCopyPad(localRows, rp, rowCopy, DataCopyPadExtParams<IndexT>{false, 0, 0, 0});
      if (e) {
        DataCopyExtParams colCopy{1, static_cast<uint32_t>(e * sizeof(IndexT)), 0, 0, 0};
        DataCopyPad(localCols, ci, colCopy, DataCopyPadExtParams<IndexT>{false, 0, 0, 0});
      }
      PipeBarrier<PIPE_ALL>();
      DispatchWalks<true>();
    } else {
      DispatchWalks<false>();
    }
  }

private:
  __aicore__ inline bool CheckIndexBounds(GlobalTensor<IndexT>& source, uint32_t size) {
    auto indices = checkBuffer.Get<IndexT>();
    auto values = boundsFloatBuffer.Get<float>();
    auto work = boundsWorkBuffer.Get<float>();
    auto result = boundsResultBuffer.Get<float>();
    const uint64_t begin = static_cast<uint64_t>(size) * GetBlockIdx() / checkBlocks;
    const uint64_t end = static_cast<uint64_t>(size) * (GetBlockIdx() + 1) / checkBlocks;
    for (uint64_t off = begin; off < end; off += 512) {
      const uint32_t count = end - off < 512 ? end - off : 512;
      DataCopyExtParams cp{1, count * 4, 0, 0, 0};
      DataCopyPad(indices, source[off], cp, DataCopyPadExtParams<IndexT>{false, 0, 0, 0});
      PipeBarrier<PIPE_ALL>();
      Cast(values, indices, RoundMode::CAST_NONE, count); PipeBarrier<PIPE_V>();
      ReduceMin(result, values, work, count, false); PipeBarrier<PIPE_ALL>();
      const float minimum = result.GetValue(0);
      ReduceMax(result, values, work, count, false); PipeBarrier<PIPE_ALL>();
      const float maximum = result.GetValue(0);
      if (!(minimum >= 0 && maximum < static_cast<float>(static_cast<int32_t>(n)))) return false;
    }
    return true;
  }

  TPipe pipe;
  static constexpr uint32_t kRngLanes = 2048;
  TBuf<QuePosition::VECCALC> rngStateBuffer;
  TBuf<QuePosition::VECCALC> rngIndexBuffer;
  TQue<QuePosition::VECOUT, 2> rngOutQueue;
  static constexpr uint32_t kWalkBatch = 8;
  static constexpr uint32_t kStepTile = 64;
  static constexpr uint32_t kNodeTile = 80;
  static constexpr uint32_t kHybridBatch = 256;
  static constexpr uint32_t kHybridSteps = 16;
  static constexpr uint32_t kHybridNodePitch = 24;
  TBuf<QuePosition::VECCALC> rowBuffer, colBuffer;
  TBuf<QuePosition::VECCALC> randomBuffer, nodeBuffer, edgeBuffer;
  TBuf<QuePosition::VECCALC> checkBuffer;
  TBuf<QuePosition::VECCALC> shardBuffer;
  TBuf<QuePosition::VECCALC> offsetBuffer;
  TBuf<QuePosition::VECCALC> streamBuffer;
  TBuf<QuePosition::VECCALC> hybridRandomBuffer, hybridOffsetBuffer, hybridNodesBuffer;
  TBuf<QuePosition::VECCALC> hybridEdgesBuffer, hybridOutputBuffer, hybridIndicesBuffer, hybridStateBuffer;
  TBuf<QuePosition::VECCALC> boundsFloatBuffer, boundsWorkBuffer, boundsResultBuffer;
  LocalTensor<IndexT> localRows, localCols;
  GlobalTensor<IndexT> rp, ci, starts, walkGm, edgeGm;
  GlobalTensor<float> rnd;
  GlobalTensor<uint64_t> ciCache;
  GlobalTensor<int32_t> graphFlags;
  uint32_t n, e, s, len, nodesPitch, edgesPitch, first, last;
  float returnWeight, farWeight;
  uint32_t phase, checkBlocks, regularDegree;
  uint32_t outputMode, walkSpan;
  GM_ADDR shardWorkspace;
  bool uniform, localGraph, degreeOne, trustedGraph;
};

extern "C" __global__ __aicore__ void random_walk(
    GM_ADDR rowptr, GM_ADDR col, GM_ADDR start, GM_ADDR random,
    GM_ADDR walks, GM_ADDR edgeIds, GM_ADDR flags, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(data, tiling);
  KernelRandomWalk<DTYPE_ROWPTR> op;
  op.Init(rowptr, col, start, random, walks, edgeIds, flags,
          data.numNodes, data.numEdges, data.numWalks, data.walkLength,
          data.nodesPitch, data.edgesPitch, data.walksPerCore,
          data.phase, data.checkBlocks, data.p, data.q, data.outputMode,
          data.outputMode && data.phase == 2 ? GetUserWorkspace(workspace) : nullptr);
  op.Process();
}
