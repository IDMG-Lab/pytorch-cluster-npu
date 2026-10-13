
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * radius 算子内核（SoA + hash-grid 空间索引；保留向量化暴力回退）
 *
 * 输入：
 *   x: [F,N] SoA, y: [F,M] SoA
 *   ptr_x/ptr_y: int32 CSR（可选）
 *   sorted_x: [F,N] 各段内按 cell 排序后的 x（段间按 ptr_x 连续拼接）
 *   order:    [N]   排序位置 -> 全局原 x 下标
 *   cell_start: [Σ(G_q+1)] 各段 cell 起止（段间拼接）
 *   cell_start_off: [B+1]  各段在 cell_start 中的偏移
 *   grid_min: [B*F] float  各段各维网格原点
 *   grid_g:   [B*F] int32  各段各维 cell 数
 *   use_grid: [1] int32    1=hash-grid，0=暴力
 * 输出：
 *   out: [3, maxSize]，row0=x 下标，row1=y 下标，-1 填充
 *
 * use_grid=1 时：cell 边长 = r，对每个 y 只扫描同 cell 及相邻 cell（3^F 个）内的点是
 * 连续存放的 sorted_x，可向量化分块计算，复杂度从 O(N·M) 降到 O(N+M+Σ候选)。
 * use_grid=0（F>4 或 G 过大等）时回退到 SoA 向量化暴力 + 包围盒/块级剪枝。
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t TILE = 64;
constexpr int32_t GROUP_CANDIDATES = 256;
constexpr int32_t REDUCE_WORK = 256;
constexpr int32_t BBOX_CAP = 8192;

template <typename TYPE_X>
class KernelRadius
{
public:
  __aicore__ inline KernelRadius() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR ptr_x, GM_ADDR ptr_y,
                              GM_ADDR sorted_x, GM_ADDR order, GM_ADDR cell_start,
                              GM_ADDR cell_start_off, GM_ADDR grid_min, GM_ADDR grid_g,
                              GM_ADDR use_grid, GM_ADDR compact_state, GM_ADDR packed, GM_ADDR out,
                              float r, uint32_t ignoreSameIndex, uint32_t maxNumNeighbors,
                              uint32_t xSize, uint32_t ySize, uint32_t itemLength,
                              uint32_t ptrXLen, uint32_t ptrYLen, uint32_t phase,
                              uint32_t gridCells, uint32_t nativeCores, uint32_t queryMode)
  {
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    this->coreNum = GetBlockNum();
    this->coreId = GetBlockIdx();
    this->r = r;
    this->r2 = r * r;
    this->ignoreSameIndex = ignoreSameIndex;
    this->maxNumNeighbors = maxNumNeighbors;
    this->xSize = xSize;
    this->ySize = ySize;
    this->itemLength = itemLength;
    this->maxSize = maxNumNeighbors * ySize;
    this->ptrXLen = ptrXLen;
    this->ptrYLen = ptrYLen;
    this->phase = phase;
    this->gridCells = gridCells;
    this->nativeCores = nativeCores;
    this->gridStride = (gridCells + 8) / 8 * 8;
    this->histogramStride = this->gridStride;
    this->queryMode = queryMode;
    this->numTiles = (xSize + TILE - 1) / TILE;
    compactStateGm.SetGlobalBuffer((__gm__ int32_t*)compact_state, 656);
    packedGm.SetGlobalBuffer((__gm__ int32_t*)packed, static_cast<uint64_t>(maxSize) * 4);
    outGm.SetGlobalBuffer((__gm__ int32_t*)out, static_cast<uint64_t>(maxSize) * 2);
    if (phase == 7 || phase == 8) {
      xGm.SetGlobalBuffer((__gm__ TYPE_X*)x, static_cast<uint64_t>(itemLength) * xSize);
      boundsGm.SetGlobalBuffer((__gm__ float*)compact_state, 656);
      gridMinGm.SetGlobalBuffer((__gm__ float*)grid_min, itemLength);
      gridGGm.SetGlobalBuffer((__gm__ int32_t*)grid_g, itemLength);
      useGridGm.SetGlobalBuffer((__gm__ int32_t*)use_grid, 1);
      cellStartOffGm.SetGlobalBuffer((__gm__ int32_t*)cell_start_off, 2);
      if (phase == 7) {
        pipe.InitBuffer(xFpBuf, NATIVE_TILE * sizeof(float));
        pipe.InitBuffer(reduceWorkBuf, REDUCE_WORK * sizeof(float));
        pipe.InitBuffer(reduceOutBuf, 8 * sizeof(float));
      }
      return;
    }
    if (phase == 6) {
      xGm.SetGlobalBuffer((__gm__ TYPE_X*)x, static_cast<uint64_t>(itemLength) * xSize);
      yGm.SetGlobalBuffer((__gm__ TYPE_X*)y, static_cast<uint64_t>(itemLength) * ySize);
      pipe.InitBuffer(equalityBuf, COMPACT_TILE * 2 * sizeof(float));
      pipe.InitBuffer(compactMaskBuf, COMPACT_TILE / 8);
      return;
    }
    if (phase == 1 || phase == 2) {
      pipe.InitBuffer(compactXBuf, (COMPACT_TILE + 16) * sizeof(int32_t));
      pipe.InitBuffer(compactYBuf, (COMPACT_TILE + 16) * sizeof(int32_t));
      pipe.InitBuffer(compactSrcBuf, COMPACT_TILE * sizeof(int32_t));
      pipe.InitBuffer(compactFloatBuf, COMPACT_TILE * sizeof(float));
      pipe.InitBuffer(compactMaskBuf, COMPACT_TILE / 8);
      pipe.InitBuffer(expandOffsetsBuf, COMPACT_TILE * 2 * sizeof(int32_t));
      pipe.InitBuffer(expandedBuf, COMPACT_TILE * 2 * sizeof(int32_t));
      pipe.InitBuffer(expandWorkBuf, COMPACT_TILE * 2 * sizeof(int32_t));
      return;
    }

    xGm.SetGlobalBuffer((__gm__ TYPE_X *)x, (int64_t)itemLength * xSize);
    yGm.SetGlobalBuffer((__gm__ TYPE_X *)y, (int64_t)itemLength * ySize);
    outGm.SetGlobalBuffer((__gm__ int32_t *)out, (int64_t)maxSize * 2 + 1);
    if (ptrXLen != 0)
    {
      ptrXGm.SetGlobalBuffer((__gm__ int32_t *)ptr_x, ptrXLen);
    }
    if (ptrYLen != 0)
    {
      ptrYGm.SetGlobalBuffer((__gm__ int32_t *)ptr_y, ptrYLen);
    }
    sortedXGm.SetGlobalBuffer((__gm__ TYPE_X *)sorted_x, (int64_t)itemLength * xSize);
    orderGm.SetGlobalBuffer((__gm__ int32_t *)order, xSize > 0 ? xSize : 1);
    gridMinGm.SetGlobalBuffer((__gm__ float *)grid_min, 1);
    gridGGm.SetGlobalBuffer((__gm__ int32_t *)grid_g, 1);
    useGridGm.SetGlobalBuffer((__gm__ int32_t *)use_grid, 1);
    const int32_t gridMode = useGridGm.GetValue(0);
    useGrid = (gridMode != 0) && (itemLength <= 4) && (r > 0.0f);
    groupedGrid = gridMode == 2 || gridMode == 3;
    if (gridMode == 3 && phase == 0) {
      for (uint32_t core = 0; core < coreNum; ++core) {
        if (compactStateGm.GetValue(core * 16) != 1) { groupedGrid = false; break; }
      }
    }
    groupedGrid = groupedGrid && maxNumNeighbors <= TILE;

    // cell_start / cell_start_off 仅在 use_grid 时使用，这里延迟绑定（大小未知，用足够大的逻辑长度）
    if (useGrid)
    {
      cellStartGm.SetGlobalBuffer((__gm__ int32_t *)cell_start, (int64_t)1 << 40);
      cellStartOffGm.SetGlobalBuffer((__gm__ int32_t *)cell_start_off, (int64_t)1 << 40);
    }
    if (queryMode && useGrid) {
      this->gridCells = 1;
      for (uint32_t d = 0; d < itemLength; ++d) this->gridCells *= gridGGm.GetValue(d);
      this->gridStride = (this->gridCells + 8) / 8 * 8;
    }
    if (phase >= 3 && phase <= 5) {
      if (!useGrid) return;
      pipe.InitBuffer(nativeTableBuf, 4 * gridStride * sizeof(int32_t));
      pipe.InitBuffer(nativeCellsBuf, 3 * NATIVE_TILE * sizeof(int32_t));
      pipe.InitBuffer(nativeCoordsBuf, itemLength * NATIVE_TILE * sizeof(float));
      const uint32_t capacity = 96 * 1024 / (gridStride * (itemLength + 1) * sizeof(int32_t));
      bucketCapacity = 1;
      while (bucketCapacity < 16 && bucketCapacity * 2 <= capacity) bucketCapacity *= 2;
      if (phase == 5)
        pipe.InitBuffer(nativeBucketsBuf, gridStride * bucketCapacity * (itemLength + 1) * sizeof(int32_t));
      return;
    }

    pipe.InitBuffer(xTileBuf, TILE * sizeof(TYPE_X));
    pipe.InitBuffer(xFpBuf, (groupedGrid ? GROUP_CANDIDATES : TILE) * sizeof(float));
    pipe.InitBuffer(xAllBuf, 4 * TILE * sizeof(float));
    pipe.InitBuffer(sumBuf, (groupedGrid ? GROUP_CANDIDATES : TILE) * sizeof(float));
    pipe.InitBuffer(reduceWorkBuf, REDUCE_WORK * sizeof(float));
    pipe.InitBuffer(reduceOutBuf, 8 * sizeof(float));
    pipe.InitBuffer(bboxMinBuf, BBOX_CAP * sizeof(float));
    pipe.InitBuffer(bboxMaxBuf, BBOX_CAP * sizeof(float));
    pipe.InitBuffer(yValBuf, AlignUp((int)itemLength) * sizeof(float));
    pipe.InitBuffer(cellDimsBuf, 8 * sizeof(int32_t));
    pipe.InitBuffer(resXBuf, AlignUp((int)maxNumNeighbors) * sizeof(int32_t));
    pipe.InitBuffer(resYBuf, AlignUp((int)maxNumNeighbors) * sizeof(int32_t));
    pipe.InitBuffer(orderTileBuf, TILE * sizeof(int32_t));
    if (groupedGrid) {
      pipe.InitBuffer(groupRawXBuf, 4 * GROUP_CANDIDATES * sizeof(float));
      pipe.InitBuffer(groupYBuf, 4 * TILE * sizeof(float));
      pipe.InitBuffer(groupIdBuf, TILE * sizeof(int32_t));
      pipe.InitBuffer(groupCandidateIdBuf, GROUP_CANDIDATES * sizeof(int32_t));
      pipe.InitBuffer(groupOutBuf, GROUP_CANDIDATES * sizeof(int32_t));
      pipe.InitBuffer(compactMaskBuf, 32);
    }
  }

  __aicore__ inline int AlignUp(int x) { return (x + 31) / 32 * 32; }

  __aicore__ inline void Process()
  {
    if (phase == 6) { ProcessEquality(); return; }
    if (phase == 7 || phase == 8) {
      if constexpr (std::is_same_v<TYPE_X, float>) ProcessGridMetadata();
      return;
    }
    if (phase >= 3 && phase <= 5) {
      if constexpr (std::is_same_v<TYPE_X, float>) { if (useGrid) ProcessNativeGrid(); }
      return;
    }
    if (phase) { ProcessCompact(); return; }
    if (useGrid)
    {
      if constexpr (std::is_same_v<TYPE_X, float>) {
        if (groupedGrid) { ProcessGridGrouped(); return; }
      }
      ProcessGrid();
    }
    else
    {
      this->useBbox = ((int64_t)itemLength * this->numTiles <= BBOX_CAP) && (numTiles > 0);
      PrecomputeBbox();
      ProcessBrute();
    }
  }

private:
  __aicore__ inline void ProcessGridMetadata() {
    if (phase == 7) {
      auto values = xFpBuf.Get<float>();
      auto work = reduceWorkBuf.Get<float>();
      auto result = reduceOutBuf.Get<float>();
      const uint32_t span = (xSize + nativeCores - 1) / nativeCores;
      const uint32_t begin = coreId * span;
      const uint32_t end = begin + span < xSize ? begin + span : xSize;
      for (uint32_t d = 0; d < itemLength; ++d) {
        float minimum = 3.402823e38f, maximum = -3.402823e38f;
        for (uint32_t off = begin; off < end; off += NATIVE_TILE) {
          const uint32_t length = end - off < NATIVE_TILE ? end - off : NATIVE_TILE;
          DataCopyExtParams cp{1, length * 4, 0, 0, 0};
          DataCopyPad(values, xGm[static_cast<uint64_t>(d) * xSize + off], cp,
                      DataCopyPadExtParams<float>{false, 0, 0, 0}); PipeBarrier<PIPE_ALL>();
          ReduceMin(result, values, work, length, false); PipeBarrier<PIPE_ALL>();
          const float lo = result.GetValue(0);
          ReduceMax(result, values, work, length, false); PipeBarrier<PIPE_ALL>();
          const float hi = result.GetValue(0);
          minimum = lo < minimum ? lo : minimum;
          maximum = hi > maximum ? hi : maximum;
        }
        boundsGm.SetValue(coreId * 16 + d, minimum);
        boundsGm.SetValue(coreId * 16 + 4 + d, maximum);
      }
      DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(boundsGm);
      return;
    }
    uint32_t cells = 1;
    bool valid = true;
    for (uint32_t d = 0; d < itemLength; ++d) {
      float minimum = 3.402823e38f, maximum = -3.402823e38f;
      for (uint32_t core = 0; core < nativeCores; ++core) {
        const float lo = boundsGm.GetValue(core * 16 + d);
        const float hi = boundsGm.GetValue(core * 16 + 4 + d);
        minimum = lo < minimum ? lo : minimum;
        maximum = hi > maximum ? hi : maximum;
      }
      const float extent = (maximum - minimum) / r;
      int32_t dimension = 1;
      if (!(extent >= 0.0f && extent < 4096.0f)) valid = false;
      else dimension = static_cast<int32_t>(extent) + 1;
      if (cells > 4096U / static_cast<uint32_t>(dimension)) valid = false;
      else cells *= dimension;
      gridMinGm.SetValue(d, minimum);
      gridGGm.SetValue(d, dimension);
    }
    int32_t mode = 1;
    if (queryMode >= 2 && maxNumNeighbors <= TILE &&
        (static_cast<uint64_t>(xSize) >= 2ULL * maxNumNeighbors * cells ||
         (xSize >= 10240 && static_cast<uint64_t>(xSize) >= static_cast<uint64_t>(maxNumNeighbors) * cells / 2)))
      mode = queryMode;
    useGridGm.SetValue(0, valid ? mode : 0);
    cellStartOffGm.SetValue(0, 0);
    cellStartOffGm.SetValue(1, cells + 1);
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(gridMinGm);
  }

  __aicore__ inline void FlushBucket(uint32_t cell, uint32_t count) {
    auto tables = nativeTableBuf.Get<int32_t>();
    auto buckets = nativeBucketsBuf.Get<int32_t>();
    auto tmp = nativeCellsBuf.Get<int32_t>()[NATIVE_TILE];
    auto offsets = nativeCellsBuf.Get<uint32_t>()[2 * NATIVE_TILE];
    const uint32_t destination = tables.GetValue(cell);
    DataCopyExtParams cp{1, count * 4, 0, 0, 0};
    for (uint32_t d = 0; d <= itemLength; ++d) {
      Gather(tmp, buckets, offsets, (d * gridStride + cell) * bucketCapacity * 4, count);
      PipeBarrier<PIPE_ALL>();
      if (d == itemLength) DataCopyPad(orderGm[destination], tmp, cp);
      else DataCopyPad(sortedXGm[static_cast<uint64_t>(d) * xSize + destination], tmp.ReinterpretCast<float>(), cp);
      PipeBarrier<PIPE_ALL>();
    }
    tables.SetValue(cell, destination + count);
    tables.SetValue(gridStride + cell, 0);
  }

  // Private per-core histograms provide non-overlapping scatter ranges. No
  // atomics, fixed-size bucket truncation, or global stable sort is needed.
  __aicore__ inline void ProcessNativeGrid() {
    auto table = nativeTableBuf.Get<int32_t>();
    auto counts = table[gridStride];
    auto prefix = table[2 * gridStride];
    auto cells = nativeCellsBuf.Get<int32_t>();
    auto temporary = cells[NATIVE_TILE];
    auto coords = nativeCoordsBuf.Get<float>();
    const uint64_t histogramBase = 656 + static_cast<uint64_t>(xSize);
    if (phase == 4) {
      Duplicate(table, int32_t(0), gridStride); PipeBarrier<PIPE_ALL>();
      DataCopyExtParams cp{1, gridStride * 4, 0, 0, 0};
      for (uint32_t core = 0; core < nativeCores; ++core) {
        DataCopyPad(counts, compactStateGm[histogramBase + core * histogramStride], cp,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0}); PipeBarrier<PIPE_ALL>();
        Add(table, table, counts, gridStride); PipeBarrier<PIPE_ALL>();
      }
      int32_t sum = 0;
      for (uint32_t cell = 0; cell < gridCells; ++cell) {
        prefix.SetValue(cell, sum);
        sum += table.GetValue(cell);
      }
      prefix.SetValue(gridCells, sum); PipeBarrier<PIPE_ALL>();
      DataCopyExtParams startsCopy{1, (gridCells + 1) * 4, 0, 0, 0};
      DataCopyPad(cellStartGm, prefix, startsCopy); PipeBarrier<PIPE_ALL>();
      for (uint32_t core = 0; core < nativeCores; ++core) {
        DataCopyPad(counts, compactStateGm[histogramBase + core * histogramStride], cp,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0}); PipeBarrier<PIPE_ALL>();
        DataCopyPad(compactStateGm[histogramBase + core * histogramStride], prefix, cp);
        PipeBarrier<PIPE_ALL>();
        Add(prefix, prefix, counts, gridCells); PipeBarrier<PIPE_ALL>();
      }
      return;
    }
    const uint32_t span = ((xSize + nativeCores - 1) / nativeCores + 15) / 16 * 16;
    const uint32_t begin = coreId * span < xSize ? coreId * span : xSize;
    const uint32_t end = begin + span < xSize ? begin + span : xSize;
    if (phase == 3) {
      Duplicate(table, int32_t(0), gridStride); PipeBarrier<PIPE_ALL>();
      for (uint32_t off = begin; off < end; off += NATIVE_TILE) {
        const uint32_t length = end - off < NATIVE_TILE ? end - off : NATIVE_TILE;
        DataCopyExtParams cp{1, length * 4, 0, 0, 0};
        for (uint32_t d = 0; d < itemLength; ++d)
          DataCopyPad(coords[d * NATIVE_TILE], xGm[static_cast<uint64_t>(d) * xSize + off], cp,
                      DataCopyPadExtParams<float>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Duplicate(cells, int32_t(0), length); PipeBarrier<PIPE_V>();
        int32_t stride = 1;
        for (uint32_t d = 0; d < itemLength; ++d) {
          const int32_t dimension = gridGGm.GetValue(d);
          Adds(coords[d * NATIVE_TILE], coords[d * NATIVE_TILE], -gridMinGm.GetValue(d), length);
          Muls(coords[d * NATIVE_TILE], coords[d * NATIVE_TILE], 1.0f / r, length);
          Cast(temporary, coords[d * NATIVE_TILE], RoundMode::CAST_FLOOR, length); PipeBarrier<PIPE_V>();
          Maxs(temporary, temporary, int32_t(0), length);
          Mins(temporary, temporary, dimension - 1, length);
          Muls(temporary, temporary, stride, length);
          Add(cells, cells, temporary, length); PipeBarrier<PIPE_ALL>();
          stride *= dimension;
        }
        for (uint32_t u = 0; u < length; ++u) {
          const uint32_t cell = cells.GetValue(u);
          table.SetValue(cell, table.GetValue(cell) + 1);
        }
        DataCopyPad(compactStateGm[656 + off], cells, cp); PipeBarrier<PIPE_ALL>();
      }
      DataCopyExtParams cp{1, gridStride * 4, 0, 0, 0};
      DataCopyPad(compactStateGm[histogramBase + coreId * histogramStride], table, cp);
      PipeBarrier<PIPE_ALL>();
      return;
    }
    auto buckets = nativeBucketsBuf.Get<int32_t>();
    auto offsets = cells[2 * NATIVE_TILE].ReinterpretCast<uint32_t>();
    CreateVecIndex(offsets.ReinterpretCast<int32_t>(), int32_t(0), 32); PipeBarrier<PIPE_V>();
    Muls(offsets.ReinterpretCast<int32_t>(), offsets.ReinterpretCast<int32_t>(), int32_t(4), 32);
    Duplicate(counts, int32_t(0), gridStride); PipeBarrier<PIPE_ALL>();
    DataCopyExtParams cursorCopy{1, gridStride * 4, 0, 0, 0};
    DataCopyPad(table, compactStateGm[histogramBase + coreId * histogramStride], cursorCopy,
                DataCopyPadExtParams<int32_t>{false, 0, 0, 0}); PipeBarrier<PIPE_ALL>();
    for (uint32_t off = begin; off < end; off += NATIVE_TILE) {
      const uint32_t length = end - off < NATIVE_TILE ? end - off : NATIVE_TILE;
      DataCopyExtParams cp{1, length * 4, 0, 0, 0};
      DataCopyPad(cells, compactStateGm[656 + off], cp, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
      for (uint32_t d = 0; d < itemLength; ++d)
        DataCopyPad(coords[d * NATIVE_TILE], xGm[static_cast<uint64_t>(d) * xSize + off], cp,
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
      PipeBarrier<PIPE_ALL>();
      for (uint32_t u = 0; u < length; ++u) {
        const uint32_t cell = cells.GetValue(u);
        const uint32_t slot = counts.GetValue(cell);
        for (uint32_t d = 0; d < itemLength; ++d)
          buckets.ReinterpretCast<float>().SetValue((d * gridStride + cell) * bucketCapacity + slot,
                                                    coords.GetValue(d * NATIVE_TILE + u));
        buckets.SetValue((itemLength * gridStride + cell) * bucketCapacity + slot, off + u);
        counts.SetValue(cell, slot + 1);
        if (slot + 1 == bucketCapacity) { PipeBarrier<PIPE_ALL>(); FlushBucket(cell, slot + 1); }
      }
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t cell = 0; cell < gridCells; ++cell) {
      const uint32_t count = counts.GetValue(cell);
      if (count) FlushBucket(cell, count);
    }
  }

  __aicore__ inline void ProcessEquality() {
    bool equal = true;
    if constexpr (std::is_same_v<TYPE_X, float>) {
      auto a = equalityBuf.Get<float>();
      auto b = a[COMPACT_TILE];
      auto mask = compactMaskBuf.Get<uint8_t>();
      const uint64_t size = static_cast<uint64_t>(xSize) * itemLength;
      const uint64_t span = (size + coreNum - 1) / coreNum;
      const uint64_t begin = coreId * span;
      const uint64_t end = begin + span < size ? begin + span : size;
      for (uint64_t off = begin; off < end && equal; off += COMPACT_TILE) {
        const uint32_t length = end - off < COMPACT_TILE ? end - off : COMPACT_TILE;
        DataCopyExtParams cp{1, length * 4, 0, 0, 0};
        DataCopyPad(a, xGm[off], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(b, yGm[off], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Compare(mask, a, b, CMPMODE::EQ, length); PipeBarrier<PIPE_ALL>();
        auto words = mask.ReinterpretCast<uint32_t>();
        for (uint32_t i = 0; i < length / 32; ++i)
          if (words.GetValue(i) != 0xffffffffU) { equal = false; break; }
        if (length % 32) {
          const uint32_t expected = (1U << (length % 32)) - 1;
          if ((words.GetValue(length / 32) & expected) != expected) equal = false;
        }
      }
    } else equal = false;
    compactStateGm.SetValue(coreId * 16, equal ? 1 : 0);
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(compactStateGm);
  }

  // Reuse x's cell ordering only for a shared view or device-proven equal y.
  // Candidate DMA is amortized across every query in a cell; SIMD compression
  // replaces the scalar candidate scan. Sparse queries use the complete path.
  __aicore__ inline void ProcessGridGrouped() {
    auto raw = groupRawXBuf.Get<float>();
    auto queries = groupYBuf.Get<float>();
    auto ids = groupIdBuf.Get<int32_t>();
    auto candidateIds = groupCandidateIdBuf.Get<int32_t>();
    auto filtered = groupOutBuf.Get<int32_t>();
    auto mask = compactMaskBuf.Get<uint8_t>();
    auto distance = sumBuf.Get<float>();
    auto temporary = xFpBuf.Get<float>();
    auto resX = resXBuf.Get<int32_t>();
    auto resY = resYBuf.Get<int32_t>();
    float origin[4]; int32_t dims[4], strides[4];
    int32_t stride = 1;
    for (uint32_t d = 0; d < itemLength; ++d) {
      origin[d] = gridMinGm.GetValue(d);
      dims[d] = gridGGm.GetValue(d);
      strides[d] = stride;
      stride *= dims[d];
    }
    int32_t lastCell = -1;
    uint32_t candidateCount = 0;
    // Contiguous spatial ranges put sparse tails on a few cores. Interleave
    // small sorted-query chunks so the expensive complete-search fallback is
    // distributed over every core while preserving reuse within each chunk.
    constexpr uint32_t queryChunk = 8;
    for (uint32_t chunk = coreId * queryChunk; chunk < ySize; chunk += coreNum * queryChunk) {
    uint32_t position = chunk;
    const uint32_t end = chunk + queryChunk < ySize ? chunk + queryChunk : ySize;
    while (position < end) {
      int32_t cell = 0;
      if (gridCells) {
        // Native construction already produced each original point's cell ID.
        // Reuse it instead of scalar floating-point coordinate conversion.
        const uint32_t original = orderGm.GetValue(position);
        cell = compactStateGm.GetValue(656 + original);
      } else {
        for (uint32_t d = 0; d < itemLength; ++d) {
          int32_t coordinate = static_cast<int32_t>((sortedXGm.GetValue(static_cast<uint64_t>(d) * xSize + position) - origin[d]) / r);
          coordinate = coordinate < 0 ? 0 : (coordinate >= dims[d] ? dims[d] - 1 : coordinate);
          cell += coordinate * strides[d];
        }
      }
      const uint32_t firstCandidate = cellStartGm.GetValue(cell);
      const uint32_t cellEnd = cellStartGm.GetValue(cell + 1);
      if (cellEnd <= position || firstCandidate > position) {
        const int32_t original = orderGm.GetValue(position);
        ComputeNeighborsGrid(original, 0, 0, xSize, position);
        ++position; continue;
      }
      uint32_t batch = cellEnd - position;
      if (batch > TILE) batch = TILE;
      if (batch > end - position) batch = end - position;
      if (lastCell != cell) {
        candidateCount = cellEnd - firstCandidate < GROUP_CANDIDATES ? cellEnd - firstCandidate : GROUP_CANDIDATES;
        DataCopyExtParams cp{1, candidateCount * 4, 0, 0, 0};
        for (uint32_t d = 0; d < itemLength; ++d)
          DataCopyPad(raw[d * GROUP_CANDIDATES], sortedXGm[static_cast<uint64_t>(d) * xSize + firstCandidate],
                      cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(candidateIds, orderGm[firstCandidate], cp, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        lastCell = cell;
      }
      DataCopyExtParams cp{1, batch * 4, 0, 0, 0};
      for (uint32_t d = 0; d < itemLength; ++d)
        DataCopyPad(queries[d * TILE], sortedXGm[static_cast<uint64_t>(d) * xSize + position],
                    cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
      DataCopyPad(ids, orderGm[position], cp, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
      PipeBarrier<PIPE_ALL>();
      for (uint32_t b = 0; b < batch; ++b) {
        const int32_t original = ids.GetValue(b);
        Duplicate(distance, float(1e30), GROUP_CANDIDATES); PipeBarrier<PIPE_V>();
        Duplicate(distance, float(0), candidateCount); PipeBarrier<PIPE_V>();
        for (uint32_t d = 0; d < itemLength; ++d) {
          Adds(temporary, raw[d * GROUP_CANDIDATES], -queries.GetValue(d * TILE + b), candidateCount);
          Mul(temporary, temporary, temporary, candidateCount);
          Add(distance, distance, temporary, candidateCount);
        }
        PipeBarrier<PIPE_ALL>();
        if (ignoreSameIndex) {
          for (uint32_t u = 0; u < candidateCount; ++u)
            if (candidateIds.GetValue(u) == original) distance.SetValue(u, float(1e30));
          PipeBarrier<PIPE_ALL>();
        }
        CompareScalar(mask, distance, r2, CMPMODE::LE, GROUP_CANDIDATES); PipeBarrier<PIPE_V>();
        uint64_t count = 0;
        GatherMask(filtered, candidateIds, mask.ReinterpretCast<uint32_t>(), true, GROUP_CANDIDATES,
                   GatherMaskParams{1, 1, 8, 0}, count); PipeBarrier<PIPE_ALL>();
        if (count < maxNumNeighbors) {
          // Keep the validated prefix and continue beyond the cached points;
          // restarting the whole query repeats both DMA and distance work.
          Duplicate(resX, int32_t(-1), AlignUp(maxNumNeighbors));
          Duplicate(resY, original, AlignUp(maxNumNeighbors));
          if (count) Adds(resX, filtered, int32_t(0), static_cast<uint32_t>(count));
          auto cachedY = yValBuf.Get<float>();
          for (uint32_t d = 0; d < itemLength; ++d)
            cachedY.SetValue(d, queries.GetValue(d * TILE + b));
          PipeBarrier<PIPE_ALL>();
          ComputeNeighborsGrid(original, 0, 0, xSize, position + b,
                               static_cast<int32_t>(count), cell, candidateCount);
        } else {
          Adds(resX, filtered, int32_t(0), maxNumNeighbors);
          Duplicate(resY, original, maxNumNeighbors);
          WriteOutput(position + b, resX, resY);
        }
      }
      position += batch;
    }
    }
  }

  // Two passes avoid dynamic NonZero and two large framework gathers. Each
  // core first counts its contiguous edge range, then writes at its prefix.
  __aicore__ inline void ProcessCompact()
  {
    auto source = compactSrcBuf.Get<int32_t>();
    auto values = compactFloatBuf.Get<float>();
    auto mask = compactMaskBuf.Get<uint8_t>();
    auto compactX = compactXBuf.Get<int32_t>();
    auto compactY = compactYBuf.Get<int32_t>();
    auto offsets = expandOffsetsBuf.Get<int32_t>();
    auto expanded = expandedBuf.Get<int32_t>();
    auto work = expandWorkBuf.Get<int32_t>();
    const uint32_t span = ((maxSize + coreNum - 1) / coreNum + 15) / 16 * 16;
    const uint32_t begin = coreId * span < maxSize ? coreId * span : maxSize;
    const uint32_t end = begin + span < maxSize ? begin + span : maxSize;
    uint32_t destination = 0, total = 0, count = 0;
    if (phase == 2) {
      for (uint32_t core = 0; core < coreNum; ++core) {
        const uint32_t value = compactStateGm.GetValue(core * 16);
        if (core < coreId) destination += value;
        total += value;
      }
      if (coreId == 0) {
        compactStateGm.SetValue(640, total);
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(compactStateGm);
      }
      // Interleave each int32 index with a zero high word using vector Gather.
      CreateVecIndex(work, int32_t(0), COMPACT_TILE * 2); PipeBarrier<PIPE_V>();
      ShiftRight(offsets, work, int32_t(1), COMPACT_TILE * 2); PipeBarrier<PIPE_V>();
      Muls(expanded, offsets, int32_t(2), COMPACT_TILE * 2); PipeBarrier<PIPE_V>();
      Sub(work, work, expanded, COMPACT_TILE * 2); PipeBarrier<PIPE_V>();
      Muls(expanded, work, int32_t(-1), COMPACT_TILE * 2); PipeBarrier<PIPE_V>();
      Adds(expanded, expanded, int32_t(1), COMPACT_TILE * 2); PipeBarrier<PIPE_V>();
      Muls(offsets, offsets, int32_t(4), COMPACT_TILE * 2); PipeBarrier<PIPE_V>();
      Mul(offsets, offsets, expanded, COMPACT_TILE * 2); PipeBarrier<PIPE_V>();
      Muls(work, work, int32_t(4 * (COMPACT_TILE + 8)), COMPACT_TILE * 2); PipeBarrier<PIPE_V>();
      Add(offsets, offsets, work, COMPACT_TILE * 2); PipeBarrier<PIPE_V>();
      Duplicate(compactX[COMPACT_TILE + 8], int32_t(0), 8);
      Duplicate(compactY[COMPACT_TILE + 8], int32_t(0), 8); PipeBarrier<PIPE_ALL>();
    }
    for (uint32_t off = begin; off < end; off += COMPACT_TILE) {
      const uint32_t length = end - off < COMPACT_TILE ? end - off : COMPACT_TILE;
      const uint32_t padded = (length + 63) / 64 * 64;
      Duplicate(values, float(-1), padded); PipeBarrier<PIPE_ALL>();
      DataCopyExtParams cp{1, length * 4, 0, 0, 0};
      DataCopyPad(source, outGm[off], cp, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
      PipeBarrier<PIPE_ALL>();
      Cast(values, source, RoundMode::CAST_NONE, length); PipeBarrier<PIPE_V>();
      CompareScalar(mask, values, float(0), CMPMODE::GE, padded); PipeBarrier<PIPE_V>();
      uint64_t selected = 0;
      GatherMask(compactX, source, mask.ReinterpretCast<uint32_t>(), true, padded,
                 GatherMaskParams{1, 1, 8, 0}, selected);
      PipeBarrier<PIPE_ALL>();
      if (phase == 2 && selected) {
        DataCopyPad(source, outGm[static_cast<uint64_t>(maxSize) + off], cp,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0}); PipeBarrier<PIPE_ALL>();
        uint64_t rows = 0;
        GatherMask(compactY, source, mask.ReinterpretCast<uint32_t>(), true, padded,
                   GatherMaskParams{1, 1, 8, 0}, rows); PipeBarrier<PIPE_ALL>();
        DataCopyExtParams outputCopy{1, static_cast<uint32_t>(selected * 8), 0, 0, 0};
        Gather(expanded, compactY, offsets.ReinterpretCast<uint32_t>(), 0, selected * 2);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(packedGm[static_cast<uint64_t>(destination + count) * 2], expanded, outputCopy);
        PipeBarrier<PIPE_ALL>();
        Gather(expanded, compactX, offsets.ReinterpretCast<uint32_t>(), 0, selected * 2);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(packedGm[static_cast<uint64_t>(total + destination + count) * 2], expanded, outputCopy);
        PipeBarrier<PIPE_ALL>();
      }
      count += selected;
    }
    if (phase == 1) {
      compactStateGm.SetValue(coreId * 16, count);
      DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(compactStateGm);
    }
  }

  // ===================== 通用：从 GM 载入一列到 UB(fp32) =====================
  __aicore__ inline void LoadColumnFrom(GlobalTensor<TYPE_X> &gm, uint32_t d, int32_t ts,
                                        int32_t Tn, LocalTensor<float> &xf)
  {
    if constexpr (std::is_same_v<TYPE_X, float>)
    {
      DataCopyExtParams cp{1, (uint32_t)(Tn * (int32_t)sizeof(float)), 0, 0, 0};
      DataCopyPad(xf, gm[(int64_t)d * xSize + ts], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
    }
    else
    {
      LocalTensor<TYPE_X> xt = xTileBuf.Get<TYPE_X>();
      DataCopyExtParams cp{1, (uint32_t)(Tn * (int32_t)sizeof(TYPE_X)), 0, 0, 0};
      DataCopyPad(xt, gm[(int64_t)d * xSize + ts], cp,
                  DataCopyPadExtParams<TYPE_X>{false, 0, 0, (TYPE_X)0});
      PipeBarrier<PIPE_ALL>();
      constexpr auto mode = std::is_same_v<TYPE_X, int32_t> ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
      Cast(xf, xt, mode, Tn);
    }
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void ComputeDistTileFrom(GlobalTensor<TYPE_X> &gm, int32_t ts, int32_t Tn,
                                             LocalTensor<float> &yVal)
  {
    LocalTensor<float> sum = sumBuf.Get<float>();
    // xAllBuf holds up to 4 feature columns; the grid path guarantees F<=4.
    // This fast path only exists for float inputs.
    if constexpr (std::is_same_v<TYPE_X, float>)
    {
      if (itemLength <= 4)
      {
        // Load every feature column of this tile with the raw DMA engine, then
        // a single MTE2->V barrier: a tile costs two barriers instead of 2F+1.
        LocalTensor<float> cols = xAllBuf.Get<float>();
        for (uint32_t d = 0; d < itemLength; ++d)
        {
          DataCopyExtParams cp{1, (uint32_t)(Tn * (int32_t)sizeof(float)), 0, 0, 0};
          DataCopyPad(cols[(int64_t)d * TILE], gm[(int64_t)d * xSize + ts], cp,
                      DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
        Duplicate(sum, 0.0f, Tn);
        PipeBarrier<PIPE_V>();
        for (uint32_t d = 0; d < itemLength; ++d)
        {
          LocalTensor<float> xf = cols[(int64_t)d * TILE];
          Adds(xf, xf, -yVal.GetValue(d), Tn);
          Mul(xf, xf, xf, Tn);
          Add(sum, sum, xf, Tn);
        }
        PipeBarrier<PIPE_V>();
        return;
      }
    }
    Duplicate(sum, 0.0f, Tn);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t d = 0; d < itemLength; ++d)
    {
      LocalTensor<float> xf = xFpBuf.Get<float>();
      LoadColumnFrom(gm, d, ts, Tn, xf);
      Adds(xf, xf, -yVal.GetValue(d), Tn);
      Mul(xf, xf, xf, Tn);
      Add(sum, sum, xf, Tn);
      PipeBarrier<PIPE_ALL>();
    }
  }

  __aicore__ inline void LoadYVals(int32_t i, LocalTensor<float> &yVal)
  {
    for (uint32_t d = 0; d < itemLength; ++d)
    {
      yVal.SetValue(d, (float)yGm.GetValue((int64_t)d * ySize + i));
    }
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void WriteOutput(int32_t i, LocalTensor<int32_t> &resX, LocalTensor<int32_t> &resY)
  {
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams cp{1, (uint32_t)((int32_t)maxNumNeighbors * (int32_t)sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(outGm[i * (int32_t)maxNumNeighbors], resX, cp);
    DataCopyPad(outGm[(int32_t)maxSize + i * (int32_t)maxNumNeighbors], resY, cp);
    PipeBarrier<PIPE_ALL>();
  }

  // ===================== hash-grid 查询 =====================
  __aicore__ inline void ProcessGrid()
  {
    if (ptrYLen == 0)
    {
      int32_t yPerCore = (int32_t)((ySize + coreNum - 1) / coreNum);
      int32_t yStart = (int32_t)(coreId * yPerCore);
      int32_t yEnd = yStart + yPerCore;
      if (yStart >= (int32_t)ySize)
      {
        return;
      }
      if (yEnd > (int32_t)ySize)
      {
        yEnd = (int32_t)ySize;
      }
      for (int32_t i = yStart; i < yEnd; i++)
      {
        ComputeNeighborsGrid(i, 0, 0, (int32_t)xSize);
      }
    }
    else
    {
      int32_t segNum = (int32_t)ptrYLen - 1;
      for (int32_t q = 0; q < segNum; q++)
      {
        int32_t yStart = ptrYGm.GetValue(q);
        int32_t yEnd = ptrYGm.GetValue(q + 1);
        int32_t xStart = ptrXGm.GetValue(q);
        int32_t xEnd = ptrXGm.GetValue(q + 1);
        if (yStart == yEnd || xStart == xEnd)
        {
          continue;
        }
        int32_t yCount = yEnd - yStart;
        int32_t yPerCore = (yCount + (int32_t)coreNum - 1) / (int32_t)coreNum;
        int32_t yCoreStart = yStart + (int32_t)coreId * yPerCore;
        int32_t yCoreEnd = yCoreStart + yPerCore;
        if (yCoreStart >= yEnd)
        {
          continue;
        }
        if (yCoreEnd > yEnd)
        {
          yCoreEnd = yEnd;
        }
        for (int32_t i = yCoreStart; i < yCoreEnd; i++)
        {
          ComputeNeighborsGrid(i, q, xStart, xEnd);
        }
      }
    }
  }

  __aicore__ inline void ComputeNeighborsGrid(int32_t i, int32_t q, int32_t xs, int32_t xe,
                                              int32_t outputSlot = -1, int32_t initialCount = 0,
                                              int32_t processedCell = -1, uint32_t processedCandidates = 0)
  {
    LocalTensor<int32_t> resX = resXBuf.Get<int32_t>();
    LocalTensor<int32_t> resY = resYBuf.Get<int32_t>();
    LocalTensor<float> yVal = yValBuf.Get<float>();
    if (processedCell < 0) {
      Duplicate(resX, (int32_t)-1, AlignUp((int)maxNumNeighbors));
      Duplicate(resY, (int32_t)-1, AlignUp((int)maxNumNeighbors));
      PipeBarrier<PIPE_ALL>();
      LoadYVals(i, yVal);
    }

    int32_t csOff = cellStartOffGm.GetValue(q);
    int32_t base = xs;
    int32_t gd[4];
    int32_t str[4];
    int32_t c0[4];
    int32_t G = 1;
    for (uint32_t d = 0; d < itemLength; ++d)
    {
      float mnd = gridMinGm.GetValue((int64_t)q * itemLength + d);
      int32_t g = gridGGm.GetValue((int64_t)q * itemLength + d);
      if (g < 1)
      {
        g = 1;
      }
      int32_t c = (int32_t)((yVal.GetValue(d) - mnd) / r);
      if (c < 0)
      {
        c = 0;
      }
      if (c >= g)
      {
        c = g - 1;
      }
      gd[d] = g;
      str[d] = G;
      c0[d] = c;
      G *= g;
    }

    int32_t combos = 1;
    for (uint32_t d = 0; d < itemLength; ++d)
    {
      combos *= 3;
    }

    LocalTensor<float> sum = sumBuf.Get<float>();
    LocalTensor<float> rOut = reduceOutBuf.Get<float>();
    LocalTensor<float> rWork = reduceWorkBuf.Get<float>();

    int32_t count = initialCount;
    for (int32_t c = 0; c < combos; ++c)
    {
      // The current cell has the highest acceptance rate for capped queries.
      // Try it first; enumerate every remaining neighbour exactly once.
      const int32_t center = (combos - 1) / 2;
      int32_t t = c == 0 ? center : (c <= center ? c - 1 : c);
      bool ok = true;
      int32_t cell = 0;
      for (uint32_t d = 0; d < itemLength; ++d)
      {
        int32_t delta = (t % 3) - 1;
        t /= 3;
        int32_t nd = c0[d] + delta;
        if (nd < 0 || nd >= gd[d])
        {
          ok = false;
          break;
        }
        cell += nd * str[d];
      }
      if (!ok)
      {
        continue;
      }
      int32_t s = cellStartGm.GetValue(csOff + cell);
      int32_t e = cellStartGm.GetValue(csOff + cell + 1);
      int32_t n = e - s;
      if (n <= 0)
      {
        continue;
      }
      int32_t abase = base + s;
      const int32_t firstUnprocessed = cell == processedCell ? processedCandidates : 0;
      for (int32_t off = firstUnprocessed; off < n && count < (int32_t)maxNumNeighbors; off += TILE)
      {
        int32_t Tn = (n - off < TILE) ? (n - off) : TILE;
        ComputeDistTileFrom(sortedXGm, abase + off, Tn, yVal);
        ReduceMin<float>(rOut, sum, rWork, Tn, false);
        PipeBarrier<PIPE_ALL>();
        if (rOut.GetValue(0) > this->r2)
        {
          continue;
        }
        auto orderTile = orderTileBuf.Get<int32_t>();
        DataCopyExtParams orderCopy{1, static_cast<uint32_t>(Tn * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(orderTile, orderGm[abase + off], orderCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        for (int32_t u = 0; u < Tn; ++u)
        {
          if (sum.GetValue(u) <= this->r2)
          {
            int32_t orig = orderTile.GetValue(u);
            if (ignoreSameIndex && orig == i)
            {
              continue;
            }
            resX.SetValue(count, orig);
            resY.SetValue(count, (int32_t)i);
            if (++count == (int32_t)maxNumNeighbors)
            {
              break;
            }
          }
        }
      }
      if (count == (int32_t)maxNumNeighbors)
      {
        break;
      }
    }
    WriteOutput(outputSlot >= 0 ? outputSlot : i, resX, resY);
  }

  // ===================== 暴力回退（SoA + 包围盒 + 块级剪枝） =====================
  __aicore__ inline void PrecomputeBbox()
  {
    if (!useBbox)
    {
      return;
    }
    LocalTensor<float> bmin = bboxMinBuf.Get<float>();
    LocalTensor<float> bmax = bboxMaxBuf.Get<float>();
    LocalTensor<float> xf = xFpBuf.Get<float>();
    LocalTensor<float> rOut = reduceOutBuf.Get<float>();
    LocalTensor<float> rWork = reduceWorkBuf.Get<float>();
    for (uint32_t d = 0; d < itemLength; ++d)
    {
      for (int32_t t = 0; t < numTiles; ++t)
      {
        int32_t ts = t * TILE;
        int32_t Tn = (xSize - ts < TILE) ? (xSize - ts) : TILE;
        LoadColumnFrom(xGm, d, ts, Tn, xf);
        ReduceMin<float>(rOut, xf, rWork, Tn, false);
        PipeBarrier<PIPE_ALL>();
        bmin.SetValue((int32_t)d * numTiles + t, rOut.GetValue(0));
        ReduceMax<float>(rOut, xf, rWork, Tn, false);
        PipeBarrier<PIPE_ALL>();
        bmax.SetValue((int32_t)d * numTiles + t, rOut.GetValue(0));
      }
    }
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void ProcessBrute()
  {
    if (ptrYLen == 0)
    {
      int32_t yPerCore = (int32_t)((ySize + coreNum - 1) / coreNum);
      int32_t yStart = (int32_t)(coreId * yPerCore);
      int32_t yEnd = yStart + yPerCore;
      if (yStart >= (int32_t)ySize)
      {
        return;
      }
      if (yEnd > (int32_t)ySize)
      {
        yEnd = (int32_t)ySize;
      }
      for (int32_t i = yStart; i < yEnd; i++)
      {
        ComputeNeighborsBrute(i, 0, (int32_t)xSize);
      }
    }
    else
    {
      int32_t segNum = (int32_t)ptrYLen - 1;
      for (int32_t q = 0; q < segNum; q++)
      {
        int32_t yStart = ptrYGm.GetValue(q);
        int32_t yEnd = ptrYGm.GetValue(q + 1);
        int32_t xStart = ptrXGm.GetValue(q);
        int32_t xEnd = ptrXGm.GetValue(q + 1);
        if (yStart == yEnd || xStart == xEnd)
        {
          continue;
        }
        int32_t yCount = yEnd - yStart;
        int32_t yPerCore = (yCount + (int32_t)coreNum - 1) / (int32_t)coreNum;
        int32_t yCoreStart = yStart + (int32_t)coreId * yPerCore;
        int32_t yCoreEnd = yCoreStart + yPerCore;
        if (yCoreStart >= yEnd)
        {
          continue;
        }
        if (yCoreEnd > yEnd)
        {
          yCoreEnd = yEnd;
        }
        for (int32_t i = yCoreStart; i < yCoreEnd; i++)
        {
          ComputeNeighborsBrute(i, xStart, xEnd);
        }
      }
    }
  }

  __aicore__ inline void ComputeNeighborsBrute(int32_t i, int32_t xs, int32_t xe)
  {
    LocalTensor<int32_t> resX = resXBuf.Get<int32_t>();
    LocalTensor<int32_t> resY = resYBuf.Get<int32_t>();
    Duplicate(resX, (int32_t)-1, AlignUp((int)maxNumNeighbors));
    Duplicate(resY, (int32_t)-1, AlignUp((int)maxNumNeighbors));
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> yVal = yValBuf.Get<float>();
    LoadYVals(i, yVal);

    LocalTensor<float> sum = sumBuf.Get<float>();
    LocalTensor<float> rOut = reduceOutBuf.Get<float>();
    LocalTensor<float> rWork = reduceWorkBuf.Get<float>();
    LocalTensor<float> bmin = bboxMinBuf.Get<float>();
    LocalTensor<float> bmax = bboxMaxBuf.Get<float>();

    int32_t count = 0;
    int32_t tStart = xs / TILE;
    int32_t tEnd = (xe - 1) / TILE;
    for (int32_t t = tStart; t <= tEnd && count < (int32_t)maxNumNeighbors; ++t)
    {
      int32_t ts = t * TILE;
      if (ts < xs)
      {
        ts = xs;
      }
      int32_t te = (t + 1) * TILE;
      if (te > xe)
      {
        te = xe;
      }
      int32_t Tn = te - ts;
      if (Tn <= 0)
      {
        continue;
      }
      if (useBbox)
      {
        bool skip = false;
        for (uint32_t d = 0; d < itemLength; ++d)
        {
          float yv = yVal.GetValue(d);
          float lo = bmin.GetValue((int32_t)d * numTiles + t);
          float hi = bmax.GetValue((int32_t)d * numTiles + t);
          if (yv < lo - r || yv > hi + r)
          {
            skip = true;
            break;
          }
        }
        if (skip)
        {
          continue;
        }
      }
      ComputeDistTileFrom(xGm, ts, Tn, yVal);
      ReduceMin<float>(rOut, sum, rWork, Tn, false);
      PipeBarrier<PIPE_ALL>();
      if (rOut.GetValue(0) > this->r2)
      {
        continue;
      }
      for (int32_t u = 0; u < Tn; ++u)
      {
        if (sum.GetValue(u) <= this->r2)
        {
          int32_t j = ts + u;
          if (j == i && ignoreSameIndex)
          {
            continue;
          }
          resX.SetValue(count, (int32_t)j);
          resY.SetValue(count, (int32_t)i);
          if (++count == (int32_t)maxNumNeighbors)
          {
            break;
          }
        }
      }
    }
    WriteOutput(i, resX, resY);
  }

  TPipe pipe;
  static constexpr uint32_t COMPACT_TILE = 1024;
  static constexpr uint32_t NATIVE_TILE = 256;
  uint32_t phase;
  uint32_t gridCells, gridStride, nativeCores, bucketCapacity, histogramStride, queryMode;
  GlobalTensor<float> boundsGm;
  TBuf<QuePosition::VECCALC> nativeTableBuf, nativeCellsBuf, nativeCoordsBuf, nativeBucketsBuf;
  GlobalTensor<int32_t> compactStateGm, packedGm;
  TBuf<QuePosition::VECCALC> compactXBuf, compactYBuf, compactSrcBuf, compactFloatBuf, compactMaskBuf;
  TBuf<QuePosition::VECCALC> expandOffsetsBuf, expandedBuf, expandWorkBuf;
  GlobalTensor<TYPE_X> xGm, yGm, sortedXGm;
  GlobalTensor<int32_t> ptrXGm, ptrYGm, outGm, orderGm, cellStartGm, cellStartOffGm, gridGGm, useGridGm;
  GlobalTensor<float> gridMinGm;
  TBuf<QuePosition::VECCALC> xTileBuf, xFpBuf, xAllBuf, sumBuf, reduceWorkBuf, reduceOutBuf;
  TBuf<QuePosition::VECCALC> bboxMinBuf, bboxMaxBuf, yValBuf, cellDimsBuf, resXBuf, resYBuf;
  TBuf<QuePosition::VECCALC> orderTileBuf;
  TBuf<QuePosition::VECCALC> equalityBuf, groupRawXBuf, groupYBuf, groupIdBuf, groupOutBuf, groupCandidateIdBuf;
  bool groupedGrid = false;
  float r, r2;
  bool useBbox = false;
  bool useGrid = false;
  uint32_t ignoreSameIndex, maxNumNeighbors;
  uint32_t ySize, itemLength, xSize, maxSize, ptrXLen, ptrYLen, coreNum, coreId;
  int32_t numTiles;
};

extern "C" __global__ __aicore__ void radius(GM_ADDR x, GM_ADDR y, GM_ADDR ptr_x, GM_ADDR ptr_y,
                                             GM_ADDR sorted_x, GM_ADDR order, GM_ADDR cell_start,
                                             GM_ADDR cell_start_off, GM_ADDR grid_min, GM_ADDR grid_g,
                                             GM_ADDR use_grid, GM_ADDR compact_state, GM_ADDR packed, GM_ADDR out,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
  GET_TILING_DATA(tiling_data, tiling);
  KernelRadius<DTYPE_X> op;
  op.Init(x, y, ptr_x, ptr_y, sorted_x, order, cell_start, cell_start_off, grid_min, grid_g,
          use_grid, compact_state, packed, out,
          tiling_data.r, tiling_data.ignore_same_index, tiling_data.max_num_neighbors,
          tiling_data.xSize, tiling_data.ySize, tiling_data.itemLength,
          tiling_data.ptrXLen, tiling_data.ptrYLen, tiling_data.phase,
          tiling_data.gridCells, tiling_data.nativeCores, tiling_data.queryMode);
  op.Process();
}
