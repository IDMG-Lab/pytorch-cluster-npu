
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

constexpr int32_t TILE = 256;
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
                              GM_ADDR use_grid, GM_ADDR out,
                              float r, uint32_t ignoreSameIndex, uint32_t maxNumNeighbors,
                              uint32_t xSize, uint32_t ySize, uint32_t itemLength,
                              uint32_t ptrXLen, uint32_t ptrYLen)
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
    this->numTiles = (xSize + TILE - 1) / TILE;

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
    useGrid = (useGridGm.GetValue(0) != 0) && (itemLength <= 4) && (r > 0.0f);

    // cell_start / cell_start_off 仅在 use_grid 时使用，这里延迟绑定（大小未知，用足够大的逻辑长度）
    if (useGrid)
    {
      cellStartGm.SetGlobalBuffer((__gm__ int32_t *)cell_start, (int64_t)1 << 40);
      cellStartOffGm.SetGlobalBuffer((__gm__ int32_t *)cell_start_off, (int64_t)1 << 40);
    }

    pipe.InitBuffer(xTileBuf, TILE * sizeof(TYPE_X));
    pipe.InitBuffer(xFpBuf, TILE * sizeof(float));
    pipe.InitBuffer(sumBuf, TILE * sizeof(float));
    pipe.InitBuffer(reduceWorkBuf, REDUCE_WORK * sizeof(float));
    pipe.InitBuffer(reduceOutBuf, 8 * sizeof(float));
    pipe.InitBuffer(bboxMinBuf, BBOX_CAP * sizeof(float));
    pipe.InitBuffer(bboxMaxBuf, BBOX_CAP * sizeof(float));
    pipe.InitBuffer(yValBuf, AlignUp((int)itemLength) * sizeof(float));
    pipe.InitBuffer(cellDimsBuf, 8 * sizeof(int32_t));
    pipe.InitBuffer(resXBuf, AlignUp((int)maxNumNeighbors) * sizeof(int32_t));
    pipe.InitBuffer(resYBuf, AlignUp((int)maxNumNeighbors) * sizeof(int32_t));
  }

  __aicore__ inline int AlignUp(int x) { return (x + 31) / 32 * 32; }

  __aicore__ inline void Process()
  {
    if (useGrid)
    {
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

  __aicore__ inline void ComputeNeighborsGrid(int32_t i, int32_t q, int32_t xs, int32_t xe)
  {
    LocalTensor<int32_t> resX = resXBuf.Get<int32_t>();
    LocalTensor<int32_t> resY = resYBuf.Get<int32_t>();
    Duplicate(resX, (int32_t)-1, AlignUp((int)maxNumNeighbors));
    Duplicate(resY, (int32_t)-1, AlignUp((int)maxNumNeighbors));
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> yVal = yValBuf.Get<float>();
    LoadYVals(i, yVal);

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

    int32_t count = 0;
    for (int32_t c = 0; c < combos; ++c)
    {
      int32_t t = c;
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
      for (int32_t off = 0; off < n && count < (int32_t)maxNumNeighbors; off += TILE)
      {
        int32_t Tn = (n - off < TILE) ? (n - off) : TILE;
        ComputeDistTileFrom(sortedXGm, abase + off, Tn, yVal);
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
            int32_t orig = orderGm.GetValue(abase + off + u);
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
    WriteOutput(i, resX, resY);
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
  GlobalTensor<TYPE_X> xGm, yGm, sortedXGm;
  GlobalTensor<int32_t> ptrXGm, ptrYGm, outGm, orderGm, cellStartGm, cellStartOffGm, gridGGm, useGridGm;
  GlobalTensor<float> gridMinGm;
  TBuf<QuePosition::VECCALC> xTileBuf, xFpBuf, sumBuf, reduceWorkBuf, reduceOutBuf;
  TBuf<QuePosition::VECCALC> bboxMinBuf, bboxMaxBuf, yValBuf, cellDimsBuf, resXBuf, resYBuf;
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
                                             GM_ADDR use_grid, GM_ADDR out,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
  GET_TILING_DATA(tiling_data, tiling);
  KernelRadius<DTYPE_X> op;
  op.Init(x, y, ptr_x, ptr_y, sorted_x, order, cell_start, cell_start_off, grid_min, grid_g,
          use_grid, out,
          tiling_data.r, tiling_data.ignore_same_index, tiling_data.max_num_neighbors,
          tiling_data.xSize, tiling_data.ySize, tiling_data.itemLength,
          tiling_data.ptrXLen, tiling_data.ptrYLen);
  op.Process();
}
