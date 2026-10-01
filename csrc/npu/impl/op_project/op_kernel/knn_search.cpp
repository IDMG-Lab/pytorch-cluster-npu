#include "kernel_operator.h"
using namespace AscendC;

class KernelKNNSearch {
public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR ptrX,
                              GM_ADDR ptrY, GM_ADDR out,
                              uint32_t n, uint32_t m, uint32_t d,
                              uint32_t batches, uint32_t k,
                              uint32_t cosine, uint32_t queriesPerCore) {
    this->n = n;
    this->m = m;
    this->d = d;
    this->batches = batches;
    this->k = k;
    this->cosine = cosine != 0;
    this->begin = GetBlockIdx() * queriesPerCore;
    this->end = begin + queriesPerCore < m ? begin + queriesPerCore : m;
    xGm.SetGlobalBuffer((__gm__ float*)x, static_cast<uint64_t>(n) * d);
    yGm.SetGlobalBuffer((__gm__ float*)y, static_cast<uint64_t>(m) * d);
    px.SetGlobalBuffer((__gm__ int64_t*)ptrX, batches + 1);
    py.SetGlobalBuffer((__gm__ int64_t*)ptrY, batches + 1);
    output.SetGlobalBuffer((__gm__ int64_t*)out,
                           static_cast<uint64_t>(2) * m * k);
  }

  __aicore__ inline void Process() {
    const uint64_t plane = static_cast<uint64_t>(m) * k;
    for (uint32_t queryId = begin; queryId < end; ++queryId) {
      int64_t batch = -1;
      for (uint32_t b = 0; b < batches; ++b) {
        if (queryId >= py.GetValue(b) && queryId < py.GetValue(b + 1)) {
          batch = b;
          break;
        }
      }
      for (uint32_t slot = 0; slot < k; ++slot) {
        best[slot] = 3.402823466e38f;
        bestId[slot] = -1;
      }
      const uint32_t cached = d < 1024 ? d : 1024;
      for (uint32_t feature = 0; feature < cached; ++feature)
        query[feature] = yGm.GetValue(static_cast<uint64_t>(queryId) * d + feature);
      if (batch >= 0) {
        const int64_t first = px.GetValue(batch);
        const int64_t last = px.GetValue(batch + 1);
        for (int64_t reference = first; reference < last; ++reference) {
          float distance = 0.0f;
          for (uint32_t feature = 0; feature < d; ++feature) {
            const float a = xGm.GetValue(static_cast<uint64_t>(reference) * d + feature);
            const float b = feature < cached ? query[feature] :
                yGm.GetValue(static_cast<uint64_t>(queryId) * d + feature);
            if (cosine) distance += a * b;
            else { const float delta = a - b; distance += delta * delta; }
          }
          if (cosine) distance = 1.0f - distance;
          for (uint32_t slot = 0; slot < k; ++slot) {
            if (distance < best[slot]) {
              for (uint32_t move = k - 1; move > slot; --move) {
                best[move] = best[move - 1];
                bestId[move] = bestId[move - 1];
              }
              best[slot] = distance;
              bestId[slot] = reference;
              break;
            }
          }
        }
      }
      for (uint32_t slot = 0; slot < k; ++slot) {
        const uint64_t position = static_cast<uint64_t>(queryId) * k + slot;
        output.SetValue(position, queryId);
        output.SetValue(plane + position, bestId[slot]);
      }
    }
    DataCacheCleanAndInvalid<int64_t, CacheLine::ENTIRE_DATA_CACHE>(output);
  }

private:
  GlobalTensor<float> xGm, yGm;
  GlobalTensor<int64_t> px, py, output;
  float query[1024];
  float best[100];
  int64_t bestId[100];
  uint32_t n, m, d, batches, k, begin, end;
  bool cosine;
};

extern "C" __global__ __aicore__ void knn_search(
    GM_ADDR x, GM_ADDR y, GM_ADDR ptrX, GM_ADDR ptrY,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(data, tiling);
  KernelKNNSearch op;
  op.Init(x, y, ptrX, ptrY, out, data.numReference, data.numQuery,
          data.dimension, data.numBatches, data.k, data.cosine,
          data.queriesPerCore);
  op.Process();
}
