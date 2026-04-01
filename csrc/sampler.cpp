// sampler.cpp  (NPU-aware version)
#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/sampler_cpu.h"

// Note: original sampler.cpp has no CUDA implementation
#ifdef WITH_NPU
#include "npu/sampler_npu.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
PyMODINIT_FUNC PyInit__sampler_cpu(void) { return NULL; }
#endif
#endif

#define CLUSTER_API

CLUSTER_API torch::Tensor neighbor_sampler(torch::Tensor start,
                                            torch::Tensor rowptr,
                                            int64_t count,
                                            double factor) {
  if (start.device().is_cuda()) {
    AT_ERROR("neighbor_sampler: CUDA implementation not available");
  } else if (start.device().type() == c10::DeviceType::PrivateUse1) {
#ifdef WITH_NPU
    return neighbor_sampler_npu(start, rowptr, count, factor);
#else
    AT_ERROR("Not compiled with NPU support");
#endif
  } else {
    return neighbor_sampler_cpu(start, rowptr, count, factor);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::neighbor_sampler",
                                  &neighbor_sampler);
