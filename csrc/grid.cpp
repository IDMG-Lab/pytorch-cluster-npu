// grid.cpp  (NPU-aware version)
#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/grid_cpu.h"

#ifdef WITH_CUDA
#include "cuda/grid_cuda.h"
#endif

#ifdef WITH_NPU
#include "npu/grid_npu.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__grid_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__grid_cpu(void) { return NULL; }
#endif
#endif
#endif

#define CLUSTER_API

CLUSTER_API torch::Tensor grid(torch::Tensor pos, torch::Tensor size,
                                std::optional<torch::Tensor> optional_start,
                                std::optional<torch::Tensor> optional_end) {
  if (pos.device().is_cuda()) {
#ifdef WITH_CUDA
    return grid_cuda(pos, size, optional_start, optional_end);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else if (pos.device().type() == c10::DeviceType::PrivateUse1) {
#ifdef WITH_NPU
    return grid_npu(pos, size, optional_start, optional_end);
#else
    AT_ERROR("Not compiled with NPU support");
#endif
  } else {
    return grid_cpu(pos, size, optional_start, optional_end);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::grid", &grid);
