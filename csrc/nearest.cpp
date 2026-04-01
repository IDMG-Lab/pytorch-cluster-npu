// nearest.cpp  (NPU-aware version)
#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

// nearest has only CUDA implementation in the original library (no CPU impl)
#ifdef WITH_CUDA
#include "cuda/nearest_cuda.h"
#endif

#ifdef WITH_NPU
#include "npu/nearest_npu.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__nearest_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__nearest_cpu(void) { return NULL; }
#endif
#endif
#endif

#define CLUSTER_API

CLUSTER_API torch::Tensor nearest(torch::Tensor x, torch::Tensor y,
                                   torch::Tensor ptr_x,
                                   torch::Tensor ptr_y) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return nearest_cuda(x, y, ptr_x, ptr_y);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else if (x.device().type() == c10::DeviceType::PrivateUse1) {
#ifdef WITH_NPU
    return nearest_npu(x, y, ptr_x, ptr_y);
#else
    AT_ERROR("Not compiled with NPU support");
#endif
  } else {
    AT_ERROR("nearest: CPU implementation not available; "
             "please use CUDA or NPU device");
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::nearest", &nearest);
