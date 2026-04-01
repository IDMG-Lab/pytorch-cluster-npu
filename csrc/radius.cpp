// radius.cpp  (NPU-aware version)
#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/radius_cpu.h"

#ifdef WITH_CUDA
#include "cuda/radius_cuda.h"
#endif

#ifdef WITH_NPU
#include "npu/radius_npu.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__radius_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__radius_cpu(void) { return NULL; }
#endif
#endif
#endif

#define CLUSTER_API

CLUSTER_API torch::Tensor radius(torch::Tensor x, torch::Tensor y,
                                  std::optional<torch::Tensor> ptr_x,
                                  std::optional<torch::Tensor> ptr_y,
                                  double r, int64_t max_num_neighbors,
                                  int64_t num_workers,
                                  bool ignore_same_index) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return radius_cuda(x, y, ptr_x, ptr_y, r, max_num_neighbors,
                       num_workers, ignore_same_index);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else if (x.device().type() == c10::DeviceType::PrivateUse1) {
#ifdef WITH_NPU
    return radius_npu(x, y, ptr_x, ptr_y, r, max_num_neighbors,
                      num_workers, ignore_same_index);
#else
    AT_ERROR("Not compiled with NPU support");
#endif
  } else {
    return radius_cpu(x, y, ptr_x, ptr_y, r, max_num_neighbors,
                      num_workers, ignore_same_index);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::radius", &radius);
