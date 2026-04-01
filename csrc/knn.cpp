// knn.cpp  (NPU-aware version)
#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/knn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/knn_cuda.h"
#endif

#ifdef WITH_NPU
#include "npu/knn_npu.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__knn_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__knn_cpu(void) { return NULL; }
#endif
#endif
#endif

#define CLUSTER_API

CLUSTER_API torch::Tensor knn(torch::Tensor x, torch::Tensor y,
                               std::optional<torch::Tensor> ptr_x,
                               std::optional<torch::Tensor> ptr_y,
                               int64_t k, bool cosine, int64_t num_workers) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return knn_cuda(x, y, ptr_x, ptr_y, k, cosine, num_workers);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else if (x.device().type() == c10::DeviceType::PrivateUse1) {
#ifdef WITH_NPU
    return knn_npu(x, y, ptr_x, ptr_y, k, cosine, num_workers);
#else
    AT_ERROR("Not compiled with NPU support");
#endif
  } else {
    return knn_cpu(x, y, ptr_x, ptr_y, k, cosine, num_workers);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::knn", &knn);
