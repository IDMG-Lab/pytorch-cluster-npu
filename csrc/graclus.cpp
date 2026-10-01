// graclus.cpp  (NPU-aware version)
#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/graclus_cpu.h"

#ifdef WITH_CUDA
#include "cuda/graclus_cuda.h"
#endif

#ifdef WITH_NPU
#include "npu/graclus_npu.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__graclus_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__graclus_cpu(void) { return NULL; }
#endif
#endif
#endif

#define CLUSTER_API

static torch::Tensor graclus_dispatch_npu(torch::Tensor rowptr, torch::Tensor col,
                                   std::optional<torch::Tensor> optional_weight) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    return graclus_cuda(rowptr, col, optional_weight);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else if (rowptr.device().type() == c10::DeviceType::PrivateUse1) {
#ifdef WITH_NPU
    return graclus_npu(rowptr, col, optional_weight);
#else
    AT_ERROR("Not compiled with NPU support");
#endif
  } else {
    return graclus_cpu(rowptr, col, optional_weight);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::graclus", &graclus_dispatch_npu);
