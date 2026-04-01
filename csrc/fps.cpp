// fps.cpp  (NPU-aware version)
// Extends the original pytorch_cluster fps operator with NPU dispatch.
//
// Device routing:
//   CUDA  -> fps_cuda  (original CUDA implementation)
//   NPU   -> fps_npu   (Ascend NPU implementation via EXEC_NPU_CMD)
//   CPU   -> fps_cpu   (original CPU implementation)

#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/fps_cpu.h"

#ifdef WITH_CUDA
#include "cuda/fps_cuda.h"
#endif

#ifdef WITH_NPU
#include "npu/fps_npu.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__fps_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__fps_cpu(void) { return NULL; }
#endif
#endif
#endif

#define CLUSTER_API

CLUSTER_API torch::Tensor fps(torch::Tensor src, torch::Tensor ptr,
                               torch::Tensor ratio, bool random_start) {
  if (src.device().is_cuda()) {
#ifdef WITH_CUDA
    return fps_cuda(src, ptr, ratio, random_start);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else if (src.device().type() == c10::DeviceType::PrivateUse1) {
#ifdef WITH_NPU
    return fps_npu(src, ptr, ratio, random_start);
#else
    AT_ERROR("Not compiled with NPU support");
#endif
  } else {
    return fps_cpu(src, ptr, ratio, random_start);
  }
}

static auto registry =
    torch::RegisterOperators().op("torch_cluster::fps", &fps);
