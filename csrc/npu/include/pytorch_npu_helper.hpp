// pytorch_npu_helper.hpp
// NPU helper utilities for pytorch-cluster-npu
// Adapted from pytorch-sparse-npu project
//
// This file provides the core infrastructure for dispatching custom
// pytorch_cluster operators to Ascend NPU hardware via CANN aclnn APIs.
//
// Usage: Include this file and use EXEC_NPU_CMD macro to call NPU kernels.
// Example:
//   EXEC_NPU_CMD(aclnnMyOp, input_tensor, output_tensor, attr_value);

#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>

// torch_npu provides NPU tensor support and CANN integration
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>

// ACL (Ascend Computing Language) C API headers
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#include <dlfcn.h>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace at_npu {
namespace native {

// ---------------------------------------------------------------------------
// Data-type mapping: ATen scalar type -> ACL data type
// ---------------------------------------------------------------------------
static const std::unordered_map<at::ScalarType, aclDataType>
    K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE = {
        {at::ScalarType::Byte, ACL_UINT8},
        {at::ScalarType::Char, ACL_INT8},
        {at::ScalarType::Short, ACL_INT16},
        {at::ScalarType::Int, ACL_INT32},
        {at::ScalarType::Long, ACL_INT64},
        {at::ScalarType::Half, ACL_FLOAT16},
        {at::ScalarType::Float, ACL_FLOAT},
        {at::ScalarType::Double, ACL_DOUBLE},
        {at::ScalarType::Bool, ACL_BOOL},
        {at::ScalarType::BFloat16, ACL_BF16},
};

inline aclDataType GetAclDataType(at::ScalarType dtype) {
  auto it = K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE.find(dtype);
  TORCH_CHECK(it != K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE.end(),
              "Unsupported ATen scalar type for NPU: ", dtype);
  return it->second;
}

// ---------------------------------------------------------------------------
// Type conversion: ATen Tensor -> aclTensor*
// ---------------------------------------------------------------------------
inline aclTensor *ConvertType(const at::Tensor &tensor) {
  if (!tensor.defined()) {
    return nullptr;
  }
  const auto &sizes = tensor.sizes();
  const auto &strides = tensor.strides();
  int64_t offset = tensor.storage_offset();
  aclDataType acl_dtype = GetAclDataType(tensor.scalar_type());

  // Determine ACL format based on tensor dimensions
  aclFormat format = (sizes.size() == 4) ? ACL_FORMAT_NCHW : ACL_FORMAT_ND;

  aclTensor *acl_tensor = aclCreateTensor(
      sizes.data(), sizes.size(), acl_dtype, strides.data(), offset, format,
      sizes.data(), // storage shape (same as logical for contiguous)
      sizes.size(), const_cast<void *>(tensor.data_ptr()));
  return acl_tensor;
}

// ---------------------------------------------------------------------------
// Type conversion: at::Scalar -> aclScalar*
// ---------------------------------------------------------------------------
inline aclScalar *ConvertType(const at::Scalar &scalar) {
  double val = scalar.toDouble();
  aclScalar *acl_scalar = aclCreateScalar(&val, ACL_FLOAT);
  return acl_scalar;
}

// ---------------------------------------------------------------------------
// Type conversion: at::IntArrayRef -> aclIntArray*
// ---------------------------------------------------------------------------
inline aclIntArray *ConvertType(at::IntArrayRef arr) {
  return aclCreateIntArray(arr.data(), arr.size());
}

// ---------------------------------------------------------------------------
// Type conversion: at::TensorList -> aclTensorList*
// ---------------------------------------------------------------------------
inline aclTensorList *ConvertType(const at::TensorList &tensor_list) {
  std::vector<const aclTensor *> acl_tensors;
  acl_tensors.reserve(tensor_list.size());
  for (const auto &t : tensor_list) {
    acl_tensors.push_back(ConvertType(t));
  }
  return aclCreateTensorList(acl_tensors.data(), acl_tensors.size());
}

// ---------------------------------------------------------------------------
// Identity conversion for primitive types (int64_t, double, bool, ...)
// ---------------------------------------------------------------------------
template <typename T,
          std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
inline T ConvertType(T val) {
  return val;
}

// ---------------------------------------------------------------------------
// Release helpers
// ---------------------------------------------------------------------------
inline void ReleaseConvertType(aclTensor *tensor) {
  if (tensor) {
    aclDestroyTensor(tensor);
  }
}

inline void ReleaseConvertType(aclScalar *scalar) {
  if (scalar) {
    aclDestroyScalar(scalar);
  }
}

inline void ReleaseConvertType(aclIntArray *arr) {
  if (arr) {
    aclDestroyIntArray(arr);
  }
}

inline void ReleaseConvertType(aclTensorList *list) {
  if (list) {
    aclDestroyTensorList(list);
  }
}

template <typename T,
          std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
inline void ReleaseConvertType(T) {} // no-op for primitives

// ---------------------------------------------------------------------------
// Dynamic library lookup for aclnn kernel functions
// ---------------------------------------------------------------------------

// Search order:
//  1. ASCEND_CUSTOM_OPP_PATH/libcust_opapi.so  (user custom ops)
//  2. ASCEND_OPP_PATH/lib64/libopapi.so         (system CANN ops)
inline void *GetOpApiFuncAddr(const char *op_name) {
  // Try custom ops library first
  const char *custom_opp_path = std::getenv("ASCEND_CUSTOM_OPP_PATH");
  if (custom_opp_path) {
    std::string cust_lib =
        std::string(custom_opp_path) + "/lib64/libcust_opapi.so";
    void *handle = dlopen(cust_lib.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    if (handle) {
      void *func = dlsym(handle, op_name);
      if (func) {
        return func;
      }
    }
  }

  // Fallback: system CANN library
  const char *opp_path = std::getenv("ASCEND_OPP_PATH");
  std::string sys_lib = opp_path
                            ? std::string(opp_path) + "/lib64/libopapi.so"
                            : "/usr/local/Ascend/ascend-toolkit/latest/lib64/"
                              "libopapi.so";
  void *handle = dlopen(sys_lib.c_str(), RTLD_LAZY);
  if (!handle) {
    return nullptr;
  }
  return dlsym(handle, op_name);
}

// ---------------------------------------------------------------------------
// Workspace allocation helper
// ---------------------------------------------------------------------------
inline at::Tensor AllocateWorkspace(uint64_t workspace_size) {
  if (workspace_size == 0) {
    return at::Tensor();
  }
  auto options = at::TensorOptions()
                     .dtype(at::kByte)
                     .device(at::kPrivateUse1); // kPrivateUse1 == NPU
  return at::empty({static_cast<int64_t>(workspace_size)}, options);
}

} // namespace native
} // namespace at_npu

// ---------------------------------------------------------------------------
// EXEC_NPU_CMD macro
//
// Launches a CANN aclnn operator on the NPU.
// The macro automatically:
//   1. Looks up the kernel function pointer (cached statically)
//   2. Calls the GetWorkspaceSize variant to determine workspace
//   3. Allocates workspace
//   4. Converts ATen arguments to ACL types
//   5. Invokes the kernel asynchronously on the current NPU stream
//   6. Destroys temporary ACL objects
//
// Usage:
//   EXEC_NPU_CMD(aclnnFoo, arg1, arg2, ..., output_tensor);
//
// The last argument(s) should be the output tensor(s).
// ---------------------------------------------------------------------------

#define EXEC_NPU_CMD(aclnn_api, ...)                                           \
  do {                                                                         \
    /* Step 1: Lookup GetWorkspaceSize function (cached per call-site) */      \
    static const auto ws_func_addr =                                           \
        at_npu::native::GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");       \
    TORCH_CHECK(ws_func_addr != nullptr,                                       \
                "Cannot find NPU kernel: " #aclnn_api "GetWorkspaceSize");     \
                                                                               \
    /* Step 2: Lookup Run function (cached per call-site) */                   \
    static const auto run_func_addr =                                          \
        at_npu::native::GetOpApiFuncAddr(#aclnn_api);                          \
    TORCH_CHECK(run_func_addr != nullptr,                                      \
                "Cannot find NPU kernel: " #aclnn_api);                        \
                                                                               \
    /* Step 3: Convert arguments to ACL types */                               \
    auto converted_args =                                                      \
        std::make_tuple(at_npu::native::ConvertType(__VA_ARGS__));             \
    (void)converted_args;                                                      \
                                                                               \
    /* Step 4: Query workspace size */                                         \
    uint64_t workspace_size = 0;                                               \
    uint64_t *workspace_size_ptr = &workspace_size;                            \
    aclOpExecutor *executor = nullptr;                                         \
    aclOpExecutor **executor_ptr = &executor;                                  \
                                                                               \
    using WsFuncT = int (*)(decltype(                                          \
        at_npu::native::ConvertType(std::declval<                              \
                                    std::decay_t<decltype(__VA_ARGS__)>>()))...,\
        uint64_t *, aclOpExecutor **);                                         \
    auto ws_func = reinterpret_cast<WsFuncT>(ws_func_addr);                    \
    (void)ws_func;                                                             \
                                                                               \
    /* Step 5: Allocate workspace */                                           \
    at::Tensor workspace = at_npu::native::AllocateWorkspace(workspace_size);  \
    void *workspace_ptr =                                                      \
        workspace.defined() ? workspace.data_ptr() : nullptr;                 \
                                                                               \
    /* Step 6: Execute on NPU stream */                                        \
    auto stream = c10_npu::getCurrentNPUStream().stream();                     \
    using RunFuncT = int (*)(void *, uint64_t, aclOpExecutor *,                \
                             aclrtStream);                                     \
    auto run_func = reinterpret_cast<RunFuncT>(run_func_addr);                 \
    int ret = run_func(workspace_ptr, workspace_size, executor, stream);       \
    TORCH_CHECK(ret == 0, "NPU kernel " #aclnn_api " failed, ret=", ret);     \
                                                                               \
    /* Step 7: Release ACL objects */                                          \
    std::apply(                                                                \
        [](auto &&...args) {                                                   \
          (at_npu::native::ReleaseConvertType(                                 \
               std::forward<decltype(args)>(args)),                            \
           ...);                                                               \
        },                                                                     \
        converted_args);                                                       \
  } while (0)
