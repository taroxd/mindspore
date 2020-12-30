
#include "backend/kernel_compiler/gpu/arrays/scaled_gather_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  ScaledGather,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ScaledGatherKernel, float, int)
MS_REG_GPU_KERNEL_TWO(
  ScaledGather,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ScaledGatherKernel, half, int)
}  // namespace kernel
}  // namespace mindspore
