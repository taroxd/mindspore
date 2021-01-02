
#include "backend/kernel_compiler/gpu/math/scale_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_ONE(Scale, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      ScaleGpuKernel, float)
MS_REG_GPU_KERNEL_ONE(Scale, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      ScaleGpuKernel, float16)
}  // namespace kernel
}  // namespace mindspore
