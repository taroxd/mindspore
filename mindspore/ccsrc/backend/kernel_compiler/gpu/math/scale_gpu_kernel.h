#ifndef MINDSPORE_SCALE_GPU_KERNEL_H
#define MINDSPORE_SCALE_GPU_KERNEL_H

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class ScaleGpuKernel : public GpuKernel {
  public:
    ScaleGpuKernel()
      : input_size_(sizeof(T)),
        output_size_(sizeof(T)) {}
    ~ScaleGpuKernel() override = default;

    const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
    const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
    const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *alpha_addr = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(output_addr, input_addr, input_size_,
      cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync failed in ScaleGpuKernel::Launch.");

    cublasPointerMode_t orig_ptr_mode;
    CHECK_CUBLAS_RET_WITH_EXCEPT(cublasGetPointerMode(blas_handle_, &orig_ptr_mode), "cublasGetPointerMode failed");
    CHECK_CUBLAS_RET_WITH_EXCEPT(cublasSetPointerMode(blas_handle_, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode failed");

    CHECK_CUBLAS_RET_WITH_EXCEPT(cublasScalEx(blas_handle_, input_size_ / sizeof(T),
      alpha_addr, dtype_alpha_, output_addr, dtype_x_, 1, CUDA_R_32F),
      "cublasScalEx Failed");

    CHECK_CUBLAS_RET_WITH_EXCEPT(cublasSetPointerMode(blas_handle_, orig_ptr_mode), "cublasSetPointerMode restore failed");

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    blas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();

    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but scale needs 2 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but scale needs 1 output.";
      return false;
    }

    dtype_x_ = GetCudaDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    dtype_alpha_ = GetCudaDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 1)));

    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }

    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "input is null";
      InitSizeLists();
      return true;
    }

    auto alpha_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    if (alpha_shape.size() > 1 || (alpha_shape.size() == 1 && alpha_shape.front() != 1)) {
      MS_LOG(WARNING) << "Alpha shape is not [1]. Only the first value is used.";
    }

    output_size_ = input_size_;
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(sizeof(T));
    output_size_list_.push_back(output_size_);
  }

 private:
  size_t input_size_;
  size_t output_size_;
  bool is_null_input_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  cublasHandle_t blas_handle_;
  cudaDataType_t dtype_x_;
  cudaDataType_t dtype_alpha_;
};

}  // namespace kernel
}  // namespace mindspore

#endif
