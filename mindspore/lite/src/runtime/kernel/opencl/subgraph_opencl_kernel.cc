/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include <set>
#include "src/runtime/opencl/opencl_executor.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "include/errorcode.h"
#include "src/common/utils.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
SubGraphOpenCLKernel::~SubGraphOpenCLKernel() { UnInit(); }

void SubGraphOpenCLKernel::ReplaceOutTensorAndKernelToNull(
  const std::vector<lite::Tensor *> &in_tensors, const std::vector<std::vector<kernel::LiteKernel *>> &in_kernels,
  OpenCLMemType mem_type) {
  for (size_t i = 0; i < in_tensors.size(); ++i) {
    for (auto &jv : in_kernels.at(i)) {
      MS_ASSERT(jv);
      auto tensors = (mem_type == OpenCLMemType::IMG) ? jv->in_tensors() : jv->out_tensors();
      auto ft = std::find_if(tensors.begin(), tensors.end(),
                             [&in_tensors, &i](lite::Tensor *kv) { return kv == in_tensors.at(i); });
      if (ft != tensors.end()) {
        *ft = nullptr;
      }
      auto kernels = (mem_type == OpenCLMemType::IMG) ? jv->in_kernels() : jv->out_kernels();
      std::replace_if(
        kernels.begin(), kernels.end(),
        [this, &in_tensors, &i](kernel::LiteKernel *kv) {
          MS_ASSERT(kv);
          return std::find_if(kv->in_tensors().begin(), kv->in_tensors().end(),
                              [&in_tensors, &i](lite::Tensor *xv) { return xv == in_tensors.at(i); }) !=
                   kv->in_tensors().end() &&
                 this->nodes_set_.count(kv) == 0;
        },
        nullptr);
      if (mem_type == OpenCLMemType::IMG) {
        jv->set_in_tensors(tensors);
        jv->set_in_kernels(kernels);
      } else {
        jv->set_out_tensors(tensors);
        jv->set_out_kernels(kernels);
      }
    }
  }
}

void SubGraphOpenCLKernel::ReplaceOutTensorAndKernelToConvert(const lite::Tensor *in_tensor,
                                                              const std::vector<kernel::LiteKernel *> &in_kernels,
                                                              lite::Tensor *new_tensor,
                                                              kernel::LiteKernel *in_convert_op,
                                                              OpenCLMemType mem_type) {
  auto in_opencl_op = reinterpret_cast<OpenCLKernel *>(in_convert_op);
  MS_ASSERT(in_opencl_op);
  for (auto &iv : in_kernels) {
    MS_ASSERT(iv);
    auto kernels = (mem_type == OpenCLMemType::IMG) ? iv->in_kernels() : iv->out_kernels();
    auto fk = std::find_if(kernels.begin(), kernels.end(), [&](kernel::LiteKernel *kv) { return kv == nullptr; });
    if (fk != kernels.end()) {
      *fk = in_convert_op;
    } else {
      kernels.emplace_back(in_convert_op);
    }
    auto tensors = (mem_type == OpenCLMemType::IMG) ? iv->in_tensors() : iv->out_tensors();
    auto ft = std::find_if(tensors.begin(), tensors.end(), [&](lite::Tensor *kv) { return kv == nullptr; });
    if (ft != tensors.end()) {
      *ft = new_tensor;
    } else {
      tensors.emplace_back(new_tensor);
    }
    if (mem_type == OpenCLMemType::IMG) {
      iv->set_in_kernels(kernels);
      iv->set_in_tensors(tensors);
      in_opencl_op->AddOutKernel(iv);
    } else {
      iv->set_out_kernels(kernels);
      iv->set_out_tensors(tensors);
      in_convert_op->AddInKernel(iv);
    }
  }
}

int SubGraphOpenCLKernel::GenToFormatOp(const std::vector<lite::Tensor *> &in_tensors,
                                        const std::vector<std::vector<kernel::LiteKernel *>> &in_kernels,
                                        std::vector<lite::Tensor *> *out_tensors,
                                        std::vector<OpenCLToFormatParameter *> *out_parameters,
                                        std::vector<LiteKernel *> *out_convert_ops, OpenCLMemType mem_type) {
  MS_ASSERT(out_tensors);
  MS_ASSERT(out_parameters);
  MS_ASSERT(out_convert_ops);
  out_tensors->clear();
  out_parameters->clear();
  out_convert_ops->clear();
  MS_ASSERT(in_tensors.size() == to_kernels.size());
  MS_ASSERT(in_tensors.size() == from_kernels.size());
  std::vector<std::vector<kernel::LiteKernel *>> loop_kernels;
  if (mem_type == OpenCLMemType::BUF) {
    GetKernelFromToTensor(in_tensors, nodes_, &loop_kernels, true);
  }

  ReplaceOutTensorAndKernelToNull(in_tensors, in_kernels, mem_type);

  for (size_t i = 0; i < in_tensors.size(); ++i) {
    auto dst_format = (mem_type == OpenCLMemType::IMG) ? schema::Format::Format_NHWC4 : schema::Format::Format_NHWC;
    auto src_format = (mem_type == OpenCLMemType::IMG) ? schema::Format::Format_NHWC : schema::Format::Format_NHWC4;
    auto *new_tensor = new (std::nothrow) lite::Tensor();
    MS_ASSERT(new_tensor);
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "SubGraphOpenCLKernel new tensor failed!";
      return RET_ERROR;
    }
    new_tensor->CopyTensor(*in_tensors[i]);
    if (mem_type == OpenCLMemType::IMG) {
      new_tensor->set_format(dst_format);
      in_tensors[i]->set_format(src_format);
    } else {
      new_tensor->set_format(src_format);
      in_tensors[i]->set_format(dst_format);
    }

    out_tensors->emplace_back(new_tensor);
    KernelKey desc{kGPU, kNumberTypeFloat32, schema::PrimitiveType_ToFormat};
    if (mem_type == OpenCLMemType::IMG && ocl_runtime_->GetFp16Enable()) {
      desc.data_type = kNumberTypeFloat16;
      new_tensor->set_data_type(kNumberTypeFloat16);
    }
    auto *parameter = static_cast<OpenCLToFormatParameter *>(malloc(sizeof(OpenCLToFormatParameter)));
    MS_ASSERT(parameter);
    if (parameter == nullptr) {
      MS_LOG(ERROR) << "SubGraphOpenCLKernel new parameter failed!";
      delete new_tensor;
      new_tensor = nullptr;
      return RET_ERROR;
    }
    parameter->op_parameter.type_ = mindspore::schema::PrimitiveType_ToFormat;
    parameter->src_format = src_format;
    parameter->dst_format = dst_format;
    parameter->out_mem_type = mem_type;
    out_parameters->emplace_back(parameter);
    LiteKernel *in_convert_op = nullptr;
    if (mem_type == OpenCLMemType::IMG) {
      in_convert_op = lite::GetOpenCLKernel({in_tensors[i]}, {new_tensor}, reinterpret_cast<OpParameter *>(parameter),
                                            context_, desc);
    } else {
      in_convert_op = lite::GetOpenCLKernel({new_tensor}, {in_tensors[i]}, reinterpret_cast<OpParameter *>(parameter),
                                            context_, desc);
    }
    MS_ASSERT(in_convert_op);
    if (in_convert_op == nullptr) {
      MS_LOG(ERROR) << "SubGraphOpenCLKernel create op failed!";
      delete new_tensor;
      new_tensor = nullptr;
      free(parameter);
      parameter = nullptr;
      return RET_ERROR;
    }

    ReplaceOutTensorAndKernelToConvert(in_tensors.at(i), in_kernels.at(i), new_tensor, in_convert_op, mem_type);

    // replace in_tensor of inner kernel which use out tensor
    if (mem_type == OpenCLMemType::BUF) {
      for (auto &iv : loop_kernels[i]) {
        MS_ASSERT(iv);
        auto tensors = iv->in_tensors();
        auto jv = std::find(tensors.begin(), tensors.end(), in_tensors.at(i));
        if (jv != tensors.end()) {
          *jv = new_tensor;
          iv->set_in_tensors(tensors);
        }
      }
    }

    out_convert_ops->emplace_back(in_convert_op);
  }
  return RET_OK;
}

int SubGraphOpenCLKernel::Init() {
  allocator_ = ocl_runtime_->GetAllocator();
  MS_LOG(DEBUG) << "input num=" << in_tensors_.size() << ", output num=" << out_tensors_.size();
  for (const auto tensor : in_tensors_) {
    MS_ASSERT(tensor);
    tensor->set_allocator(allocator_);
  }
  for (const auto tensor : out_tensors_) {
    MS_ASSERT(tensor);
    tensor->set_allocator(allocator_);
  }

  GetInOutNodes();

  std::vector<std::vector<kernel::LiteKernel *>> from_kernels_;
  GetKernelFromToTensor(in_tensors_, in_nodes_, &from_kernels_, true);
  int ret = GenToFormatOp(in_tensors_, from_kernels_, &in_convert_tensors_, &in_parameters_, &in_convert_ops_,
                          OpenCLMemType::IMG);
  if (ret != RET_OK) {
    return ret;
  }
  nodes_.insert(nodes_.begin(), in_convert_ops_.begin(), in_convert_ops_.end());

  std::vector<std::vector<kernel::LiteKernel *>> to_kernels_;
  GetKernelFromToTensor(out_tensors_, out_nodes_, &to_kernels_, false);
  ret = GenToFormatOp(out_tensors_, to_kernels_, &out_convert_tensors_, &out_parameters_, &out_convert_ops_,
                      OpenCLMemType::BUF);
  if (ret != RET_OK) {
    return ret;
  }
  nodes_.insert(nodes_.end(), out_convert_ops_.begin(), out_convert_ops_.end());

  UpdateTensorDataType();

  MallocTensorWithReuse();

  return RET_OK;
}

void SubGraphOpenCLKernel::UpdateTensorDataType() {
  bool is_fp16 = ocl_runtime_->GetFp16Enable();
  MS_ASSERT(in_tensors_[0]);
  if (is_fp16 && (in_tensors_[0]->data_type() == kNumberTypeFloat32)) {
    std::set<lite::Tensor *> out_set;
    out_set.insert(in_tensors_.begin(), in_tensors_.end());
    out_set.insert(out_tensors_.begin(), out_tensors_.end());
    for (auto iv : nodes_) {
      MS_ASSERT(iv);
      auto cur_outs = iv->out_tensors();
      for (auto jv : cur_outs) {
        if (out_set.count(jv) == 0) {
          MS_ASSERT(jv);
          jv->set_data_type(kNumberTypeFloat16);
        }
      }
    }
  }
}

void SubGraphOpenCLKernel::MallocTensorWithReuse() {
  int ret;
  kernel::LiteKernelUtil::InitTensorRefCount(nodes_);
  for (auto *kernel : nodes_) {
    MS_ASSERT(kernel);
    auto *op_kernel = reinterpret_cast<kernel::OpenCLKernel *>(kernel);
    auto outputs = kernel->out_tensors();
    for (auto i = 0; i < outputs.size(); ++i) {
      auto *output = outputs.at(i);
      MS_ASSERT(output);
      if (op_kernel->GetMemType() == OpenCLMemType::IMG) {
        std::vector<size_t> img_size;
        ret = op_kernel->GetImageSize(i, &img_size);
        if (ret != RET_OK) {
          MS_LOG(WARNING) << "GetImageSize failed";
        }
        auto data_ptr = allocator_->Malloc(output->Size(), img_size);
        output->set_data(data_ptr);
      } else {
        ret = output->MallocData(allocator_);
        if (ret != RET_OK) {
          MS_LOG(WARNING) << "MallocData failed";
        }
      }
      output->set_allocator(allocator_);
    }
    for (auto input_kernel : kernel->in_kernels()) {
      MS_ASSERT(input_kernel);
      ret = input_kernel->DecOutTensorRefCount();
      if (ret != RET_OK) {
        MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << kernel->name() << " failed";
      }
    }
  }
  for (auto kernel : out_kernels_) {
    MS_ASSERT(kernel);
    ret = kernel->DecOutTensorRefCount();
    if (ret != RET_OK) {
      MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << kernel->name() << " failed";
    }
  }
}

void SubGraphOpenCLKernel::GetKernelFromToTensor(const std::vector<lite::Tensor *> &in_tensors,
                                                 const std::vector<kernel::LiteKernel *> &in_kernels,
                                                 std::vector<std::vector<kernel::LiteKernel *>> *out_kernels,
                                                 bool is_from) {
  std::vector<std::set<lite::Tensor *>> ksets;
  for (auto jv : in_kernels) {
    MS_ASSERT(jv);
    auto tens = is_from ? jv->in_tensors() : jv->out_tensors();
    std::set<lite::Tensor *> kset;
    kset.insert(tens.begin(), tens.end());
    ksets.emplace_back(kset);
  }
  MS_ASSERT(out_kernels);
  for (auto in_tensor : in_tensors) {
    std::vector<kernel::LiteKernel *> kvec;
    for (size_t j = 0; j < in_kernels.size(); ++j) {
      if (ksets[j].count(in_tensor)) {
        kvec.emplace_back(in_kernels[j]);
      }
    }
    out_kernels->emplace_back(kvec);
  }
}

void SubGraphOpenCLKernel::GetInOutNodes() {
  std::vector<std::set<lite::Tensor *>> ksets_in;
  std::vector<std::set<lite::Tensor *>> ksets_out;
  for (auto jv : nodes_) {
    MS_ASSERT(jv);
    std::set<lite::Tensor *> kset;
    kset.insert(jv->in_tensors().begin(), jv->in_tensors().end());
    ksets_in.emplace_back(kset);

    kset.clear();
    kset.insert(jv->out_tensors().begin(), jv->out_tensors().end());
    ksets_out.emplace_back(kset);
  }
  for (size_t j = 0; j < nodes_.size(); ++j) {
    if (std::find_if(in_tensors_.begin(), in_tensors_.end(),
                     [&ksets_in, &j](lite::Tensor *val) { return ksets_in[j].count(val); }) != in_tensors_.end()) {
      in_nodes_.emplace_back(nodes_.at(j));
    }
    if (std::find_if(out_tensors_.begin(), out_tensors_.end(),
                     [&ksets_out, &j](lite::Tensor *val) { return ksets_out[j].count(val); }) != out_tensors_.end()) {
      out_nodes_.emplace_back(nodes_.at(j));
    }
  }
}

int SubGraphOpenCLKernel::Prepare() {
  auto ret = Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "OpenCL subgraph init fail";
    return ret;
  }
  ret = SubGraphKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "OpenCL prepare fail";
    return ret;
  }
  return RET_OK;
}

void SubGraphOpenCLKernel::UnInit() {
  for (const auto &tensor : in_convert_tensors_) {
    delete tensor;
  }
  in_convert_tensors_.clear();
  for (const auto &tensor : out_convert_tensors_) {
    delete tensor;
  }
  out_convert_tensors_.clear();
  for (const auto &op : nodes_) {
    delete op;
  }
  nodes_.clear();
  in_convert_ops_.clear();
  out_convert_ops_.clear();
  delete this->executor_;
}

int SubGraphOpenCLKernel::InferShape() { return RET_OK; }

int SubGraphOpenCLKernel::ReSize() { return RET_OK; }

int SubGraphOpenCLKernel::Run() {
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "executor is nullptr";
    return RET_ERROR;
  }
  int ret;
  for (auto &tensor : in_tensors_) {
    MS_ASSERT(tensor);
    if (tensor->data_c() == nullptr) {
      MS_LOG(ERROR) << "OpenCL subgraph input tensor data is null";
      return RET_ERROR;
    }
    ret = allocator_->UnmapBuffer(tensor->data_c());
    if (ret != RET_OK) {
      return ret;
    }
  }

  ret = executor_->Run(in_tensors_, out_tensors_, nodes_, allocator_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run opencl executor failed: " << ret;
    return ret;
  }
  if (!ocl_runtime_->SyncCommandQueue()) {
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel