// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/tensor_meta.h"
#include "paddle/pten/hapi/all.h"

#include "paddle/fluid/eager/function_api.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"

namespace egr {

template <typename T>
bool CompareGradTensorWithValue(const egr::EagerTensor& target, T value) {
  egr::AutogradMeta* meta = egr::EagerUtils::unsafe_autograd_meta(target);
  auto grad_dense =
      std::dynamic_pointer_cast<pten::DenseTensor>(meta->Grad().impl());
  T* ptr = grad_dense->mutable_data<T>();

  std::vector<T> host_data(grad_dense->numel());
  if (grad_dense->backend() == pten::Backend::CUDA) {
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx = dynamic_cast<paddle::platform::CUDADeviceContext*>(
        pool.Get(paddle::platform::CUDAPlace()));
    auto stream = dev_ctx->stream();

    paddle::memory::Copy(paddle::platform::CPUPlace(), host_data.data(),
                         paddle::platform::CUDAPlace(), ptr,
                         sizeof(T) * grad_dense->numel(), stream);
    ptr = host_data.data();
  }

  PADDLE_ENFORCE(grad_dense->numel() != 0,
                 paddle::platform::errors::Fatal("Grad tensor is empty"));
  for (int i = 0; i < grad_dense->numel(); i++) {
    PADDLE_ENFORCE(value == ptr[i],
                   paddle::platform::errors::Fatal(
                       "Numerical Error, Expected %f, got %f", value, ptr[i]));
  }
  return true;
}

template <typename T>
bool CompareTensorWithValue(const egr::EagerTensor& target, T value) {
  // TODO(jiabin): Support Selected Rows later
  auto dense_t = std::dynamic_pointer_cast<pten::DenseTensor>(target.impl());
  T* ptr = dense_t->mutable_data<T>();

  std::vector<T> host_data(dense_t->numel());
  if (dense_t->backend() == pten::Backend::CUDA) {
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx = dynamic_cast<paddle::platform::CUDADeviceContext*>(
        pool.Get(paddle::platform::CUDAPlace()));
    auto stream = dev_ctx->stream();

    paddle::memory::Copy(paddle::platform::CPUPlace(), host_data.data(),
                         paddle::platform::CUDAPlace(), ptr,
                         sizeof(T) * dense_t->numel(), stream);
    ptr = host_data.data();
  }

  PADDLE_ENFORCE(dense_t->numel() != 0,
                 paddle::platform::errors::Fatal("Tensor is empty"));
  for (int i = 0; i < dense_t->numel(); i++) {
    PADDLE_ENFORCE(value == ptr[i],
                   paddle::platform::errors::Fatal(
                       "Numerical Error, Expected %f, got %f", value, ptr[i]));
  }
  return true;
}

template <typename T>
bool CompareVariableWithValue(const egr::EagerTensor& target, T value) {
  // TODO(jiabin): Support Selected Rows later
  auto lod_tensor = target.Var().Get<paddle::framework::LoDTensor>();
  T* ptr = lod_tensor.data<T>();

  std::vector<T> host_data(lod_tensor.numel());
  if (paddle::platform::is_gpu_place(lod_tensor.place())) {
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx = dynamic_cast<paddle::platform::CUDADeviceContext*>(
        pool.Get(paddle::platform::CUDAPlace()));
    auto stream = dev_ctx->stream();

    paddle::memory::Copy(paddle::platform::CPUPlace(), host_data.data(),
                         paddle::platform::CUDAPlace(), ptr,
                         sizeof(T) * lod_tensor.numel(), stream);
    ptr = host_data.data();
  }

  for (int i = 0; i < lod_tensor.numel(); i++) {
    PADDLE_ENFORCE(value == ptr[i],
                   paddle::platform::errors::Fatal(
                       "Numerical Error, Expected %f, got %f", value, ptr[i]));
  }
  return true;
}

template <typename T>
bool CompareGradVariableWithValue(const egr::EagerTensor& target, T value) {
  // TODO(jiabin): Support Selected Rows later
  egr::AutogradMeta* meta = egr::EagerUtils::unsafe_autograd_meta(target);
  auto lod_tensor = meta->Grad().Var().Get<paddle::framework::LoDTensor>();
  T* ptr = lod_tensor.data<T>();

  std::vector<T> host_data(lod_tensor.numel());
  if (paddle::platform::is_gpu_place(lod_tensor.place())) {
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx = dynamic_cast<paddle::platform::CUDADeviceContext*>(
        pool.Get(paddle::platform::CUDAPlace()));
    auto stream = dev_ctx->stream();

    paddle::memory::Copy(paddle::platform::CPUPlace(), host_data.data(),
                         paddle::platform::CUDAPlace(), ptr,
                         sizeof(T) * lod_tensor.numel(), stream);
    ptr = host_data.data();
  }

  for (int i = 0; i < lod_tensor.numel(); i++) {
    PADDLE_ENFORCE(value == ptr[i],
                   paddle::platform::errors::Fatal(
                       "Numerical Error, Expected %f, got %f", value, ptr[i]));
  }
  return true;
}

inline void InitEnv(paddle::platform::Place place) {
  // Prepare Device Contexts
  // Init DeviceContextPool
  paddle::framework::InitDevices();

  // Init Tracer Place
  Controller::Instance().SetExpectedPlace(place);
}
}  // namespace egr
