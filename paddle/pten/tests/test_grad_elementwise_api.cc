/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <memory>

#include "paddle/pten/hapi/include/grad_elementwise.h"

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/cuda/utils.h"

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(API, grad_elementwise_add_cpu) {
  // 1. create tensor
  pten::TensorMeta x_meta = pten::TensorMeta(framework::make_ddim({3, 3}),
                                             pten::Backend::CPU,
                                             pten::DataType::FLOAT32,
                                             pten::DataLayout::NCHW);

  auto dense_x =
      std::make_shared<pten::DenseTensor>(x_meta, pten::TensorStatus());
  auto* dense_x_data = dense_x->mutable_data<float>();

  pten::TensorMeta y_meta = pten::TensorMeta(framework::make_ddim({3, 3}),
                                             pten::Backend::CPU,
                                             pten::DataType::FLOAT32,
                                             pten::DataLayout::NCHW);

  auto dense_y =
      std::make_shared<pten::DenseTensor>(y_meta, pten::TensorStatus());
  auto* dense_y_data = dense_y->mutable_data<float>();

  pten::TensorMeta grad_out_meta =
      pten::TensorMeta(framework::make_ddim({3, 3}),
                       pten::Backend::CPU,
                       pten::DataType::FLOAT32,
                       pten::DataLayout::NCHW);

  auto dense_grad_out =
      std::make_shared<pten::DenseTensor>(grad_out_meta, pten::TensorStatus());
  auto* dense_grad_out_data = dense_grad_out->mutable_data<float>();

  for (size_t i = 0; i < 9; ++i) {
    dense_x_data[i] = 1.0;
    dense_y_data[i] = 2.0;
    dense_grad_out_data[i] = 3.0;
  }
  std::vector<float> grad_x_result(9, 3.0);
  std::vector<float> grad_y_result(9, 3.0);

  paddle::experimental::Tensor x(dense_x);
  paddle::experimental::Tensor y(dense_y);
  paddle::experimental::Tensor grad_out(dense_grad_out);

  // 2. test API
  std::vector<paddle::experimental::Tensor> outs =
      paddle::experimental::grad_elementwise_add(x, y, grad_out, -1);

  paddle::experimental::Tensor& grad_x = outs[0];
  paddle::experimental::Tensor& grad_y = outs[1];

  // 3. check result
  ASSERT_EQ(grad_x.shape().size(), 2);
  ASSERT_EQ(grad_x.shape()[0], 3);
  ASSERT_EQ(grad_x.shape()[1], 3);
  ASSERT_EQ(grad_x.numel(), 9);
  ASSERT_EQ(grad_x.type(), pten::DataType::FLOAT32);
  ASSERT_EQ(grad_x.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(grad_x.initialized(), true);

  auto dense_grad_x =
      std::dynamic_pointer_cast<pten::DenseTensor>(grad_x.impl());
  for (size_t i = 0; i < 9; i++) {
    ASSERT_NEAR(grad_x_result[i], dense_grad_x->data<float>()[i], 1e-6f);
  }

  ASSERT_EQ(grad_y.shape().size(), 2);
  ASSERT_EQ(grad_y.shape()[0], 3);
  ASSERT_EQ(grad_y.shape()[1], 3);
  ASSERT_EQ(grad_y.numel(), 9);
  ASSERT_EQ(grad_y.type(), pten::DataType::FLOAT32);
  ASSERT_EQ(grad_y.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(grad_y.initialized(), true);

  auto dense_grad_y =
      std::dynamic_pointer_cast<pten::DenseTensor>(grad_y.impl());
  for (size_t i = 0; i < 9; i++) {
    ASSERT_NEAR(grad_y_result[i], dense_grad_y->data<float>()[i], 1e-6f);
  }
}

TEST(API, grad_elementwise_add_cuda) {
  pten::TensorMeta ref_x_meta = pten::TensorMeta(framework::make_ddim({3, 3}),
                                                 pten::Backend::CPU,
                                                 pten::DataType::FLOAT32,
                                                 pten::DataLayout::NCHW);

  auto dense_ref_x =
      std::make_shared<pten::DenseTensor>(ref_x_meta, pten::TensorStatus());
  auto* dense_ref_x_data = dense_ref_x->mutable_data<float>();

  pten::TensorMeta ref_y_meta = pten::TensorMeta(framework::make_ddim({3, 3}),
                                                 pten::Backend::CPU,
                                                 pten::DataType::FLOAT32,
                                                 pten::DataLayout::NCHW);

  auto dense_ref_y =
      std::make_shared<pten::DenseTensor>(ref_y_meta, pten::TensorStatus());
  auto* dense_ref_y_data = dense_ref_y->mutable_data<float>();

  pten::TensorMeta ref_grad_out_meta =
      pten::TensorMeta(framework::make_ddim({3, 3}),
                       pten::Backend::CPU,
                       pten::DataType::FLOAT32,
                       pten::DataLayout::NCHW);

  auto dense_ref_grad_out = std::make_shared<pten::DenseTensor>(
      ref_grad_out_meta, pten::TensorStatus());
  auto* dense_ref_grad_out_data = dense_ref_grad_out->mutable_data<float>();

  for (size_t i = 0; i < 9; ++i) {
    dense_ref_x_data[i] = 1.0;
    dense_ref_y_data[i] = 2.0;
    dense_ref_grad_out_data[i] = 3.0;
  }
  std::vector<float> grad_x_result(9, 3.0);
  std::vector<float> grad_y_result(9, 3.0);

  // 1. create tensor
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto place = paddle::platform::CUDAPlace();
  auto* dev_ctx = pool.GetByPlace(place);

  pten::TensorMeta x_meta = pten::TensorMeta(framework::make_ddim({3, 3}),
                                             pten::Backend::CUDA,
                                             pten::DataType::FLOAT32,
                                             pten::DataLayout::NCHW);

  auto dense_x =
      std::make_shared<pten::DenseTensor>(x_meta, pten::TensorStatus());

  pten::TensorMeta y_meta = pten::TensorMeta(framework::make_ddim({3, 3}),
                                             pten::Backend::CUDA,
                                             pten::DataType::FLOAT32,
                                             pten::DataLayout::NCHW);

  auto dense_y =
      std::make_shared<pten::DenseTensor>(y_meta, pten::TensorStatus());

  pten::TensorMeta grad_out_meta =
      pten::TensorMeta(framework::make_ddim({3, 3}),
                       pten::Backend::CUDA,
                       pten::DataType::FLOAT32,
                       pten::DataLayout::NCHW);

  auto dense_grad_out =
      std::make_shared<pten::DenseTensor>(grad_out_meta, pten::TensorStatus());

  pten::Copy(*dev_ctx, *dense_ref_x.get(), dense_x.get());
  pten::Copy(*dev_ctx, *dense_ref_y.get(), dense_y.get());
  pten::Copy(*dev_ctx, *dense_ref_grad_out.get(), dense_grad_out.get());

  paddle::experimental::Tensor x(dense_x);
  paddle::experimental::Tensor y(dense_y);
  paddle::experimental::Tensor grad_out(dense_grad_out);

  // 2. test API
  std::vector<paddle::experimental::Tensor> outs =
      paddle::experimental::grad_elementwise_add(x, y, grad_out, -1);

  paddle::experimental::Tensor& grad_x = outs[0];
  paddle::experimental::Tensor& grad_y = outs[1];

  // 3. check result
  ASSERT_EQ(grad_x.shape().size(), 2);
  ASSERT_EQ(grad_x.shape()[0], 3);
  ASSERT_EQ(grad_x.shape()[1], 3);
  ASSERT_EQ(grad_x.numel(), 9);
  ASSERT_EQ(grad_x.type(), pten::DataType::FLOAT32);
  ASSERT_EQ(grad_x.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(grad_x.initialized(), true);

  auto dense_grad_x =
      std::dynamic_pointer_cast<pten::DenseTensor>(grad_x.impl());
  pten::TensorMeta grad_x_meta = pten::TensorMeta(dense_grad_x->dims(),
                                                  pten::Backend::CPU,
                                                  pten::DataType::FLOAT32,
                                                  pten::DataLayout::NCHW);

  auto ref_grad_x =
      std::make_shared<pten::DenseTensor>(grad_x_meta, pten::TensorStatus());
  pten::Copy(*dev_ctx, *dense_grad_x.get(), ref_grad_x.get());

  for (size_t i = 0; i < 9; i++) {
    ASSERT_NEAR(grad_x_result[i], ref_grad_x->data<float>()[i], 1e-6f);
  }

  ASSERT_EQ(grad_y.shape().size(), 2);
  ASSERT_EQ(grad_y.shape()[0], 3);
  ASSERT_EQ(grad_y.shape()[1], 3);
  ASSERT_EQ(grad_y.numel(), 9);
  ASSERT_EQ(grad_y.type(), pten::DataType::FLOAT32);
  ASSERT_EQ(grad_y.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(grad_y.initialized(), true);

  auto dense_grad_y =
      std::dynamic_pointer_cast<pten::DenseTensor>(grad_y.impl());

  pten::TensorMeta grad_y_meta = pten::TensorMeta(dense_grad_y->dims(),
                                                  pten::Backend::CPU,
                                                  pten::DataType::FLOAT32,
                                                  pten::DataLayout::NCHW);

  auto ref_grad_y =
      std::make_shared<pten::DenseTensor>(grad_y_meta, pten::TensorStatus());
  pten::Copy(*dev_ctx, *dense_grad_y.get(), ref_grad_y.get());
  for (size_t i = 0; i < 9; i++) {
    ASSERT_NEAR(grad_y_result[i], ref_grad_y->data<float>()[i], 1e-6f);
  }
}
