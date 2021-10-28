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

#include "paddle/pten/hapi/include/reduce.h"

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/cuda/utils.h"

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(API, reduce_sum_cpu) {
  // 1. create tensor
  auto dense_x = std::make_shared<pten::DenseTensor>(
      pten::TensorMeta(framework::make_ddim({10, 10}),
                       pten::Backend::CPU,
                       pten::DataType::FLOAT32,
                       pten::DataLayout::NCHW),
      pten::TensorStatus());
  auto* dense_x_data = dense_x->mutable_data<float>();

  float result = 0.0;
  for (size_t i = 0; i < 100; ++i) {
    dense_x_data[i] = static_cast<float>(i);
    result += dense_x_data[i];
  }
  paddle::experimental::Tensor x(dense_x);

  // 2. test API
  auto out = paddle::experimental::reduce_sum(x,
                                              true,   // reduce_all
                                              {0},    // dim
                                              false,  // keep_dim
                                              -1);    // out_dtype

  // 3. check result
  ASSERT_EQ(out.shape().size(), 1);
  ASSERT_EQ(out.shape()[0], 1);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pten::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<pten::DenseTensor>(out.impl());

  ASSERT_NEAR(result, dense_out->data<float>()[0], 1e-6f);
}

TEST(API, reduce_sum_cuda) {
  // Prepare CPU Dense Tensor
  pten::TensorMeta ref_x_meta = pten::TensorMeta(framework::make_ddim({10, 10}),
                                                 pten::Backend::CPU,
                                                 pten::DataType::FLOAT32,
                                                 pten::DataLayout::NCHW);

  auto ref_x =
      std::make_shared<pten::DenseTensor>(ref_x_meta, pten::TensorStatus());
  auto* ref_x_data = ref_x->mutable_data<float>();

  float result = 0.0;
  for (size_t i = 0; i < 100; ++i) {
    ref_x_data[i] = static_cast<float>(i);
    result += ref_x_data[i];
  }

  // 1. create tensor
  auto dense_x = std::make_shared<pten::DenseTensor>(
      pten::TensorMeta(framework::make_ddim({10, 10}),
                       pten::Backend::CUDA,
                       pten::DataType::FLOAT32,
                       pten::DataLayout::NCHW),
      pten::TensorStatus());

  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto place = paddle::platform::CUDAPlace();
  auto* dev_ctx = pool.GetByPlace(place);

  pten::Copy(*dev_ctx, *ref_x.get(), dense_x.get());

  paddle::experimental::Tensor x(dense_x);

  // 2. test API
  auto out = paddle::experimental::reduce_sum(x,
                                              true,   // reduce_all
                                              {0},    // dim
                                              false,  // keep_dim
                                              -1);    // out_dtype

  // 3. check result
  ASSERT_EQ(out.shape().size(), 1);
  ASSERT_EQ(out.shape()[0], 1);
  ASSERT_EQ(out.type(), pten::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<pten::DenseTensor>(out.impl());

  pten::TensorMeta out_meta = pten::TensorMeta(out.shape(),
                                               pten::Backend::CPU,
                                               pten::DataType::FLOAT32,
                                               pten::DataLayout::NCHW);
  auto ref_out =
      std::make_shared<pten::DenseTensor>(out_meta, pten::TensorStatus());
  pten::Copy(*dev_ctx, *dense_out.get(), ref_out.get());

  ASSERT_NEAR(result, ref_out->data<float>()[0], 1e-6f);
}
