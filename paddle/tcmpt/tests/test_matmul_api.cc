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

#include "paddle/tcmpt/hapi/include/linalg.h"

#include "paddle/tcmpt/core/dense_tensor.h"
#include "paddle/tcmpt/core/kernel_registry.h"

PT_DECLARE_MODULE(LinalgCPU);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_MODULE(LinalgCUDA);
#endif

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(API, matmul) {
  // 1. create tensor
  auto dense_x = std::make_shared<pt::DenseTensor>(
      pt::TensorMeta(framework::make_ddim({3, 3}),
                     pt::Backend::kCPU,
                     pt::DataType::kFLOAT32,
                     pt::DataLayout::kNCHW),
      pt::TensorStatus());
  auto* dense_x_data = dense_x->mutable_data<float>();

  auto dense_y = std::make_shared<pt::DenseTensor>(
      pt::TensorMeta(framework::make_ddim({3, 3}),
                     pt::Backend::kCPU,
                     pt::DataType::kFLOAT32,
                     pt::DataLayout::kNCHW),
      pt::TensorStatus());
  auto* dense_y_data = dense_y->mutable_data<float>();

  for (size_t i = 0; i < 12; ++i) {
    dense_x_data[i] = 1.0;
    dense_y_data[i] = 2.0;
  }
  std::vector<float> sum(9, 6.0);

  paddle::experimental::Tensor x(dense_x);
  paddle::experimental::Tensor y(dense_y);

  // 2. test API
  auto out = paddle::experimental::matmul(x, y, false, false);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 2);
  ASSERT_EQ(out.shape()[0], 3);
  ASSERT_EQ(out.shape()[1], 3);
  ASSERT_EQ(out.numel(), 9);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pt::DataType::kFLOAT32);
  ASSERT_EQ(out.layout(), pt::DataLayout::kNCHW);
  ASSERT_EQ(out.initialized(), true);

  auto dense_out = std::dynamic_pointer_cast<pt::DenseTensor>(out.impl());

  for (size_t i = 0; i < 9; i++) {
    ASSERT_NEAR(sum[i], dense_out->data<float>()[i], 1e-6f);
  }
}
