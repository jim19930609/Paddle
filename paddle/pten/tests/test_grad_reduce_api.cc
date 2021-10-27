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

#include "paddle/tcmpt/hapi/include/grad_reduce.h"

#include "paddle/tcmpt/core/dense_tensor.h"
#include "paddle/tcmpt/core/kernel_registry.h"
#include "paddle/tcmpt/kernels/cuda/utils.h"

PT_DECLARE_MODULE(GradReduceCPU);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_MODULE(GradReduceCUDA);
#endif

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

TEST(API, grad_reduce_sum_cpu) {
  // 1. create tensor
  pt::TensorMeta x_meta = pt::TensorMeta(framework::make_ddim({3, 3}),
                                         pt::Backend::kCPU,
                                         pt::DataType::kFLOAT32,
                                         pt::DataLayout::kNCHW);

  auto dense_x = std::make_shared<pt::DenseTensor>(x_meta, pt::TensorStatus());
  auto* dense_x_data = dense_x->mutable_data<float>();

  pt::TensorMeta out_meta = pt::TensorMeta(framework::make_ddim({1}),
                                           pt::Backend::kCPU,
                                           pt::DataType::kFLOAT32,
                                           pt::DataLayout::kNCHW);

  auto dense_out =
      std::make_shared<pt::DenseTensor>(out_meta, pt::TensorStatus());
  auto* dense_out_data = dense_out->mutable_data<float>();

  pt::TensorMeta grad_out_meta = pt::TensorMeta(framework::make_ddim({1}),
                                                pt::Backend::kCPU,
                                                pt::DataType::kFLOAT32,
                                                pt::DataLayout::kNCHW);

  auto dense_grad_out =
      std::make_shared<pt::DenseTensor>(grad_out_meta, pt::TensorStatus());
  auto* dense_grad_out_data = dense_grad_out->mutable_data<float>();

  dense_out_data[0] = 9.0;
  dense_grad_out_data[0] = 4.0;
  for (size_t i = 0; i < 9; ++i) {
    dense_x_data[i] = 1.0;
  }

  std::vector<float> grad_x_result(9, 4.0);

  paddle::experimental::Tensor x(dense_x);
  paddle::experimental::Tensor out(dense_out);
  paddle::experimental::Tensor grad_out(dense_grad_out);

  // 2. test API
  paddle::experimental::Tensor grad_x =
      paddle::experimental::grad_reduce_sum(x, out, grad_out, true, {0}, -1);

  // 3. check result
  ASSERT_EQ(grad_x.shape().size(), 2);
  ASSERT_EQ(grad_x.shape()[0], 3);
  ASSERT_EQ(grad_x.shape()[1], 3);
  ASSERT_EQ(grad_x.numel(), 9);
  ASSERT_EQ(grad_x.type(), pt::DataType::kFLOAT32);
  ASSERT_EQ(grad_x.layout(), pt::DataLayout::kNCHW);
  ASSERT_EQ(grad_x.initialized(), true);

  auto dense_grad_x = std::dynamic_pointer_cast<pt::DenseTensor>(grad_x.impl());
  for (size_t i = 0; i < 9; i++) {
    ASSERT_NEAR(grad_x_result[i], dense_grad_x->data<float>()[i], 1e-6f);
  }
}

TEST(API, grad_reduce_sum_cuda) {
  pt::TensorMeta ref_x_meta = pt::TensorMeta(framework::make_ddim({3, 3}),
                                             pt::Backend::kCPU,
                                             pt::DataType::kFLOAT32,
                                             pt::DataLayout::kNCHW);

  auto dense_ref_x =
      std::make_shared<pt::DenseTensor>(ref_x_meta, pt::TensorStatus());
  auto* dense_ref_x_data = dense_ref_x->mutable_data<float>();

  pt::TensorMeta ref_out_meta = pt::TensorMeta(framework::make_ddim({1}),
                                               pt::Backend::kCPU,
                                               pt::DataType::kFLOAT32,
                                               pt::DataLayout::kNCHW);

  auto dense_ref_out =
      std::make_shared<pt::DenseTensor>(ref_out_meta, pt::TensorStatus());
  auto* dense_ref_out_data = dense_ref_out->mutable_data<float>();

  pt::TensorMeta ref_grad_out_meta = pt::TensorMeta(framework::make_ddim({1}),
                                                    pt::Backend::kCPU,
                                                    pt::DataType::kFLOAT32,
                                                    pt::DataLayout::kNCHW);

  auto dense_ref_grad_out =
      std::make_shared<pt::DenseTensor>(ref_grad_out_meta, pt::TensorStatus());
  auto* dense_ref_grad_out_data = dense_ref_grad_out->mutable_data<float>();

  dense_ref_out_data[0] = 9.0;
  dense_ref_grad_out_data[0] = 4.0;
  for (size_t i = 0; i < 9; ++i) {
    dense_ref_x_data[i] = 1.0;
  }
  std::vector<float> grad_x_result(9, 4.0);

  // 1. create tensor
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto place = paddle::platform::CUDAPlace();
  auto* dev_ctx = pool.GetByPlace(place);

  pt::TensorMeta x_meta = pt::TensorMeta(framework::make_ddim({3, 3}),
                                         pt::Backend::kCUDA,
                                         pt::DataType::kFLOAT32,
                                         pt::DataLayout::kNCHW);

  auto dense_x = std::make_shared<pt::DenseTensor>(x_meta, pt::TensorStatus());

  pt::TensorMeta out_meta = pt::TensorMeta(framework::make_ddim({1}),
                                           pt::Backend::kCUDA,
                                           pt::DataType::kFLOAT32,
                                           pt::DataLayout::kNCHW);

  auto dense_out =
      std::make_shared<pt::DenseTensor>(out_meta, pt::TensorStatus());

  pt::TensorMeta grad_out_meta = pt::TensorMeta(framework::make_ddim({1}),
                                                pt::Backend::kCUDA,
                                                pt::DataType::kFLOAT32,
                                                pt::DataLayout::kNCHW);

  auto dense_grad_out =
      std::make_shared<pt::DenseTensor>(grad_out_meta, pt::TensorStatus());

  pt::Copy(*dev_ctx, *dense_ref_x.get(), dense_x.get());
  pt::Copy(*dev_ctx, *dense_ref_out.get(), dense_out.get());
  pt::Copy(*dev_ctx, *dense_ref_grad_out.get(), dense_grad_out.get());

  paddle::experimental::Tensor x(dense_x);
  paddle::experimental::Tensor out(dense_out);
  paddle::experimental::Tensor grad_out(dense_grad_out);

  // 2. test API
  paddle::experimental::Tensor grad_x =
      paddle::experimental::grad_reduce_sum(x, out, grad_out, true, {0}, -1);

  // 3. check result
  ASSERT_EQ(grad_x.shape().size(), 2);
  ASSERT_EQ(grad_x.shape()[0], 3);
  ASSERT_EQ(grad_x.shape()[1], 3);
  ASSERT_EQ(grad_x.numel(), 9);
  ASSERT_EQ(grad_x.type(), pt::DataType::kFLOAT32);
  ASSERT_EQ(grad_x.layout(), pt::DataLayout::kNCHW);
  ASSERT_EQ(grad_x.initialized(), true);

  auto dense_grad_x = std::dynamic_pointer_cast<pt::DenseTensor>(grad_x.impl());
  pt::TensorMeta grad_x_meta = pt::TensorMeta(dense_grad_x->dims(),
                                              pt::Backend::kCPU,
                                              pt::DataType::kFLOAT32,
                                              pt::DataLayout::kNCHW);

  auto ref_grad_x =
      std::make_shared<pt::DenseTensor>(grad_x_meta, pt::TensorStatus());
  pt::Copy(*dev_ctx, *dense_grad_x.get(), ref_grad_x.get());
  for (size_t i = 0; i < 9; i++) {
    ASSERT_NEAR(grad_x_result[i], ref_grad_x->data<float>()[i], 1e-6f);
  }
}
