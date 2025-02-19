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

#include "paddle/pten/api/include/api.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"

namespace paddle {
namespace tests {

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;

// TODO(chenweihang): Remove this test after the API is used in the dygraph
TEST(API, empty_like) {
  // 1. create tensor
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());
  auto dense_x = std::make_shared<pten::DenseTensor>(
      alloc,
      pten::DenseTensorMeta(pten::DataType::FLOAT32,
                            framework::make_ddim({3, 2}),
                            pten::DataLayout::NCHW));

  paddle::experimental::Tensor x(dense_x);

  // 2. test API
  auto out = paddle::experimental::empty_like(x, pten::DataType::FLOAT32);

  // 3. check result
  ASSERT_EQ(out.dims().size(), 2);
  ASSERT_EQ(out.dims()[0], 3);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pten::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);
}

TEST(API, empty1) {
  // 1. create tensor
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  auto dense_shape = std::make_shared<pten::DenseTensor>(
      alloc,
      pten::DenseTensorMeta(pten::DataType::INT64,
                            framework::make_ddim({2}),
                            pten::DataLayout::NCHW));
  auto* shape_data = dense_shape->mutable_data<int64_t>();
  shape_data[0] = 2;
  shape_data[1] = 3;

  paddle::experimental::Tensor tensor_shape(dense_shape);

  // 2. test API
  auto out = paddle::experimental::empty(tensor_shape, pten::DataType::FLOAT32);

  // 3. check result
  ASSERT_EQ(out.shape().size(), 2UL);
  ASSERT_EQ(out.shape()[0], 2);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pten::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);
}

TEST(API, empty2) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  auto dense_scalar = std::make_shared<pten::DenseTensor>(
      alloc,
      pten::DenseTensorMeta(pten::DataType::INT32,
                            framework::make_ddim({1}),
                            pten::DataLayout::NCHW));
  dense_scalar->mutable_data<int32_t>()[0] = 2;

  paddle::experimental::Tensor shape_scalar1(dense_scalar);
  paddle::experimental::Tensor shape_scalar2(dense_scalar);
  std::vector<paddle::experimental::Tensor> list_shape{shape_scalar1,
                                                       shape_scalar2};

  auto out = paddle::experimental::empty(list_shape, pten::DataType::FLOAT32);

  ASSERT_EQ(out.shape().size(), 2UL);
  ASSERT_EQ(out.shape()[0], 2);
  ASSERT_EQ(out.numel(), 4);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pten::DataType::FLOAT32);
  ASSERT_EQ(out.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);
}

TEST(API, empty3) {
  std::vector<int64_t> vector_shape{2, 3};

  auto out = paddle::experimental::empty(vector_shape, pten::DataType::INT32);

  ASSERT_EQ(out.shape().size(), 2UL);
  ASSERT_EQ(out.shape()[0], 2);
  ASSERT_EQ(out.numel(), 6);
  ASSERT_EQ(out.is_cpu(), true);
  ASSERT_EQ(out.type(), pten::DataType::INT32);
  ASSERT_EQ(out.layout(), pten::DataLayout::NCHW);
  ASSERT_EQ(out.initialized(), true);
}

}  // namespace tests
}  // namespace paddle
