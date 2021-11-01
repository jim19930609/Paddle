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

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

// Eager
#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/tests/test_utils.h"

// Pten
#include "paddle/pten/api/all.h"
#include "paddle/pten/hapi/all.h"

namespace egr {

TEST(KernelAccuracy, Reduce) {
  for (pten::Backend backend : {pten::Backend::CPU, pten::Backend::CUDA}) {
    if (backend == pten::Backend::CPU) {
      VLOG(1) << "CPU Backend";
      egr::InitEnv(paddle::platform::CPUPlace());
    } else {
      VLOG(1) << "CUDA Backend";
      egr::InitEnv(paddle::platform::CUDAPlace());
    }

    paddle::framework::DDim ddim = paddle::framework::make_ddim({10, 10});
    egr::EagerTensor tensor = EagerUtils::CreateTensorWithValue(
        ddim, backend, pten::DataType::FLOAT32, pten::DataLayout::NCHW, 2.0,
        true);
    RetainGradForTensor(tensor);

    EagerTensor Out = egr::reduce_sum(
        tensor, {0} /*dim*/, false /*keep_dim*/, true /*reduce_all*/,
        -1 /*in_dtype*/, -1 /*out_dtype*/, true /*trace_backward*/);

    std::vector<EagerTensor> target_tensors = {Out};
    RunBackward(target_tensors, {});

    // Examine Forward Grad (w.r.t max_num_runs = 2)
    PADDLE_ENFORCE(
        CompareTensorWithValue<float>(Out, 200) == true,
        paddle::platform::errors::Fatal("Numerical Error, Expected %f", 200.0));
    PADDLE_ENFORCE(
        CompareGradTensorWithValue<float>(tensor, 1) == true,
        paddle::platform::errors::Fatal("Numerical Error, Expected %f", 1.0));
  }
}

TEST(KernelAccuracy, ElementwiseAdd) {
  for (pten::Backend backend : {pten::Backend::CPU, pten::Backend::CUDA}) {
    if (backend == pten::Backend::CPU) {
      VLOG(1) << "CPU Backend";
      egr::InitEnv(paddle::platform::CPUPlace());
    } else {
      VLOG(1) << "CUDA Backend";
      egr::InitEnv(paddle::platform::CUDAPlace());
    }

    paddle::framework::DDim ddim_X = paddle::framework::make_ddim({10, 15});
    egr::EagerTensor X = EagerUtils::CreateTensorWithValue(
        ddim_X, backend, pten::DataType::FLOAT32, pten::DataLayout::NCHW, 1.0,
        true);
    RetainGradForTensor(X);

    paddle::framework::DDim ddim_Y = paddle::framework::make_ddim({15});
    egr::EagerTensor Y = EagerUtils::CreateTensorWithValue(
        ddim_Y, backend, pten::DataType::FLOAT32, pten::DataLayout::NCHW, 2.0,
        true);
    RetainGradForTensor(Y);

    EagerTensor Out =
        egr::elementwise_add(X, Y, -1 /*axis*/, true /*trace_backward*/);

    std::vector<EagerTensor> target_tensors = {Out};
    RunBackward(target_tensors, {});

    // Examine Forward Grad (w.r.t max_num_runs = 2)
    PADDLE_ENFORCE(
        CompareTensorWithValue<float>(Out, 3.0) == true,
        paddle::platform::errors::Fatal("Numerical Error, Expected %f", 200.0));
    PADDLE_ENFORCE(
        CompareGradTensorWithValue<float>(X, 1.0) == true,
        paddle::platform::errors::Fatal("Numerical Error, Expected %f", 1.0));
    PADDLE_ENFORCE(
        CompareGradTensorWithValue<float>(Y, 10.0) == true,
        paddle::platform::errors::Fatal("Numerical Error, Expected %f", 1.0));
  }
}

}  // namespace egr
