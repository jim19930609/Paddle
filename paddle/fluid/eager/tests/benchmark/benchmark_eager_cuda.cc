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

// Eager Dygraph
#include <chrono>

#include "gtest/gtest.h"

#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"

#include "paddle/fluid/imperative/tracer.h"

#include "paddle/fluid/eager/tests/benchmark/benchmark_utils.h"
#include "paddle/fluid/eager/tests/test_utils.h"

#include "gperftools/profiler.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

TEST(Benchmark, EagerScaleCUDA) {
  egr::InitEnv(paddle::platform::CUDAPlace());

  for (const std::string& mode : {"Accuracy", "WarmUp", "Performance"}) {
    paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 4, 4, 4});
    paddle::experimental::Tensor tensor = EagerUtils::CreateTensorWithValue(
        ddim, pt::Backend::kCUDA, pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
        5.0 /*value*/, true /*is_leaf*/);
    RetainGradForTensor(tensor);

    if (mode == "Accuracy") {
      benchmark_eager_scale(tensor, true /* accuracy_check */);

    } else if (mode == "WarmUp") {
      benchmark_eager_scale(tensor);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("eager_scale_cuda.out");

      benchmark_eager_scale(tensor);

      ProfilerStop();
      auto t_end = std::chrono::high_resolution_clock::now();
      double elapsed_time_ms =
          std::chrono::duration<double, std::milli>(t_end - t_start).count();
      std::cout << "Duration: " << elapsed_time_ms << " ms" << std::endl;

    } else {
      PADDLE_THROW(paddle::platform::errors::Fatal("Unknown benchmark mode"));
    }
  }
}

TEST(Benchmark, EagerIntermediateMatmulCUDA) {
  paddle::platform::CUDAPlace place;
  egr::InitEnv(place);

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  tracer->SetExpectedPlace(place);
  paddle::imperative::SetCurrentTracer(tracer);

  for (const std::string& mode : {"Accuracy", "WarmUp", "Performance"}) {
    paddle::framework::DDim ddimX = paddle::framework::make_ddim({2, 2});
    paddle::experimental::Tensor X = EagerUtils::CreateTensorWithValue(
        ddimX, pt::Backend::kCUDA, pt::DataType::kFLOAT32,
        pt::DataLayout::kNCHW, 1.0, true);
    RetainGradForTensor(X);

    paddle::framework::DDim ddimY = paddle::framework::make_ddim({2, 2});
    paddle::experimental::Tensor Y = EagerUtils::CreateTensorWithValue(
        ddimY, pt::Backend::kCUDA, pt::DataType::kFLOAT32,
        pt::DataLayout::kNCHW, 2.0, true);
    RetainGradForTensor(Y);

    if (mode == "Accuracy") {
      benchmark_eager_intermediate_matmul(X, Y, true /* accuracy_check */);

    } else if (mode == "WarmUp") {
      benchmark_eager_intermediate_matmul(X, Y);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("eager_intermediate_matmul_cuda.out");

      benchmark_eager_intermediate_matmul(X, Y);

      ProfilerStop();
      auto t_end = std::chrono::high_resolution_clock::now();
      double elapsed_time_ms =
          std::chrono::duration<double, std::milli>(t_end - t_start).count();
      std::cout << "Duration: " << elapsed_time_ms << " ms" << std::endl;

    } else {
      PADDLE_THROW(paddle::platform::errors::Fatal("Unknown benchmark mode"));
    }
  }
}

TEST(Benchmark, EagerIntermediateMLPCUDA) {
  paddle::platform::CUDAPlace place;
  egr::InitEnv(place);

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  tracer->SetExpectedPlace(place);
  paddle::imperative::SetCurrentTracer(tracer);

  for (const std::string& mode : {"Accuracy", "WarmUp", "Performance"}) {
    paddle::framework::DDim ddimX = paddle::framework::make_ddim({4, 16});
    paddle::experimental::Tensor X = EagerUtils::CreateTensorWithValue(
        ddimX, pt::Backend::kCUDA, pt::DataType::kFLOAT32,
        pt::DataLayout::kNCHW, 1.0, true);
    RetainGradForTensor(X);

    paddle::framework::DDim ddimW1 = paddle::framework::make_ddim({16, 32});
    paddle::experimental::Tensor W1 = EagerUtils::CreateTensorWithValue(
        ddimW1, pt::Backend::kCUDA, pt::DataType::kFLOAT32,
        pt::DataLayout::kNCHW, 2.0, true);
    RetainGradForTensor(W1);

    paddle::framework::DDim ddimW2 = paddle::framework::make_ddim({32, 64});
    paddle::experimental::Tensor W2 = EagerUtils::CreateTensorWithValue(
        ddimW2, pt::Backend::kCUDA, pt::DataType::kFLOAT32,
        pt::DataLayout::kNCHW, 3.0, true);
    RetainGradForTensor(W2);

    if (mode == "Accuracy") {
      benchmark_eager_intermediate_mlp(X, W1, W2, true /* accuracy_check */);

    } else if (mode == "WarmUp") {
      benchmark_eager_intermediate_mlp(X, W1, W2);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("eager_intermediate_matmul_mlp.out");

      benchmark_eager_intermediate_mlp(X, W1, W2);

      ProfilerStop();
      auto t_end = std::chrono::high_resolution_clock::now();
      double elapsed_time_ms =
          std::chrono::duration<double, std::milli>(t_end - t_start).count();
      std::cout << "Duration: " << elapsed_time_ms << " ms" << std::endl;

    } else {
      PADDLE_THROW(paddle::platform::errors::Fatal("Unknown benchmark mode"));
    }
  }
}
