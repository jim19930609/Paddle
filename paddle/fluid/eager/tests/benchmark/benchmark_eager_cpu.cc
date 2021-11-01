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
#include "paddle/fluid/platform/flags.h"

#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"

#include "paddle/fluid/imperative/tracer.h"

#include "paddle/fluid/eager/tests/benchmark/benchmark_utils.h"
#include "paddle/fluid/eager/tests/test_utils.h"

#include "gperftools/profiler.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

// Disable pten path
DECLARE_bool(run_pt_kernel);

TEST(Benchmark, Init) { FLAGS_run_pt_kernel = false; }

TEST(Benchmark, EagerScaleCPU) {
  // Prepare Device Contexts
  egr::InitEnv(paddle::platform::CPUPlace());

  for (const std::string& mode : {"Accuracy", "Performance"}) {
    paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 4, 4, 4});
    egr::EagerTensor tensor = EagerUtils::CreateTensorWithValue(
        ddim, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, 5.0, true);
    RetainGradForTensor(tensor);

    if (mode == "Accuracy") {
      benchmark_eager_scale(tensor, true /* accuracy_check*/);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("eager_scale_cpu.out");

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

TEST(Benchmark, EagerMatmulCPU) {
  // Prepare Device Contexts
  InitEnv(paddle::platform::CPUPlace());

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  paddle::imperative::SetCurrentTracer(tracer);

  for (const std::string& mode : {"Accuracy", "Performance"}) {
    paddle::framework::DDim ddimX = paddle::framework::make_ddim({2, 2});
    egr::EagerTensor X = EagerUtils::CreateTensorWithValue(
        ddimX, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, 1.0, true);
    RetainGradForTensor(X);

    paddle::framework::DDim ddimY = paddle::framework::make_ddim({2, 2});
    egr::EagerTensor Y = EagerUtils::CreateTensorWithValue(
        ddimY, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, 2.0, true);
    RetainGradForTensor(Y);

    if (mode == "Accuracy") {
      benchmark_eager_matmul(X, Y, true /* accuracy_check */);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("eager_intermediate_matmul_cpu.out");

      benchmark_eager_matmul(X, Y);

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

TEST(Benchmark, EagerMLPCPU) {
  // Prepare Device Contexts
  InitEnv(paddle::platform::CPUPlace());

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  paddle::imperative::SetCurrentTracer(tracer);

  for (const std::string& mode : {"Accuracy", "Performance"}) {
    paddle::framework::DDim ddimX =
        paddle::framework::make_ddim({MLP_M, MLP_N});
    egr::EagerTensor X = EagerUtils::CreateTensorWithValue(
        ddimX, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, MLP_X_VAL, true);
    RetainGradForTensor(X);

    paddle::framework::DDim ddimW1 =
        paddle::framework::make_ddim({MLP_N, MLP_K1});
    egr::EagerTensor W1 = EagerUtils::CreateTensorWithValue(
        ddimW1, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, MLP_W1_VAL, true);
    RetainGradForTensor(W1);

    paddle::framework::DDim ddimW2 =
        paddle::framework::make_ddim({MLP_K1, MLP_K2});
    egr::EagerTensor W2 = EagerUtils::CreateTensorWithValue(
        ddimW2, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, MLP_W2_VAL, true);
    RetainGradForTensor(W2);

    paddle::framework::DDim ddimB1 = paddle::framework::make_ddim({MLP_K1});
    egr::EagerTensor B1 = EagerUtils::CreateTensorWithValue(
        ddimB1, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, MLP_B1_VAL, true);
    RetainGradForTensor(B1);

    paddle::framework::DDim ddimB2 = paddle::framework::make_ddim({MLP_K2});
    egr::EagerTensor B2 = EagerUtils::CreateTensorWithValue(
        ddimB2, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, MLP_B2_VAL, true);
    RetainGradForTensor(B2);

    if (mode == "Accuracy") {
      benchmark_eager_mlp(X, W1, W2, B1, B2, true /* accuracy_check */);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("eager_intermediate_matmul_mlp.out");

      benchmark_eager_mlp(X, W1, W2, B1, B2);

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

TEST(Benchmark, EagerIntermediateMatmulCPU) {
  // Prepare Device Contexts
  InitEnv(paddle::platform::CPUPlace());

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  paddle::imperative::SetCurrentTracer(tracer);

  for (const std::string& mode : {"Accuracy", "Performance"}) {
    paddle::framework::DDim ddimX = paddle::framework::make_ddim({2, 2});
    egr::EagerTensor X = EagerUtils::CreateTensorWithValue(
        ddimX, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, 1.0, true);
    RetainGradForTensor(X);

    paddle::framework::DDim ddimY = paddle::framework::make_ddim({2, 2});
    egr::EagerTensor Y = EagerUtils::CreateTensorWithValue(
        ddimY, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, 2.0, true);
    RetainGradForTensor(Y);

    if (mode == "Accuracy") {
      benchmark_eager_intermediate_matmul(X, Y, true /* accuracy_check */);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("eager_intermediate_matmul_cpu.out");

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

TEST(Benchmark, EagerIntermediateMLPCPU) {
  // Prepare Device Contexts
  InitEnv(paddle::platform::CPUPlace());

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  paddle::imperative::SetCurrentTracer(tracer);

  for (const std::string& mode : {"Accuracy", "Performance"}) {
    paddle::framework::DDim ddimX =
        paddle::framework::make_ddim({MLP_M, MLP_N});
    egr::EagerTensor X = EagerUtils::CreateTensorWithValue(
        ddimX, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, MLP_X_VAL, true);
    RetainGradForTensor(X);

    paddle::framework::DDim ddimW1 =
        paddle::framework::make_ddim({MLP_N, MLP_K1});
    egr::EagerTensor W1 = EagerUtils::CreateTensorWithValue(
        ddimW1, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, MLP_W1_VAL, true);
    RetainGradForTensor(W1);

    paddle::framework::DDim ddimW2 =
        paddle::framework::make_ddim({MLP_K1, MLP_K2});
    egr::EagerTensor W2 = EagerUtils::CreateTensorWithValue(
        ddimW2, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, MLP_W2_VAL, true);
    RetainGradForTensor(W2);

    paddle::framework::DDim ddimB1 = paddle::framework::make_ddim({MLP_K1});
    egr::EagerTensor B1 = EagerUtils::CreateTensorWithValue(
        ddimB1, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, MLP_B1_VAL, true);
    RetainGradForTensor(B1);

    paddle::framework::DDim ddimB2 = paddle::framework::make_ddim({MLP_K2});
    egr::EagerTensor B2 = EagerUtils::CreateTensorWithValue(
        ddimB2, pten::Backend::CPU, pten::DataType::FLOAT32,
        pten::DataLayout::NCHW, MLP_B2_VAL, true);
    RetainGradForTensor(B2);

    if (mode == "Accuracy") {
      benchmark_eager_intermediate_mlp(X, W1, W2, B1, B2,
                                       true /* accuracy_check */);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("eager_intermediate_matmul_mlp.out");

      benchmark_eager_intermediate_mlp(X, W1, W2, B1, B2);

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
