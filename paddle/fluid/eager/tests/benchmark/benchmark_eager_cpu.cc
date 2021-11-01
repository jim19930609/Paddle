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
      ProfilerStart("eager_matmul_cpu.out");

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

    std::vector<EagerTensor> Ws;
    std::vector<EagerTensor> Bs;
    for (size_t i = 0; i < MLP_NUM_LINEAR; i++) {
      paddle::framework::DDim ddimW =
          paddle::framework::make_ddim({MLP_N, MLP_K});
      egr::EagerTensor W = EagerUtils::CreateTensorWithValue(
          ddimW, pten::Backend::CPU, pten::DataType::FLOAT32,
          pten::DataLayout::NCHW, MLP_W_VAL, true);
      RetainGradForTensor(W);

      paddle::framework::DDim ddimB = paddle::framework::make_ddim({MLP_K});
      egr::EagerTensor B = EagerUtils::CreateTensorWithValue(
          ddimB, pten::Backend::CPU, pten::DataType::FLOAT32,
          pten::DataLayout::NCHW, MLP_B_VAL, true);
      RetainGradForTensor(B);

      Ws.emplace_back(std::move(W));
      Bs.emplace_back(std::move(B));
    }

    if (mode == "Accuracy") {
      benchmark_eager_mlp(X, Ws, Bs, true /* accuracy_check */);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("eager_mlp_cpu.out");

      benchmark_eager_mlp(X, Ws, Bs);

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

    std::vector<EagerTensor> Ws;
    std::vector<EagerTensor> Bs;
    for (size_t i = 0; i < MLP_NUM_LINEAR; i++) {
      paddle::framework::DDim ddimW =
          paddle::framework::make_ddim({MLP_N, MLP_K});
      egr::EagerTensor W = EagerUtils::CreateTensorWithValue(
          ddimW, pten::Backend::CPU, pten::DataType::FLOAT32,
          pten::DataLayout::NCHW, MLP_W_VAL, true);
      RetainGradForTensor(W);

      paddle::framework::DDim ddimB = paddle::framework::make_ddim({MLP_K});
      egr::EagerTensor B = EagerUtils::CreateTensorWithValue(
          ddimB, pten::Backend::CPU, pten::DataType::FLOAT32,
          pten::DataLayout::NCHW, MLP_B_VAL, true);
      RetainGradForTensor(B);

      Ws.emplace_back(std::move(W));
      Bs.emplace_back(std::move(B));
    }

    if (mode == "Accuracy") {
      benchmark_eager_intermediate_mlp(X, Ws, Bs, true /* accuracy_check */);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("eager_intermediate_mlp_cpu.out");

      benchmark_eager_intermediate_mlp(X, Ws, Bs);

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
