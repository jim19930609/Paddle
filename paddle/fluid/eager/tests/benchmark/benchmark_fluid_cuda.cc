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

#include <paddle/fluid/framework/op_registry.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/fluid/eager/tests/benchmark/benchmark_utils.h"
#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/memory/memcpy.h"

#include "gperftools/profiler.h"

namespace paddle {
namespace imperative {

TEST(Benchmark, FluidScaleCUDA) {
  // Prepare Device Contexts
  platform::CUDAPlace place;
  egr::InitEnv(place);

  for (const std::string& mode : {"Accuracy", "WarmUp", "Performance"}) {
    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);

    std::vector<float> src_data(128, 5.0);
    std::vector<int64_t> dims = {2, 4, 4, 4};

    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);

    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx =
        dynamic_cast<paddle::platform::CUDADeviceContext*>(pool.Get(place));
    auto stream = dev_ctx->stream();
    paddle::memory::Copy(place, mutable_x, platform::CPUPlace(),
                         src_data.data(), sizeof(float) * src_data.size(),
                         stream);

    if (mode == "Accuracy") {
      benchmark_fluid_scale(X, platform::Place(place),
                            true /* accuracy_check */);

    } else if (mode == "WarmUp") {
      benchmark_fluid_scale(X, platform::Place(place));

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("fluid_scale_cuda.out");

      benchmark_fluid_scale(X, platform::Place(place));

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

TEST(Benchmark, FluidMatmulCUDA) {
  // Prepare Device Contexts
  platform::CUDAPlace place;
  egr::InitEnv(place);

  for (const std::string& mode : {"Accuracy", "WarmUp", "Performance"}) {
    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);
    std::shared_ptr<imperative::VarBase> Y(new imperative::VarBase(true, "Y"));
    Y->SetOverridedStopGradient(false);

    std::vector<float> x_src_data(4, 1.0);
    std::vector<float> y_src_data(4, 2.0);
    std::vector<int64_t> dims = {2, 2};

    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx =
        dynamic_cast<paddle::platform::CUDADeviceContext*>(pool.Get(place));
    auto stream = dev_ctx->stream();

    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, platform::CPUPlace(),
                         x_src_data.data(), sizeof(float) * x_src_data.size(),
                         stream);

    auto* y_tensor = Y->MutableVar()->GetMutable<framework::LoDTensor>();
    y_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_y = y_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_y, platform::CPUPlace(),
                         y_src_data.data(), sizeof(float) * y_src_data.size(),
                         stream);

    if (mode == "Accuracy") {
      benchmark_fluid_matmul(X, Y, platform::Place(place),
                             true /* accuracy_check */);

    } else if (mode == "WarmUp") {
      benchmark_fluid_matmul(X, Y, platform::Place(place));

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("fluid_matmul_cuda.out");

      benchmark_fluid_matmul(X, Y, platform::Place(place));

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

TEST(Benchmark, FluidMLPCUDA) {
  // Prepare Device Contexts
  platform::CUDAPlace place;
  egr::InitEnv(place);

  for (const std::string& mode : {"Accuracy", "WarmUp", "Performance"}) {
    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);
    std::shared_ptr<imperative::VarBase> W1(new imperative::VarBase(true, "Y"));
    W1->SetOverridedStopGradient(false);
    std::shared_ptr<imperative::VarBase> W2(new imperative::VarBase(true, "Y"));
    W2->SetOverridedStopGradient(false);

    std::vector<float> x_src_data(64, 1.0);
    std::vector<float> w1_src_data(512, 2.0);
    std::vector<float> w2_src_data(2048, 3.0);
    std::vector<int64_t> x_dims = {4, 16};
    std::vector<int64_t> w1_dims = {16, 32};
    std::vector<int64_t> w2_dims = {32, 64};

    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx =
        dynamic_cast<paddle::platform::CUDADeviceContext*>(pool.Get(place));
    auto stream = dev_ctx->stream();

    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(x_dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, platform::CPUPlace(),
                         x_src_data.data(), sizeof(float) * x_src_data.size(),
                         stream);

    auto* w1_tensor = W1->MutableVar()->GetMutable<framework::LoDTensor>();
    w1_tensor->Resize(framework::make_ddim(w1_dims));
    auto* mutable_w1 = w1_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_w1, platform::CPUPlace(),
                         w1_src_data.data(), sizeof(float) * w1_src_data.size(),
                         stream);

    auto* w2_tensor = W2->MutableVar()->GetMutable<framework::LoDTensor>();
    w2_tensor->Resize(framework::make_ddim(w2_dims));
    auto* mutable_w2 = w2_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_w2, platform::CPUPlace(),
                         w2_src_data.data(), sizeof(float) * w2_src_data.size(),
                         stream);

    if (mode == "Accuracy") {
      benchmark_fluid_mlp(X, W1, W2, platform::Place(place),
                          true /* accuracy_check */);

    } else if (mode == "WarmUp") {
      benchmark_fluid_mlp(X, W1, W2, platform::Place(place));

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("fluid_matmul_cpu.out");

      benchmark_fluid_mlp(X, W1, W2, platform::Place(place));

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

}  // namespace imperative
}  // namespace paddle

USE_OP(scale);
USE_OP(matmul_v2);
USE_OP(reduce_sum);
USE_OP(reduce_sum_grad);
