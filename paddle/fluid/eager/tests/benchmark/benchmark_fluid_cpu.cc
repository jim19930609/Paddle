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

// Disable pten path
DECLARE_bool(run_pt_kernel);

TEST(Benchmark, Init) { FLAGS_run_pt_kernel = false; }

namespace paddle {
namespace imperative {

TEST(Benchmark, FluidScaleCPU) {
  // Prepare Device Contexts
  platform::CPUPlace place;
  egr::InitEnv(place);

  for (const std::string& mode : {"Accuracy", "Performance"}) {
    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);

    std::vector<float> src_data(128, 5.0);
    std::vector<int64_t> dims = {2, 4, 4, 4};

    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, place, src_data.data(),
                         sizeof(float) * src_data.size());

    if (mode == "Accuracy") {
      benchmark_fluid_scale(X, platform::Place(place),
                            true /* accuracy_check */);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("fluid_scale_cpu.out");

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

TEST(Benchmark, FluidMatmulCPU) {
  // Prepare Device Contexts
  platform::CPUPlace place;
  egr::InitEnv(place);

  for (const std::string& mode : {"Accuracy", "Performance"}) {
    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);
    std::shared_ptr<imperative::VarBase> Y(new imperative::VarBase(true, "Y"));
    Y->SetOverridedStopGradient(false);

    std::vector<float> x_src_data(4, 1.0);
    std::vector<float> y_src_data(4, 2.0);
    std::vector<int64_t> dims = {2, 2};

    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, place, x_src_data.data(),
                         sizeof(float) * x_src_data.size());

    auto* y_tensor = Y->MutableVar()->GetMutable<framework::LoDTensor>();
    y_tensor->Resize(framework::make_ddim(dims));
    auto* mutable_y = y_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_y, place, y_src_data.data(),
                         sizeof(float) * y_src_data.size());

    if (mode == "Accuracy") {
      benchmark_fluid_matmul(X, Y, platform::Place(place),
                             true /* accuracy_check */);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("fluid_matmul_cpu.out");

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

TEST(Benchmark, FluidMLPCPU) {
  // Prepare Device Contexts
  platform::CPUPlace place;
  egr::InitEnv(place);

  for (const std::string& mode : {"Accuracy", "Performance"}) {
    std::shared_ptr<imperative::VarBase> X(new imperative::VarBase(true, "X"));
    X->SetOverridedStopGradient(false);
    std::shared_ptr<imperative::VarBase> W1(
        new imperative::VarBase(true, "W1"));
    W1->SetOverridedStopGradient(false);
    std::shared_ptr<imperative::VarBase> W2(
        new imperative::VarBase(true, "W2"));
    W2->SetOverridedStopGradient(false);
    std::shared_ptr<imperative::VarBase> B1(
        new imperative::VarBase(true, "B1"));
    B1->SetOverridedStopGradient(false);
    std::shared_ptr<imperative::VarBase> B2(
        new imperative::VarBase(true, "B2"));
    B2->SetOverridedStopGradient(false);

    std::vector<float> x_src_data(MLP_M * MLP_N, MLP_X_VAL);
    std::vector<float> w1_src_data(MLP_N * MLP_K1, MLP_W1_VAL);
    std::vector<float> w2_src_data(MLP_K1 * MLP_K2, MLP_W2_VAL);
    std::vector<float> b1_src_data(MLP_K1, MLP_B1_VAL);
    std::vector<float> b2_src_data(MLP_K2, MLP_B2_VAL);
    std::vector<int64_t> x_dims = {MLP_M, MLP_N};
    std::vector<int64_t> w1_dims = {MLP_N, MLP_K1};
    std::vector<int64_t> w2_dims = {MLP_K1, MLP_K2};
    std::vector<int64_t> b1_dims = {MLP_K1};
    std::vector<int64_t> b2_dims = {MLP_K2};

    auto* x_tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
    x_tensor->Resize(framework::make_ddim(x_dims));
    auto* mutable_x = x_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_x, place, x_src_data.data(),
                         sizeof(float) * x_src_data.size());

    auto* w1_tensor = W1->MutableVar()->GetMutable<framework::LoDTensor>();
    w1_tensor->Resize(framework::make_ddim(w1_dims));
    auto* mutable_w1 = w1_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_w1, place, w1_src_data.data(),
                         sizeof(float) * w1_src_data.size());

    auto* w2_tensor = W2->MutableVar()->GetMutable<framework::LoDTensor>();
    w2_tensor->Resize(framework::make_ddim(w2_dims));
    auto* mutable_w2 = w2_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_w2, place, w2_src_data.data(),
                         sizeof(float) * w2_src_data.size());

    auto* b1_tensor = B1->MutableVar()->GetMutable<framework::LoDTensor>();
    b1_tensor->Resize(framework::make_ddim(b1_dims));
    auto* mutable_b1 = b1_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_b1, place, b1_src_data.data(),
                         sizeof(float) * b1_src_data.size());

    auto* b2_tensor = B2->MutableVar()->GetMutable<framework::LoDTensor>();
    b2_tensor->Resize(framework::make_ddim(b2_dims));
    auto* mutable_b2 = b2_tensor->mutable_data<float>(place);
    paddle::memory::Copy(place, mutable_b2, place, b2_src_data.data(),
                         sizeof(float) * b2_src_data.size());

    if (mode == "Accuracy") {
      benchmark_fluid_mlp(X, W1, W2, B1, B2, platform::Place(place),
                          true /* accuracy_check */);

    } else if (mode == "Performance") {
      auto t_start = std::chrono::high_resolution_clock::now();
      ProfilerStart("fluid_matmul_cpu.out");

      benchmark_fluid_mlp(X, W1, W2, B1, B2, platform::Place(place));

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
