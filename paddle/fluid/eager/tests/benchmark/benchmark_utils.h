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

#pragma once

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/pten/api/all.h"
#include "paddle/pten/hapi/all.h"

/* MLP Configurations */
// Out1 = X[M, N] x W1[N, K1] + B1[K1]
// Out2 = Out1[M, K1] x W2[K1, K2] + B2[K2]
// Out  = ReduceSum(Out2)
#define MLP_M 4
#define MLP_N 16
#define MLP_K1 32
#define MLP_K2 64
#define MLP_X_VAL 1.0
#define MLP_W1_VAL 2.0
#define MLP_W2_VAL 3.0
#define MLP_B1_VAL 4.0
#define MLP_B2_VAL 5.0

namespace egr {

inline std::unordered_map<std::string, float> compute_mlp_expected_results() {
  float Out1 = MLP_N * (MLP_X_VAL * MLP_W1_VAL) + MLP_B1_VAL;
  float Out2 = MLP_K1 * (Out1 * MLP_W2_VAL) + MLP_B2_VAL;
  float Out = Out2 * MLP_M * MLP_K2;

  float GradOut = 1.0;
  float GradOut2 = GradOut;
  float GradW2 = GradOut2 * Out1 * MLP_M;
  float GradOut1 = GradOut2 * MLP_W2_VAL * MLP_K2;
  float GradX = GradOut1 * MLP_W1_VAL * MLP_K1;
  float GradW1 = GradOut1 * MLP_X_VAL * MLP_M;

  return {
      {"Out", Out}, {"GradX", GradX}, {"GradW1", GradW1}, {"GradW2", GradW2}};
}

/* ---- Eager Scale ---- */
void benchmark_eager_scale(const EagerTensor& tensor,
                           bool accuracy_check = false);

/* ---- Eager MatMul ---- */
void benchmark_eager_matmul(const EagerTensor& X, const EagerTensor& Y,
                            bool accuracy_check = false);

void benchmark_eager_mlp(const EagerTensor& X, const EagerTensor& W1,
                         const EagerTensor& W2, const EagerTensor& B1,
                         const EagerTensor& B2, bool accuracy_check = false);

void benchmark_eager_intermediate_matmul(const EagerTensor& X,
                                         const EagerTensor& Y,
                                         bool accuracy_check = false);

void benchmark_eager_intermediate_mlp(
    const EagerTensor& X, const EagerTensor& W1, const EagerTensor& W2,
    const EagerTensor& B1, const EagerTensor& B2, bool accuracy_check = false);

}  // namespace egr

namespace paddle {
namespace imperative {
/* ---- Fluid Scale ---- */
// TODO(jiabin): Change this and remove nolint
void benchmark_fluid_scale(
    const std::shared_ptr<imperative::VarBase>& X,  // NOLINT
    const paddle::platform::Place& place, bool accuracy_check = false);

/* ---- Fluid MatMul ---- */
void benchmark_fluid_matmul(
    const std::shared_ptr<imperative::VarBase>& X,
    const std::shared_ptr<imperative::VarBase>& Y,  // NOLINT
    const paddle::platform::Place& place, bool accuracy_check = false);

/* ---- Fluid MLP ---- */
void benchmark_fluid_mlp(const std::shared_ptr<imperative::VarBase>& X,
                         const std::shared_ptr<imperative::VarBase>& W1,
                         const std::shared_ptr<imperative::VarBase>& W2,
                         const std::shared_ptr<imperative::VarBase>& B1,
                         const std::shared_ptr<imperative::VarBase>& B2,
                         const paddle::platform::Place& place,
                         bool accuracy_check = false);

}  // namespace imperative
}  // namespace paddle
