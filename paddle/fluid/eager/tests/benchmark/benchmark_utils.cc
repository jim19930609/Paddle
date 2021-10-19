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

// Eager
#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/tests/test_utils.h"

// Eager Generated
#include "paddle/fluid/eager/generated/dygraph_forward_api.h"

// Fluid
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/memory/memcpy.h"

#include "paddle/fluid/eager/tests/benchmark/benchmark_utils.h"

static size_t max_num_benchmark_runs = 5000;

namespace egr {

/* --------------------- */
/* ---- Eager Scale ---- */
/* --------------------- */
void benchmark_eager_scale(const paddle::experimental::Tensor& tensor, bool accuracy_check) {
  paddle::experimental::Tensor input_tensor = tensor;
  float scale = 2.0;
  float bias = 3.0;

  size_t max_num_runs = accuracy_check ? 10 : max_num_benchmark_runs;
  for (size_t i = 0; i < max_num_runs; i++) {
    input_tensor =
        egr::scale(input_tensor, scale, bias, true /*bias_after_scale*/,
                   true /*trace_backward*/);
  }

  std::vector<paddle::experimental::Tensor> target_tensors = {input_tensor};
  RunBackward(target_tensors, {});

  if (accuracy_check) {
    // Examine Forward Grad (w.r.t max_num_runs = 10)
    PADDLE_ENFORCE(CompareTensorWithValue<float>(input_tensor, 8189.0) == true,
                   paddle::platform::errors::Fatal(
                       "Numerical Error, Expected %f", 8189.0));
    // Examine Backward Grad (w.r.t max_num_runs = 10)
    PADDLE_ENFORCE(CompareGradTensorWithValue<float>(tensor, 1024.0) == true,
                   paddle::platform::errors::Fatal(
                       "Numerical Error, Expected %f", 1024.0));
  }
}

/* ----------------------------------- */
/* ---- Eager Intermediate Matmul ---- */
/* ----------------------------------- */
void benchmark_eager_intermediate_matmul(const paddle::experimental::Tensor& X,
                                         const paddle::experimental::Tensor& Y,
                                         bool accuracy_check) {
  paddle::experimental::Tensor input_tensor0 = X;

  size_t max_num_runs = accuracy_check ? 2 : max_num_benchmark_runs;
  for (size_t i = 0; i < max_num_runs; i++) {
    input_tensor0 = matmul_v2_dygraph_function(
        input_tensor0, Y, false /*trans_x*/, false /*trans_y*/,
        false /*use_mkldnn*/, "float32" /*mkldnn_data_type*/, 0 /*op_role*/,
        {} /*op_role_var*/, "" /*op_namescope*/, {} /*op_callstack*/,
        "" /*op_device*/, false /*with_quant_attr*/, true /*trace_backward*/);
  }

  std::vector<paddle::experimental::Tensor> target_tensors = {input_tensor0};
  RunBackward(target_tensors, {});

  if (accuracy_check) {
    // Examine Forward Grad (w.r.t max_num_runs = 2)
    PADDLE_ENFORCE(CompareTensorWithValue<float>(input_tensor0, 16) == true,
                   paddle::platform::errors::Fatal(
                       "Numerical Error, Expected %f", 8189.0));
    // Examine Backward Grad (w.r.t max_num_runs = 2)
    PADDLE_ENFORCE(CompareGradTensorWithValue<float>(X, 16) == true,
                   paddle::platform::errors::Fatal(
                       "Numerical Error, Expected %f", 1024.0));
    PADDLE_ENFORCE(CompareGradTensorWithValue<float>(Y, 16) == true,
                   paddle::platform::errors::Fatal(
                       "Numerical Error, Expected %f", 1024.0));
  }
}

/* -------------------------------- */
/* ---- Eager Intermediate MLP ---- */
/* -------------------------------- */
void benchmark_eager_intermediate_mlp(const paddle::experimental::Tensor& X, const paddle::experimental::Tensor& W1,
                                      const paddle::experimental::Tensor& W2,
                                      bool accuracy_check) {
  paddle::experimental::Tensor Out1 = matmul_v2_dygraph_function(
      X, W1, false /*trans_x*/, false /*trans_y*/, false /*use_mkldnn*/,
      "float32" /*mkldnn_data_type*/, 0 /*op_role*/, {} /*op_role_var*/,
      "" /*op_namescope*/, {} /*op_callstack*/, "" /*op_device*/,
      false /*with_quant_attr*/, true /*trace_backward*/);

  paddle::experimental::Tensor Out2 = matmul_v2_dygraph_function(
      Out1, W2, false /*trans_x*/, false /*trans_y*/, false /*use_mkldnn*/,
      "float32" /*mkldnn_data_type*/, 0 /*op_role*/, {} /*op_role_var*/,
      "" /*op_namescope*/, {} /*op_callstack*/, "" /*op_device*/,
      false /*with_quant_attr*/, true /*trace_backward*/);

  paddle::experimental::Tensor Out = reduce_sum_dygraph_function(
      Out2, {0} /*dim*/, false /*keep_dim*/, true /*reduce_all*/,
      -1 /*in_dtype*/, -1 /*out_dtype*/, false /*use_mkldnn*/, 0 /*op_role*/,
      {} /*op_role_var*/, "" /*op_namescope*/, {} /*op_callstack*/,
      "" /*op_device*/, false /*with_quant_attr*/, true /*trace_backward*/);

  std::vector<paddle::experimental::Tensor> target_tensors = {Out};
  RunBackward(target_tensors, {});

  if (accuracy_check) {
    // Examine Forward Grad (w.r.t max_num_runs = 2)
    PADDLE_ENFORCE(CompareTensorWithValue<float>(Out, 786432) == true,
                   paddle::platform::errors::Fatal(
                       "Numerical Error, Expected %f", 786432));
    // Examine Backward Grad (w.r.t max_num_runs = 2)
    PADDLE_ENFORCE(CompareGradTensorWithValue<float>(X, 12288) == true,
                   paddle::platform::errors::Fatal(
                       "Numerical Error, Expected %f", 12288.0));
    PADDLE_ENFORCE(
        CompareGradTensorWithValue<float>(W1, 768) == true,
        paddle::platform::errors::Fatal("Numerical Error, Expected %f", 768.0));
    PADDLE_ENFORCE(
        CompareGradTensorWithValue<float>(W2, 128) == true,
        paddle::platform::errors::Fatal("Numerical Error, Expected %f", 128.0));
  }
}

}  // namespace egr

namespace paddle {
namespace imperative {

static void FluidCheckTensorValue(const std::shared_ptr<imperative::VarBase>& X,
                                  const paddle::platform::Place& place,
                                  float value) {
  auto* tensor = X->MutableVar()->GetMutable<framework::LoDTensor>();
  float* t_ptr = tensor->mutable_data<float>(place);
  std::vector<float> host_data(tensor->numel());
  if (place == paddle::platform::CUDAPlace()) {
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx =
        dynamic_cast<paddle::platform::CUDADeviceContext*>(pool.Get(place));
    auto stream = dev_ctx->stream();

    paddle::memory::Copy(paddle::platform::CPUPlace(), host_data.data(),
                         paddle::platform::CUDAPlace(), t_ptr,
                         sizeof(float) * tensor->numel(), stream);
    t_ptr = host_data.data();
  }
  VLOG(6) << "Tensor Value: " << t_ptr[0] << ", Expected Value: " << value;
  PADDLE_ENFORCE(t_ptr[0] == value, paddle::platform::errors::Fatal(
                                        "Numerical Error, Expected %f", value));
}

static void FluidCheckGradTensorValue(
    const std::shared_ptr<imperative::VarBase>& X,
    const paddle::platform::Place& place, float value) {
  auto* grad_tensor = X->MutableGradVar()->GetMutable<framework::LoDTensor>();
  float* g_ptr = grad_tensor->mutable_data<float>(place);
  std::vector<float> g_host_data(grad_tensor->numel());
  if (place == paddle::platform::CUDAPlace()) {
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx =
        dynamic_cast<paddle::platform::CUDADeviceContext*>(pool.Get(place));
    auto stream = dev_ctx->stream();

    paddle::memory::Copy(paddle::platform::CPUPlace(), g_host_data.data(),
                         paddle::platform::CUDAPlace(), g_ptr,
                         sizeof(float) * grad_tensor->numel(), stream);
    g_ptr = g_host_data.data();
  }
  VLOG(6) << "Tensor Value: " << g_ptr[0] << ", Expected Value: " << value;
  PADDLE_ENFORCE(g_ptr[0] == value, paddle::platform::errors::Fatal(
                                        "Numerical Error, Expected %f", value));
}

/* --------------------- */
/* ---- Fluid Scale ---- */
/* --------------------- */
// TODO(jiabin): Change this and remove nolint
void benchmark_fluid_scale(const std::shared_ptr<imperative::VarBase>& X,
                           const paddle::platform::Place& place,
                           bool accuracy_check) {
  imperative::Tracer tracer;
  framework::AttributeMap attrs;

  attrs["use_mkldnn"] = false;
  attrs["scale"] = 2;
  attrs["bias"] = 3;
  attrs["bias_after_scale"] = true;

  std::shared_ptr<imperative::VarBase> tmp_out = X;

  size_t max_num_runs = accuracy_check ? 10 : max_num_benchmark_runs;
  for (size_t i = 0; i < max_num_runs; i++) {
    imperative::NameVarBaseMap ins = {{"X", {tmp_out}}};
    imperative::NameVarBaseMap outs = {
        {"Out",
         {std::shared_ptr<imperative::VarBase>(
             new imperative::VarBase(true, "Out"))}}};

    tracer.TraceOp("scale", ins, outs, attrs, place, true);

    tmp_out = outs["Out"][0];
  }

  auto* engine = tracer.GetEngine();
  std::vector<std::shared_ptr<imperative::VarBase>> grad_tensors{nullptr};
  engine->Init({tmp_out}, grad_tensors, false /*retain_graph*/);
  engine->Execute();

  if (accuracy_check) {
    FluidCheckTensorValue(tmp_out, place, 8189.0);
    FluidCheckGradTensorValue(X, place, 1024.0);
  }
}

/* ---------------------- */
/* ---- Fluid Matmul ---- */
/* ---------------------- */
void benchmark_fluid_matmul(const std::shared_ptr<imperative::VarBase>& X,
                            const std::shared_ptr<imperative::VarBase>& Y,
                            const paddle::platform::Place& place,
                            bool accuracy_check) {
  imperative::Tracer tracer;

  std::shared_ptr<imperative::VarBase> tmp_out = X;

  size_t max_num_runs = accuracy_check ? 2 : max_num_benchmark_runs;
  for (size_t i = 0; i < max_num_runs; i++) {
    framework::AttributeMap attrs;
    imperative::NameVarBaseMap ins = {{"X", {tmp_out}}, {"Y", {Y}}};
    imperative::NameVarBaseMap outs = {
        {"Out",
         {std::shared_ptr<imperative::VarBase>(
             new imperative::VarBase(true, "Out"))}}};

    tracer.TraceOp("matmul_v2", ins, outs, attrs, place, true);

    tmp_out = outs["Out"][0];
  }

  auto* engine = tracer.GetEngine();
  std::vector<std::shared_ptr<imperative::VarBase>> grad_tensors{nullptr};
  engine->Init({tmp_out}, grad_tensors, false /*retain_graph*/);
  engine->Execute();

  if (accuracy_check) {
    FluidCheckTensorValue(tmp_out, place, 16);
    FluidCheckGradTensorValue(X, place, 16);
    FluidCheckGradTensorValue(Y, place, 16);
  }
}

/* ------------------- */
/* ---- Fluid MLP ---- */
/* ------------------- */
void benchmark_fluid_mlp(const std::shared_ptr<imperative::VarBase>& X,
                         const std::shared_ptr<imperative::VarBase>& W1,
                         const std::shared_ptr<imperative::VarBase>& W2,
                         const paddle::platform::Place& place,
                         bool accuracy_check) {
  imperative::Tracer tracer;

  // Matmul0
  framework::AttributeMap attrs;
  imperative::NameVarBaseMap ins = {{"X", {X}}, {"Y", {W1}}};
  imperative::NameVarBaseMap outs = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(true, "Out"))}}};

  tracer.TraceOp("matmul_v2", ins, outs, attrs, place, true);

  // Matmul1
  ins = {{"X", outs["Out"]}, {"Y", {W2}}};
  outs = {{"Out",
           {std::shared_ptr<imperative::VarBase>(
               new imperative::VarBase(true, "Out"))}}};

  tracer.TraceOp("matmul_v2", ins, outs, attrs, place, true);

  // ReduceSum
  ins = {{"X", outs["Out"]}};
  outs = {{"Out",
           {std::shared_ptr<imperative::VarBase>(
               new imperative::VarBase(true, "Out"))}}};
  attrs = {{"reduce_all", true}};

  tracer.TraceOp("reduce_sum", ins, outs, attrs, place, true);

  auto* engine = tracer.GetEngine();
  std::vector<std::shared_ptr<imperative::VarBase>> grad_tensors{nullptr};
  engine->Init(outs["Out"], grad_tensors, false /*retain_graph*/);
  engine->Execute();

  if (accuracy_check) {
    FluidCheckTensorValue(outs["Out"][0], place, 786432);
    FluidCheckGradTensorValue(X, place, 12288);
    FluidCheckGradTensorValue(W1, place, 768);
    FluidCheckGradTensorValue(W2, place, 128);
  }
}

}  // namespace imperative
}  // namespace paddle
