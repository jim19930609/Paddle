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

/**
 * This File should be automatically generated by coding auto generator.
 * All ops C++ autograd logic is defined here, in Python-C extension API
 * system we try to avoid any autograd related code, and move them all to
 * here.
 *
 * Currently, we just manually do some fwd autograd here. And we will replace
 * them with auto code generator later.
 * **/

#include "glog/logging.h"

#include "paddle/fluid/eager/autograd_meta.h"

#include "paddle/fluid/eager/nodes/matmul_v2_node.h"
#include "paddle/fluid/eager/nodes/reduce_sum_node.h"
#include "paddle/fluid/eager/nodes/scale_node.h"

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/function_api.h"
#include "paddle/pten/api/all.h"
#include "paddle/pten/hapi/all.h"

namespace egr {

egr::EagerTensor scale(const egr::EagerTensor& x, float scale, float bias,
                       bool bias_after_scale, bool trace_backward) {
  // 1. Run Forward
  // 1.1 Create outputs
  egr::EagerTensor out;
  // 1.2 Need by original op, we assemble ins, outs, attrs here

  // 1.3 Call forward C++ api
  ScaleAPI(x, scale, bias, bias_after_scale, &out);

  // 2. Build Backward Depends
  // 2.1 Get AutogradMetas for all ins and outs
  auto p_autograd_in = EagerUtils::unsafe_autograd_meta(x);
  // NOTE: Call EagerUtils::multi_autograd_meta  when we have vector of outputs
  auto p_autograd_out = EagerUtils::autograd_meta(&out);

  // 2.2 Add GradNode
  // 2.2.1 ComputeRequireGrad
  // TODO(jiabin) : make this function accept different kinds of input
  // TODO(zhanlve): which one is more efficient:
  //                1. construct a vector of pointers
  //                2. call "ComputeRequireGrad" multiple times
  bool require_any_grad = ComputeRequireGrad(trace_backward, p_autograd_in);
  if (require_any_grad) {
    PassStopGradient(false /*generate_grad*/, p_autograd_out);

    // 2.2.2 Set OutRankInfo for outputs this needs to be as same as Edges's
    // input_rank_
    /** Note:
    // 1. We provide EagerUtils::SetMultiOutRank(vector<AutogradMeta*>),
    // since we have some of Operator has servel slot name with duplicate
    outputs.
    // 2. We call AutogradMeta's SetOutput Rank only when we have single output
    with
    // single slot name.
    **/
    p_autograd_out->SetSingleOutRankWithSlot(0, 0);

    // Init GradNode
    auto scale_node = std::make_shared<GradNodeScale>(/* bwd_in_slot_num */ 1,
                                                      /* bwd_out_slot_num */ 1);

    // Pass Attributes to GradNode
    scale_node->SetAttributes_scale(scale);

    // Set Next Edges
    scale_node->AddEdges(*p_autograd_in, /*slot id*/ 0);

    // Set TensorWrappers
    scale_node->SetTensorWrappers_X({x});

    // Set Grad out rank as same as fwd input and set stop gradient to bwd
    scale_node->SetGradOutMeta(*p_autograd_in, /*slot id*/ 0);
    // Set Grad out rank as same as fwd input and set stop gradient to bwd
    scale_node->SetGradInMeta(*p_autograd_out, /*slot id*/ 0);

    // Set History for output set current Grad Node for
    EagerUtils::SetHistory(p_autograd_out, scale_node);
  }

  return out;
}

egr::EagerTensor matmul(const egr::EagerTensor& x, const egr::EagerTensor& y,
                        bool transpose_x, bool transpose_y,
                        bool trace_backward) {
  // 1. Run Forward
  egr::EagerTensor out;

  const std::shared_ptr<paddle::experimental::Tensor>& x_tensor = x.Tensor();
  const std::shared_ptr<paddle::experimental::Tensor>& y_tensor = y.Tensor();
  PADDLE_ENFORCE(x_tensor != nullptr,
                 paddle::platform::errors::Fatal(
                     "Underlying member \"tensor_\" of Input X is Null"));
  PADDLE_ENFORCE(y_tensor != nullptr,
                 paddle::platform::errors::Fatal(
                     "Underlying member \"tensor_\" of Input Y is Null"));
  paddle::experimental::Tensor out_tensor = paddle::experimental::matmul(
      *x_tensor.get(), *y_tensor.get(), transpose_x, transpose_y);
  out.set_tensor(std::make_shared<paddle::experimental::Tensor>(out_tensor));

  // 2. Build Backward Depends
  // 2.1 Get AutogradMetas for all ins and outs
  auto p_autograd_x = EagerUtils::unsafe_autograd_meta(x);
  auto p_autograd_y = EagerUtils::unsafe_autograd_meta(y);

  // NOTE: Call EagerUtils::multi_autograd_meta  when we have vector of outputs
  auto p_autograd_out = EagerUtils::autograd_meta(&out);

  // 2.2 Add GradNode
  // 2.2.1 ComputeRequireGrad
  // TODO(jiabin) : make this function accept different kinds of input
  // TODO(zhanlve): which one is more efficient:
  //                1. construct a vector of pointers
  //                2. call "ComputeRequireGrad" multiple times
  bool require_any_grad =
      ComputeRequireGrad(trace_backward, p_autograd_x, p_autograd_y);
  if (require_any_grad) {
    PassStopGradient(false /*generate_grad*/, p_autograd_out);

    // 2.2.2 Set OutRankInfo for outputs this needs to be as same as Edges's
    // input_rank_
    /** Note:
    // 1. We provide EagerUtils::SetMultiOutRank(vector<AutogradMeta*>),
    // since we have some of Operator has servel slot name with duplicate
    outputs.
    // 2. We call AutogradMeta's SetOutput Rank only when we have single output
    with
    // single slot name.
    **/

    // Init GradNode
    auto matmul_node =
        std::make_shared<GradNodeMatmul>(/* bwd_in_slot_num */ 1,
                                         /* bwd_out_slot_num */ 2);

    // Pass Attributes to GradNode
    matmul_node->SetAttributes_transpose_x(transpose_x);
    matmul_node->SetAttributes_transpose_y(transpose_y);

    // Set Next Edges
    matmul_node->AddEdges(*p_autograd_x, /*slot id*/ 0);
    matmul_node->AddEdges(*p_autograd_y, /*slot id*/ 1);

    // Set TensorWrappers
    matmul_node->SetTensorWrapperX(x);
    matmul_node->SetTensorWrapperY(y);

    // Set Grad out rank as same as fwd input and set stop gradient to bwd
    matmul_node->SetGradOutMeta(*p_autograd_x, /*slot id*/ 0);
    matmul_node->SetGradOutMeta(*p_autograd_y, /*slot id*/ 1);
    // Set Grad out rank as same as fwd input and set stop gradient to bwd
    matmul_node->SetGradInMeta(*p_autograd_out, /*slot id*/ 0);
    p_autograd_out->SetSingleOutRankWithSlot(0, 0);

    // Set History for output set current Grad Node for
    EagerUtils::SetHistory(p_autograd_out, matmul_node);
  }

  return out;
}

egr::EagerTensor reduce_sum(const egr::EagerTensor& x,
                            const std::vector<int>& dim, const bool keep_dim,
                            const bool reduce_all, const int in_dtype,
                            const int out_dtype, bool trace_backward) {
  // 1. Run Forward
  egr::EagerTensor Out;

  const std::shared_ptr<paddle::experimental::Tensor>& x_tensor = x.Tensor();
  PADDLE_ENFORCE(x_tensor != nullptr,
                 paddle::platform::errors::Fatal(
                     "Underlying member \"tensor_\" of Input X is Null"));
  paddle::experimental::Tensor out_tensor = paddle::experimental::reduce_sum(
      *x_tensor.get(), reduce_all, dim, keep_dim, out_dtype);
  Out.set_tensor(std::make_shared<paddle::experimental::Tensor>(out_tensor));

  // Prepare Autograd Meta
  egr::AutogradMeta& p_autograd_X = *egr::EagerUtils::unsafe_autograd_meta(x);
  egr::AutogradMeta& p_autograd_Out = *egr::EagerUtils::autograd_meta(&Out);

  bool require_any_grad =
      egr::ComputeRequireGrad(trace_backward, &p_autograd_X);
  if (require_any_grad) {
    egr::PassStopGradient(false, &p_autograd_Out);
    // Create GradOpNode
    auto grad_node = std::make_shared<GradNodeReduceSum>(1, 1);

    // Set Attributes
    grad_node->SetAttributes_in_dtype(in_dtype);
    grad_node->SetAttributes_reduce_all(reduce_all);
    grad_node->SetAttributes_dim(dim);

    // Set Tensor Wrappers
    grad_node->SetTensorWrapperX(x);
    grad_node->SetTensorWrapperOut(Out);

    grad_node->SetGradOutMeta(p_autograd_X, 0);
    grad_node->AddEdges(p_autograd_X, 0);
    grad_node->SetGradInMeta(p_autograd_Out, 0);
    egr::EagerUtils::SetOutRankWithSlot(&p_autograd_Out, 0);
    egr::EagerUtils::SetHistory(&p_autograd_Out, grad_node);
  }

  return Out;
}

}  // namespace egr
