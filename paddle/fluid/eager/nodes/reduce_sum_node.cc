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

#include "paddle/fluid/eager/nodes/reduce_sum_node.h"
#include "paddle/fluid/eager/function_api.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/pten/api/all.h"

std::vector<std::vector<egr::EagerTensor>> GradNodeReduceSum::operator()(
    const std::vector<std::vector<egr::EagerTensor>>& grads) {
  // 1. Check Output Size
  PADDLE_ENFORCE(((grads.size() == 1) && (grads[0].size() == 1)),
                 paddle::platform::errors::Fatal(
                     "MatmulGradNode should take exactly 1 grad tensor"
                     "However received: %d",
                     grads.size()));

  egr::EagerTensor grad_x;

  // 2. Create needed out parttern
  // Apply Gradient Hooks
  if (GradientHooksRegistered()) {
    // TODO(jiabin): Shall we apply hook slot by slot here or accept
    // vector<vector<pten::tensor>> to apply all hooks?
    std::vector<std::vector<egr::EagerTensor>> hooked_grads =
        ApplyGradientHooks(grads);

    const std::shared_ptr<paddle::experimental::Tensor>& x_tensor =
        X_.recover(nullptr).Tensor();
    const std::shared_ptr<paddle::experimental::Tensor>& out_tensor =
        Out_.recover(nullptr).Tensor();
    const std::shared_ptr<paddle::experimental::Tensor>& hooked_grad =
        hooked_grads[0][0].Tensor();
    PADDLE_ENFORCE(
        x_tensor != nullptr,
        paddle::platform::errors::Fatal(
            "Underlying member \"tensor_\" of InputWrapper X is Null"));
    PADDLE_ENFORCE(
        out_tensor != nullptr,
        paddle::platform::errors::Fatal(
            "Underlying member \"tensor_\" of InputWrapper Out is Null"));
    PADDLE_ENFORCE(hooked_grad != nullptr,
                   paddle::platform::errors::Fatal(
                       "Underlying member \"tensor_\" of hooked_grad is Null"));

    paddle::experimental::Tensor grad_tensor =
        paddle::experimental::grad_reduce_sum(
            *x_tensor.get(), *out_tensor.get(), *hooked_grad.get(), reduce_all_,
            dim_, in_dtype_);

    grad_x.set_tensor(
        std::make_shared<paddle::experimental::Tensor>(grad_tensor));

  } else {
    const std::shared_ptr<paddle::experimental::Tensor>& x_tensor =
        X_.recover(nullptr).Tensor();
    const std::shared_ptr<paddle::experimental::Tensor>& out_tensor =
        Out_.recover(nullptr).Tensor();
    const std::shared_ptr<paddle::experimental::Tensor>& grad =
        grads[0][0].Tensor();
    PADDLE_ENFORCE(
        x_tensor != nullptr,
        paddle::platform::errors::Fatal(
            "Underlying member \"tensor_\" of InputWrapper X is Null"));
    PADDLE_ENFORCE(
        out_tensor != nullptr,
        paddle::platform::errors::Fatal(
            "Underlying member \"tensor_\" of InputWrapper Out is Null"));
    PADDLE_ENFORCE(grad != nullptr,
                   paddle::platform::errors::Fatal(
                       "Underlying member \"tensor_\" of grad is Null"));

    paddle::experimental::Tensor grad_tensor =
        paddle::experimental::grad_reduce_sum(*x_tensor.get(),
                                              *out_tensor.get(), *grad.get(),
                                              reduce_all_, dim_, in_dtype_);

    grad_x.set_tensor(
        std::make_shared<paddle::experimental::Tensor>(grad_tensor));
  }

  // Apply Reduce Hooks
  if (ReduceHooksRegistered()) {
    ApplyReduceHooks();
  }

  return {{grad_x}};
}
