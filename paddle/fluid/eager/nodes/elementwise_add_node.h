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
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/tensor_wrapper.h"

class GradNodeElementwiseAdd : public egr::GradNodeBase {
 public:
  GradNodeElementwiseAdd() : egr::GradNodeBase() {}
  GradNodeElementwiseAdd(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~GradNodeElementwiseAdd() override = default;

  virtual std::vector<std::vector<egr::EagerTensor>> operator()(
      const std::vector<std::vector<egr::EagerTensor>>& grads) override;

  // SetX, SetY, ...
  void SetTensorWrapperX(const egr::EagerTensor& X) {
    X_ = egr::TensorWrapper(X, true /*full_reserved*/);
  }
  void SetTensorWrapperY(const egr::EagerTensor& Y) {
    Y_ = egr::TensorWrapper(Y, true /*full_reserved*/);
  }

  void SetAttributes_axis(const int axis) { axis_ = axis; }

 private:
  // TensorWrappers
  egr::TensorWrapper X_;
  egr::TensorWrapper Y_;

  // Attribute Members
  int axis_ = -1;
};
