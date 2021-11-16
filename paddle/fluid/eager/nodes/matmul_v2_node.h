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

class GradNodeMatmul : public egr::GradNodeBase {
 public:
  GradNodeMatmul() : egr::GradNodeBase() {}
  GradNodeMatmul(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~GradNodeMatmul() override = default;

  virtual std::vector<std::vector<egr::EagerTensor>> operator()(
      const std::vector<std::vector<egr::EagerTensor>>& grads) override;

  // SetX, SetY, ...
  void SetTensorWrapperX(const egr::EagerTensor& X) {
    X_ = egr::TensorWrapper(X, true /*full_reserved*/);
  }
  void SetTensorWrapperY(const egr::EagerTensor& Y) {
    Y_ = egr::TensorWrapper(Y, true /*full_reserved*/);
  }

  // SetAttr0, SetAttr1, ...
  void SetAttributes_transpose_x(const bool transpose_x) {
    transpose_x_ = transpose_x;
  }

  void SetAttributes_transpose_y(const bool transpose_y) {
    transpose_y_ = transpose_y;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper X_;
  egr::TensorWrapper Y_;

  // Attribute Members
  bool transpose_x_ = 0;
  bool transpose_y_ = 0;
};
