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

class GradNodeReduceSum : public egr::GradNodeBase {
 public:
  GradNodeReduceSum() : egr::GradNodeBase() {}
  GradNodeReduceSum(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~GradNodeReduceSum() override = default;

  virtual std::vector<std::vector<egr::EagerTensor>> operator()(
      const std::vector<std::vector<egr::EagerTensor>>& grads) override;

  // SetX, SetY, ...
  void SetTensorWrapperX(const egr::EagerTensor& X) {
    X_ = egr::TensorWrapper(X, true /*full_reserved*/);
  }

  void SetTensorWrapperOut(const egr::EagerTensor& Out) {
    Out_ = egr::TensorWrapper(Out, false /*full_reserved*/);
  }

  // SetAttr0, SetAttr1, ...
  void SetAttributes_in_dtype(const int in_dtype) { in_dtype_ = in_dtype; }
  void SetAttributes_reduce_all(const bool reduce_all) {
    reduce_all_ = reduce_all;
  }
  void SetAttributes_dim(const std::vector<int>& dim) { dim_ = dim; }

 private:
  // TensorWrappers
  egr::TensorWrapper X_;
  egr::TensorWrapper Out_;

  // Attribute Members
  int in_dtype_ = -1;
  bool reduce_all_ = 0;
  std::vector<int> dim_ = {0};
};
