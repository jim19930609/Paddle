// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

// only can include the headers in paddle/pten/api dirs
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/kernels/complex_kernel.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ConjKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    out->mutable_data<T>(context.GetPlace(), size_t(x->numel() * sizeof(T)));

    auto& dev_ctx = context.device_context<DeviceContext>();
    auto pt_x = paddle::experimental::MakePtenDenseTensor(*x);
    auto pt_out = paddle::experimental::MakePtenDenseTensor(*out);

    // call new kernel
    pten::ConjKernel<T>(dev_ctx, *pt_x.get(), pt_out.get());
  }
};

DECLARE_INPLACE_OP_INFERER(ConjOpInplaceInferer, {"X", "Out"});

}  // namespace operators
}  // namespace paddle
