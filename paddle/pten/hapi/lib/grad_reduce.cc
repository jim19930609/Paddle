/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/hapi/include/grad_reduce.h"

#include <memory>

#include "glog/logging.h"

#include "paddle/pten/api/include/core.h"
#include "paddle/pten/api/include/infershape.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_context.h"
#include "paddle/pten/hapi/lib/kernel_dispatch.h"
#include "paddle/pten/infershape/unary.h"

namespace paddle {
namespace experimental {

Tensor grad_reduce_sum(const Tensor& x,
                       const Tensor& out,
                       const Tensor& grad_out,
                       bool reduce_all,
                       const std::vector<int>& dim,
                       int in_dtype) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "grad_reduce_sum", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(*dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);

  auto dense_out = std::dynamic_pointer_cast<pten::DenseTensor>(out.impl());
  kernel_context.EmplaceBackInput(dense_out);

  auto dense_grad_out =
      std::dynamic_pointer_cast<pten::DenseTensor>(grad_out.impl());
  kernel_context.EmplaceBackInput(dense_grad_out);

  kernel_context.EmplaceBackAttr(reduce_all);
  kernel_context.EmplaceBackAttr(dim);
  kernel_context.EmplaceBackAttr(in_dtype);

  // TODO(chenweihang): add transform impl

  // 4. InferShape
  // TODO(chenweihang): how to auto selected infershape?

  // 5. Prepare outputs
  Tensor grad_x;

  auto dense_grad_x =
      std::make_shared<pten::DenseTensor>(dense_x->meta(), pten::TensorStatus());

  // TODO(chenweihang): deal with multiple outputs
  kernel_context.EmplaceBackOutput(dense_grad_x);
  grad_x.set_impl(dense_grad_x);

  // 6. Call kernel
  kernel(&kernel_context);

  return grad_x;
}

}  // namespace experimental
}  // namespace paddle
