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

#include "paddle/pten/hapi/include/grad_linalg.h"

#include <memory>

#include "glog/logging.h"

#include "paddle/pten/api/include/core.h"
#include "paddle/pten/api/include/infershape.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/hapi/lib/kernel_dispatch.h"
#include "paddle/pten/infershape/binary.h"

PT_DECLARE_MODULE(GradLinalgCPU);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_MODULE(GradLinalgCUDA);
#endif

namespace paddle {
namespace experimental {

std::vector<Tensor> grad_matmul(const Tensor& x,
                                const Tensor& y,
                                const Tensor& grad_out,
                                bool transpose_x,
                                bool transpose_y) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(grad_out);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "grad_matmul", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(*dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  auto dense_y = std::dynamic_pointer_cast<pten::DenseTensor>(y.impl());
  kernel_context.EmplaceBackInput(dense_y);
  auto dense_grad_out =
      std::dynamic_pointer_cast<pten::DenseTensor>(grad_out.impl());
  kernel_context.EmplaceBackInput(dense_grad_out);

  kernel_context.EmplaceBackAttr(transpose_x);
  kernel_context.EmplaceBackAttr(transpose_y);
  // TODO(chenweihang): add transform impl

  // 5. Prepare outputs
  Tensor grad_x_out;
  Tensor grad_y_out;
  // TODO(chenweihang): deal with multiple outputs
  auto dense_grad_x_out = std::make_shared<pten::DenseTensor>(
      dense_x->meta(), pten::TensorStatus());
  auto dense_grad_y_out = std::make_shared<pten::DenseTensor>(
      dense_y->meta(), pten::TensorStatus());

  kernel_context.EmplaceBackOutput(dense_grad_x_out);
  kernel_context.EmplaceBackOutput(dense_grad_y_out);

  grad_x_out.set_impl(dense_grad_x_out);
  grad_y_out.set_impl(dense_grad_y_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return {grad_x_out, grad_y_out};
}

}  // namespace experimental
}  // namespace paddle
