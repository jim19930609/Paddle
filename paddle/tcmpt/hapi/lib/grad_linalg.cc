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

#include "paddle/tcmpt/hapi/include/grad_linalg.h"

#include <memory>

#include "glog/logging.h"

#include "paddle/tcmpt/api/include/core.h"
#include "paddle/tcmpt/api/include/infershape.h"
#include "paddle/tcmpt/core/convert_utils.h"
#include "paddle/tcmpt/core/dense_tensor.h"
#include "paddle/tcmpt/core/kernel_context.h"
#include "paddle/tcmpt/hapi/lib/kernel_generate.h"
#include "paddle/tcmpt/infershape/binary.h"

namespace paddle {
namespace experimental {

std::vector<Tensor> grad_matmul(const Tensor& x,
                                const Tensor& y,
                                const Tensor& grad_out,
                                bool transpose_x,
                                bool transpose_y) {
  // 1. Get kernel signature and kernel
  auto kernel_signature = ParseKernelNameAndKeyByArgs("grad_matmul", grad_out);
  VLOG(1) << kernel_signature.first;
  VLOG(1) << kernel_signature.second;
  VLOG(1) << pt::KernelFactory::Instance();

  auto kernel = pt::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_signature.first, kernel_signature.second);
  VLOG(1) << kernel;

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_signature.second.backend());
  auto kernel_context = pt::KernelContext(*dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pt::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  auto dense_y = std::dynamic_pointer_cast<pt::DenseTensor>(y.impl());
  kernel_context.EmplaceBackInput(dense_y);
  auto dense_grad_out =
      std::dynamic_pointer_cast<pt::DenseTensor>(grad_out.impl());
  kernel_context.EmplaceBackInput(dense_grad_out);

  kernel_context.EmplaceBackAttr(transpose_x);
  kernel_context.EmplaceBackAttr(transpose_y);
  // TODO(chenweihang): add transform impl

  // 5. Prepare outputs
  Tensor grad_x_out;
  Tensor grad_y_out;
  // TODO(chenweihang): deal with multiple outputs
  auto dense_grad_x_out =
      std::make_shared<pt::DenseTensor>(dense_x->meta(), pt::TensorStatus());
  auto dense_grad_y_out =
      std::make_shared<pt::DenseTensor>(dense_y->meta(), pt::TensorStatus());

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
