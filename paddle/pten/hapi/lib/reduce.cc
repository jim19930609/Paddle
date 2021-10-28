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

#include "paddle/pten/hapi/include/reduce.h"

#include <memory>

#include "glog/logging.h"

#include "paddle/pten/api/include/core.h"
#include "paddle/pten/api/include/infershape.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/hapi/lib/kernel_dispatch.h"
#include "paddle/pten/infershape/unary.h"

PT_DECLARE_MODULE(ReduceCPU);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_MODULE(ReduceCUDA);
#endif

namespace paddle {
namespace experimental {

Tensor reduce_sum(const Tensor& x,
                  bool reduce_all,
                  const std::vector<int>& dim,
                  bool keep_dim,
                  int out_dtype) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "reduce_sum", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(*dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);

  kernel_context.EmplaceBackAttr(reduce_all);
  kernel_context.EmplaceBackAttr(dim);
  kernel_context.EmplaceBackAttr(keep_dim);
  kernel_context.EmplaceBackAttr(out_dtype);

  // TODO(chenweihang): add transform impl

  // 4. InferShape
  // TODO(chenweihang): how to auto selected infershape?
  auto out_meta = GenericReductionInferShape(
      dense_x->meta(), reduce_all, dim, keep_dim, out_dtype);

  // 5. Prepare outputs
  Tensor out;
  // TODO(chenweihang): deal with multiple outputs
  auto dense_out =
      std::make_shared<pten::DenseTensor>(out_meta, pten::TensorStatus());
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

}  // namespace experimental
}  // namespace paddle
