//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/tcmpt/core/convert_utils.h"
#include "paddle/tcmpt/kernels/common/eigen/common.h"
#include "paddle/tcmpt/kernels/common/math/grad_reduce_function.h"
#include "paddle/tcmpt/kernels/cuda/grad_reduce.h"

#include "paddle/fluid/platform/complex.h"

namespace pt {

template <typename T>
void GradReduceSum(const CUDAContext& dev_ctx,
                   const DenseTensor& X,
                   const DenseTensor& Out,
                   const DenseTensor& GradOut,
                   bool reduce_all,
                   // const std::vector<int>& dims,
                   int in_dtype,
                   DenseTensor* GradX) {
  std::vector<int> dims = {0};
  math::ReduceGradKernel<CUDAContext, T, math::SumGradFunctor, true> kernel;
  kernel.Compute(dev_ctx, X, Out, GradOut, reduce_all, dims, in_dtype, GradX);
}

}  // namespace pt

// TODO(chenweihang): replace by better impl
PT_REGISTER_MODULE(GradReduceCUDA);

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_KERNEL("grad_reduce_sum",
                   CUDA,
                   NCHW,
                   pt::GradReduceSum,
                   bool,
                   float,
                   double,
                   paddle::platform::bfloat16,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
