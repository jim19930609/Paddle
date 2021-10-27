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

#include "paddle/tcmpt/kernels/cpu/reduce.h"
#include "paddle/tcmpt/core/convert_utils.h"
#include "paddle/tcmpt/kernels/common/eigen/common.h"
#include "paddle/tcmpt/kernels/common/math/reduce_function.h"

#include "paddle/fluid/platform/complex.h"

namespace pt {

// ------------------------------- //
// ------ Specific Functors ------ //
// ------------------------------- //

template <typename T>
void ReduceSum(const CPUContext& dev_ctx,
               const DenseTensor& x,
               bool reduce_all,
               // const std::vector<int>& dim,
               bool keep_dim,
               int out_dtype,
               DenseTensor* out) {
  std::vector<int> dim(1);
  math::ReduceKernel<CPUContext, T, math::SumFunctor>(
      dev_ctx, x, reduce_all, dim, keep_dim, out_dtype, out);
}

}  // namespace pt

// TODO(chenweihang): replace by better impl
PT_REGISTER_MODULE(ReduceCPU);

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_KERNEL("reduce_sum",
                   CPU,
                   NCHW,
                   pt::ReduceSum,
                   bool,
                   float,
                   double,
                   paddle::platform::bfloat16,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
