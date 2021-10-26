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

#include "paddle/tcmpt/kernels/cpu/grad_linalg.h"

#include "paddle/tcmpt/core/kernel_registry.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/eigen.h"
// #include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/tcmpt/kernels/common/math/grad_matmul_function.h"

namespace pt {

template <typename T>
void grad_matmul(const CPUContext& dev_ctx,
                 const DenseTensor& X,
                 const DenseTensor& Y,
                 const DenseTensor& GradOut,
                 bool transpose_x,
                 bool transpose_y,
                 DenseTensor* GradX,
                 DenseTensor* GradY) {
  math::GradMatMulFunction<CPUContext, T>(
      dev_ctx, X, Y, GradOut, transpose_x, transpose_y, GradX, GradY);
}

}  // namespace pt

PT_REGISTER_MODULE(GradLinalgCPU);

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_KERNEL("grad_matmul",
                   CPU,
                   NCHW,
                   pt::grad_matmul,
                   float,
                   double,
                   complex64,
                   complex128) {}
