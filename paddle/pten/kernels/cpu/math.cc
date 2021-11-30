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

#include "paddle/pten/kernels/cpu/math.h"

#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/kernels/functions/cpu/elementwise.h"
#include "paddle/pten/kernels/functions/eigen/reduce.h"
#include "paddle/pten/kernels/functions/eigen/scale.h"
#include "paddle/pten/kernels/functions/eigen/sign.h"
#include "paddle/pten/kernels/functions/general/elementwise_functor.h"
#include "paddle/pten/kernels/functions/general/reduce_impl.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"

namespace pten {

template <typename T>
void Sign(const CPUContext& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  eigen::Sign<CPUContext, T>(dev_ctx, x, out);
}

template <typename T>
void Mean(const CPUContext& dev_ctx,
          const DenseTensor& x,
          const std::vector<int64_t>& dims,
          bool keep_dim,
          bool reduce_all,
          DataType in_dtype,
          DataType out_dtype,
          DenseTensor* out) {
  pten::general::Reduce<CPUContext, T, pten::eigen::MeanFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

template <typename T>
void Scale(const CPUContext& dev_ctx,
           const DenseTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  eigen::Scale<CPUContext, T>(dev_ctx, x, scale, bias, bias_after_scale, out);
}

// TODO(chenweihang): now the ScaleTensor's dtype are same as x, so we cannot
// register its dtype def
template <typename T>
void ScaleHost(const CPUContext& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& scale,
               float bias,
               bool bias_after_scale,
               DenseTensor* out) {
  eigen::Scale<CPUContext, T>(dev_ctx,
                              x,
                              static_cast<float>(*scale.data<T>()),
                              bias,
                              bias_after_scale,
                              out);
}

template <typename T>
void ElementwiseDiv(const CPUContext& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out) {
  // allocate memory for out
  out->mutable_data<T>();
  if (x.dims() == y.dims() && std::is_floating_point<T>::value) {
    SameDimsElementwiseCompute<general::SameDimsDivFunctor<CPUContext, T>>()(
        dev_ctx, x, y, out);
  } else {
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    if (x_dims.size() >= y_dims.size()) {
      ElementwiseCompute<general::DivFunctor<T>, T>(
          dev_ctx, x, y, axis, general::DivFunctor<T>(), out);
    } else {
      ElementwiseCompute<general::InverseDivFunctor<T>, T>(
          dev_ctx, x, y, axis, general::InverseDivFunctor<T>(), out);
    }
  }
}

template <typename T>
void Sum(const CPUContext& dev_ctx,
         const DenseTensor& x,
         const std::vector<int64_t>& dims,
         bool keep_dim,
         bool reduce_all,
         DataType in_dtype,
         DataType out_dtype,
         DenseTensor* out) {
  pten::general::Reduce<CPUContext, T, pten::eigen::SumFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

// Create the definition of ElementwiseAdd
DEFINE_CPU_ELEMENTWISE_OP(Add)

// Create the definition of ElementwiseSub
DEFINE_CPU_ELEMENTWISE_OP(Sub)

// Create the definition of ElementwiseMul
DEFINE_CPU_ELEMENTWISE_OP(Mul)

}  // namespace pten

// TODO(chenweihang): replace by better impl
PT_REGISTER_MODULE(MathCPU);

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

// NOTE(chenweihang): using bfloat16 will cause redefine with xpu bfloat16
// using bfloat16 = ::paddle::platform::bfloat16;

PT_REGISTER_KERNEL("sign", CPU, ANY, pten::Sign, float, double) {}
PT_REGISTER_KERNEL("reduce_mean", CPU, ANY, pten::Mean, float, double, bool) {}
PT_REGISTER_KERNEL("scale",
                   CPU,
                   ANY,
                   pten::Scale,
                   float,
                   double,
                   paddle::platform::bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
PT_REGISTER_KERNEL("scale.host",
                   CPU,
                   ANY,
                   pten::ScaleHost,
                   float,
                   double,
                   paddle::platform::bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetBackend(pten::Backend::CPU);
}
PT_REGISTER_KERNEL("elementwise_add",
                   CPU,
                   ANY,
                   pten::ElementwiseAdd,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL("elementwise_sub",
                   CPU,
                   ANY,
                   pten::ElementwiseSub,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL("elementwise_div",
                   CPU,
                   ANY,
                   pten::ElementwiseDiv,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL("elementwise_mul",
                   CPU,
                   ANY,
                   pten::ElementwiseMul,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128) {}

PT_REGISTER_KERNEL("reduce_sum",
                   CPU,
                   ANY,
                   pten::Sum,
                   bool,
                   float,
                   double,
                   paddle::platform::float16,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}
