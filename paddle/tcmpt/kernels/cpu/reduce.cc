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

#include "paddle/fluid/platform/complex.h"

namespace pt {

template <typename DeviceContext, typename T>
struct TransposeNormal {
  void operator()(const DeviceContext& context,
                  const DenseTensor& in,
                  DenseTensor* out,
                  const std::vector<int>& axis) {
    const int rank = axis.size();
    auto in_stride = paddle::framework::stride(in.dims());
    auto out_stride = paddle::framework::stride(out->dims());
    const T* in_ptr = in.data<T>();
    T* out_ptr = out->mutable_data<T>();

    auto transpose_helper = [&](int64_t beg, int64_t end) {
      for (int64_t out_idx = beg; out_idx < end; ++out_idx) {
        int64_t in_idx = 0;
        int64_t tmp_idx = out_idx;
        // calculate the input index
        for (int i = 0; i < rank; ++i) {
          const int64_t coordinate = tmp_idx / out_stride[i];
          tmp_idx -= coordinate * out_stride[i];
          in_idx += coordinate * in_stride[axis[i]];
        }
        out_ptr[out_idx] = in_ptr[in_idx];
      }
    };
    transpose_helper(0, out->numel());
  }
};

// define transpose normal
#define DEFINE_CPU_TRANS_NORMAL(TYPE) \
  template struct TransposeNormal<paddle::platform::CPUDeviceContext, TYPE>

DEFINE_CPU_TRANS_NORMAL(paddle::platform::float16);
DEFINE_CPU_TRANS_NORMAL(paddle::platform::bfloat16);
DEFINE_CPU_TRANS_NORMAL(float);
DEFINE_CPU_TRANS_NORMAL(double);
DEFINE_CPU_TRANS_NORMAL(int);
DEFINE_CPU_TRANS_NORMAL(int64_t);
DEFINE_CPU_TRANS_NORMAL(bool);
DEFINE_CPU_TRANS_NORMAL(int16_t);
DEFINE_CPU_TRANS_NORMAL(uint8_t);
DEFINE_CPU_TRANS_NORMAL(int8_t);
DEFINE_CPU_TRANS_NORMAL(paddle::platform::complex<float>);
DEFINE_CPU_TRANS_NORMAL(paddle::platform::complex<double>);

template <typename DeviceContext,
          typename T,
          size_t D,
          size_t R_D,
          typename Functor>
void ReduceFunctor(const DeviceContext& dev_ctx,
                   const DenseTensor& input,
                   DenseTensor* output,
                   const std::vector<int>& dims,
                   bool keep_dim) {
  auto x = pt::EigenTensor<T, D>::From(input);
  auto x_rank = static_cast<int>(x.dimensions().size());
  auto reduce_dim = Eigen::array<int, R_D>();
  std::vector<int> dims_ref = dims;
  for (size_t i = 0; i < dims_ref.size(); ++i) {
    if (dims_ref[i] < 0) dims_ref[i] = x_rank + dims_ref[i];
    reduce_dim[i] = dims_ref[i];
  }
  // construct the squeezed output tensor
  DDim out_dims = output->dims();
  if (keep_dim && x_rank > 1) {
    const int kDelFlag = -2;
    auto dims_vector = paddle::framework::vectorize(out_dims);
    for (size_t i = 0; i < dims_ref.size(); ++i) {
      dims_vector[dims_ref[i]] = kDelFlag;
    }
    dims_vector.erase(remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
                      dims_vector.end());
    out_dims = paddle::framework::make_ddim(dims_vector);
  }
  auto& place = *dev_ctx.eigen_device();
  Functor functor;

  if (D == 1) {
    auto out = pt::EigenScalar<T>::From(*output);
    functor(place, &x, &out, reduce_dim);
  } else {
    auto out = pt::EigenTensor<T, (D - R_D)>::From(*output, out_dims);
    functor(place, &x, &out, reduce_dim);
  }
}

#define HANDLE_DIM(NDIM, RDIM)                               \
  if (ndim == NDIM && rdim == RDIM) {                        \
    ReduceFunctor<DeviceContext, OutT, NDIM, RDIM, Functor>( \
        dev, input, output, dim, keep_dim);                  \
  }

static void GetShuffledDim(const DDim& src_dims,
                           DDim* dst_dims,
                           const std::vector<int>& reduced_dims,
                           std::vector<int>* perm_axis) {
  // check if it's a reduced dim
  std::vector<bool> src_dims_check(src_dims.size(), false);
  size_t src_size = src_dims.size();
  size_t reduce_size = reduced_dims.size();
  for (size_t i = 0; i < reduce_size; ++i) {
    dst_dims->at(src_size - reduce_size + i) = src_dims[reduced_dims[i]];
    (*perm_axis)[src_size - reduce_size + i] = reduced_dims[i];
    src_dims_check[reduced_dims[i]] = true;
  }

  size_t offset = 0;
  for (size_t i = 0; i < src_dims_check.size(); ++i) {
    bool is_reduced = src_dims_check[i];
    if (!is_reduced) {
      (*perm_axis)[offset] = i;
      dst_dims->at(offset++) = src_dims[i];
    }
  }
}

template <typename DeviceContext, typename OutT>
static void GetShuffledInput(const DeviceContext& dev_ctx,
                             const DenseTensor& input,
                             DenseTensor* shuffled_input,
                             const std::vector<int>& dims) {
  DDim shuffled_dims(input.dims());
  std::vector<int> perm_axis(input.dims().size());
  GetShuffledDim(input.dims(), &shuffled_dims, dims, &perm_axis);

  shuffled_input->Resize(shuffled_dims);
  shuffled_input->mutable_data<OutT>();

  TransposeNormal<DeviceContext, OutT> trans;
  trans(dev_ctx, input, shuffled_input, perm_axis);
}

template <typename DeviceContext, typename OutT, typename Functor>
static void HandleLargeDim(const DeviceContext& dev_ctx,
                           const DenseTensor& input,
                           DenseTensor* output,
                           const std::vector<int>& dims,
                           bool keep_dim) {
  //  shuffle the reduced dim to the end
  DenseTensor shuffled_input(input.meta(), pt::TensorStatus());
  GetShuffledInput<DeviceContext, OutT>(dev_ctx, input, &shuffled_input, dims);

  // transpose to 2D tensor whose shape is {unreduced, reduced}.
  const int64_t unreduced = output->numel();
  const int64_t reduced = shuffled_input.numel() / unreduced;
  shuffled_input.Resize({unreduced, reduced});
  DDim output_dim = output->dims();
  output->Resize({unreduced});
  ReduceFunctor<DeviceContext, OutT, 2, 1, Functor>(
      dev_ctx, shuffled_input, output, {1}, keep_dim);
  output->Resize(output_dim);
}

template <typename DeviceContext, typename T, typename Functor>
struct ReduceKernelFunctor {
  const DeviceContext& dev;
  const DenseTensor& input;
  DenseTensor* output;
  std::vector<int> dim;
  bool keep_dim;
  bool reduce_all;
  ReduceKernelFunctor(const DeviceContext& dev_ctx,
                      const DenseTensor& input,
                      const std::vector<int>& dim,
                      bool keep_dim,
                      bool reduce_all,
                      DenseTensor* output)
      : dev(dev_ctx),
        input(input),
        output(output),
        dim(dim),
        keep_dim(keep_dim),
        reduce_all(reduce_all) {}

  template <typename OutT>
  void apply() const {
    output->mutable_data<OutT>();
    if (reduce_all) {
      // Flatten and reduce 1-D tensor
      auto x = pt::EigenVector<OutT>::Flatten(input);
      auto out = pt::EigenScalar<OutT>::From(*output);
      auto& place = *dev.eigen_device();
      auto reduce_dim = Eigen::array<int, 1>({{0}});
      Functor functor;
      functor(place, &x, &out, reduce_dim);
    } else {
      int ndim = input.dims().size();
      int rdim = dim.size();
      if (ndim > 6) {
        HandleLargeDim<DeviceContext, OutT, Functor>(
            dev, input, output, dim, keep_dim);
      } else {
        HANDLE_DIM(6, 5);
        HANDLE_DIM(6, 4);
        HANDLE_DIM(6, 3);
        HANDLE_DIM(6, 2);
        HANDLE_DIM(6, 1);
        HANDLE_DIM(5, 4);
        HANDLE_DIM(5, 3);
        HANDLE_DIM(5, 2);
        HANDLE_DIM(5, 1);
        HANDLE_DIM(4, 3);
        HANDLE_DIM(4, 2);
        HANDLE_DIM(4, 1);
        HANDLE_DIM(3, 2);
        HANDLE_DIM(3, 1);
        HANDLE_DIM(2, 1);
        HANDLE_DIM(1, 1);
      }
    }
  }
};

template <typename DeviceContext, typename T, typename Functor>
static void ReduceKernel(const DeviceContext& dev_ctx,
                         const DenseTensor& x,
                         bool reduce_all,
                         const std::vector<int>& dim,
                         bool keep_dim,
                         int out_dtype,
                         DenseTensor* out) {
  // The dims has full dim, set the reduce_all is True
  const auto& input_dim_size = x.dims().size();
  std::set<int> dims_set(dim.begin(), dim.end());
  bool full_dim = true;
  for (auto i = 0; i < input_dim_size; i++) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = (reduce_all || full_dim);

  if (out_dtype < 0) {
    paddle::framework::proto::VarType::Type cast_out_dtype =
        TransToProtoVarType(x.type());
    paddle::framework::VisitDataType(
        cast_out_dtype,
        ReduceKernelFunctor<DeviceContext, T, Functor>(
            dev_ctx, x, dim, keep_dim, reduce_all, out));
  }
  /*
  else {
    Tensor tmp_tensor;
    cast_out_dtype =
  static_cast<paddle::framework::proto::VarType::Type>(out_dtype);
    auto* input = context.Input<Tensor>("X");

    tmp_tensor.Resize(input->dim());
    paddle::framework::VisitDataType(
        cast_out_dtype,
        CastOpFunctor<DeviceContext, T>(
            input, &tmp_tensor,
            context.template device_context<DeviceContext>()));
    paddle::framework::VisitDataType(
        cast_out_dtype,
        ReduceKernelFunctor<DeviceContext, T, Functor>(
            &tmp_tensor, out, dims, keep_dim, reduce_all, context));
  }
  */
}

// ------------------------------- //
// ------ Specific Functors ------ //
// ------------------------------- //
struct SumFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->sum(dim);
  }
};

template <typename T>
void ReduceSum(const CPUContext& dev_ctx,
               const DenseTensor& x,
               bool reduce_all,
               // const std::vector<int>& dim,
               bool keep_dim,
               int out_dtype,
               DenseTensor* out) {
  std::vector<int> dim(1);
  ReduceKernel<CPUContext, T, SumFunctor>(
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
