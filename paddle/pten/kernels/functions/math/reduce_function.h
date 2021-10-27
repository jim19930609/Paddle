/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/complex_functors.h"

#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/functions/eigen/common.h"
#include "paddle/pten/kernels/functions/math/transform_function.h"
#include "paddle/pten/kernels/cuda/utils.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"
#endif

namespace pten {
namespace math {

struct SumFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->sum(dim);
  }
};

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename Tx, typename Ty, typename ReduceOp, typename TransformOp>
void TensorReduce(const paddle::platform::CUDADeviceContext& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* y,
                  std::vector<int> origin_reduce_dims,
                  const Ty& init,
                  const ReduceOp& reducer,
                  const TransformOp& transformer) {
  cudaStream_t stream = dev_ctx.stream();

  auto x_dim = paddle::framework::vectorize<int>(x.dims());
  std::vector<int> new_x_dim, new_reduce_dims;
  int is_reduced = 0;
  for (auto e : origin_reduce_dims) {
    auto pos = e >= 0 ? e : e + x_dim.size();
    is_reduced |= 1 << e;
  }
  for (int i = 0; i < x_dim.size(); i++) {
    if ((i == 0) || (((is_reduced >> i) ^ (is_reduced >> (i - 1))) & 1)) {
      new_x_dim.push_back(x_dim[i]);
      if ((is_reduced >> i) & 1)
        new_reduce_dims.push_back(new_x_dim.size() - 1);
    } else {
      new_x_dim[new_x_dim.size() - 1] *= x_dim[i];
    }
  }
  x_dim = new_x_dim;
  origin_reduce_dims = new_reduce_dims;
  int x_rank = static_cast<int>(x_dim.size());
  std::set<int> left_set, reduce_set;
  for (int i = 0; i < x_rank; ++i) left_set.insert(i);

  for (auto e : origin_reduce_dims) {
    left_set.erase(e);
    reduce_set.insert(e);
  }

  std::vector<int> reduce_dim(reduce_set.begin(), reduce_set.end());
  std::vector<int> left_dim(left_set.begin(), left_set.end());

  std::vector<int> x_strides = paddle::operators::detail::GetStrides(x_dim);
  std::vector<int> reduce_strides =
      paddle::operators::detail::GetStrides(x_dim, reduce_dim);
  std::vector<int> left_strides =
      paddle::operators::detail::GetStrides(x_dim, left_dim);
  int reduce_num = reduce_strides[0] * x_dim[reduce_dim[0]];
  int left_num = 1;
  if (left_dim.size()) left_num = left_strides[0] * x_dim[left_dim[0]];

  std::vector<int> y_dim(left_dim.size());
  for (int i = 0; i < left_dim.size(); ++i) {
    y_dim[i] = x_dim[left_dim[i]];
  }
  auto x_data = x.data<Tx>();
  auto y_data = y->mutable_data<Ty>();
  if (reduce_num == 1) {
    auto out_dims = y->dims();
    pten::Copy(dev_ctx, x, y);
    y->Resize(out_dims);
    return;
  }

#define CUB_BLOCK_DIM_CASE(block_dim)                               \
  case block_dim: {                                                 \
    constexpr auto kBlockDim = block_dim;                           \
    paddle::operators::detail::                                     \
        TensorReduceImpl<Tx, Ty, block_dim, ReduceOp, TransformOp>( \
            x_data,                                                 \
            y_data,                                                 \
            x.place(),                                              \
            reducer,                                                \
            transformer,                                            \
            init,                                                   \
            left_num,                                               \
            reduce_num,                                             \
            x_strides,                                              \
            reduce_dim,                                             \
            reduce_strides,                                         \
            left_dim,                                               \
            left_strides,                                           \
            stream);                                                \
  } break

  switch (paddle::operators::detail::GetDesiredBlockDim(reduce_num)) {
    CUB_BLOCK_DIM_CASE(512);
    CUB_BLOCK_DIM_CASE(256);
    CUB_BLOCK_DIM_CASE(128);
    CUB_BLOCK_DIM_CASE(64);
    CUB_BLOCK_DIM_CASE(32);
    CUB_BLOCK_DIM_CASE(16);
    CUB_BLOCK_DIM_CASE(8);
    CUB_BLOCK_DIM_CASE(4);
    CUB_BLOCK_DIM_CASE(2);
  }
#undef CUB_BLOCK_DIM_CASE
}
#endif

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
  auto x = pten::EigenTensor<T, D>::From(input);
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
    auto out = pten::EigenScalar<T>::From(*output);
    functor(place, &x, &out, reduce_dim);
  } else {
    auto out = pten::EigenTensor<T, (D - R_D)>::From(*output, out_dims);
    functor(place, &x, &out, reduce_dim);
  }
}

#define HANDLE_DIM(NDIM, RDIM)                               \
  if (ndim == NDIM && rdim == RDIM) {                        \
    ReduceFunctor<DeviceContext, OutT, NDIM, RDIM, Functor>( \
        dev, input, output, dim, keep_dim);                  \
  }

inline void GetShuffledDim(const DDim& src_dims,
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
void GetShuffledInput(const DeviceContext& dev_ctx,
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
void HandleLargeDim(const DeviceContext& dev_ctx,
                    const DenseTensor& input,
                    DenseTensor* output,
                    const std::vector<int>& dims,
                    bool keep_dim) {
  //  shuffle the reduced dim to the end
  DenseTensor shuffled_input(input.meta(), pten::TensorStatus());
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
      auto x = pten::EigenVector<OutT>::Flatten(input);
      auto out = pten::EigenScalar<OutT>::From(*output);
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
void ReduceKernel(const DeviceContext& dev_ctx,
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
        TransToProtoVarType(x.data_type());
    paddle::framework::VisitDataType(
        cast_out_dtype,
        ReduceKernelFunctor<DeviceContext, T, Functor>(
            dev_ctx, x, dim, keep_dim, reduce_all, out));
  } else {
    DenseTensor tmp_tensor(x.meta(), TensorStatus());

    auto cast_out_dtype =
        static_cast<paddle::framework::proto::VarType::Type>(out_dtype);

    tmp_tensor.Resize(x.dims());
    paddle::framework::VisitDataType(
        cast_out_dtype,
        CastOpFunctor<DeviceContext, T>(x, &tmp_tensor, dev_ctx));
    paddle::framework::VisitDataType(
        cast_out_dtype,
        ReduceKernelFunctor<DeviceContext, T, Functor>(
            dev_ctx, tmp_tensor, dim, keep_dim, reduce_all, out));
  }
}

}  // namespace math
}  // namespace pt
