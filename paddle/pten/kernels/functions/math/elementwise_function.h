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

#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/platform/transform.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/functions/eigen/common.h"
#include "paddle/pten/kernels/functions/math/elementwise_function.cu.h"

namespace pten {
namespace math {

template <typename Functor,
          typename T,
          typename DeviceContext,
          typename OutType = T>
class TransformFunctor {
 public:
  TransformFunctor(const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* z,
                   const DeviceContext& dev_ctx,
                   Functor func,
                   const bool is_xsize_larger = true)
      : x_(x.data<T>()),
        y_(y.data<T>()),
        z_(z->mutable_data<OutType>()),
        nx_(x.numel()),
        dev_ctx_(dev_ctx),
        func_(func),
        is_xsize_larger_(is_xsize_larger) {
    if (is_xsize_larger_ == false) {
      nx_ = y.numel();
    }
  }

  inline void Run() const {
    paddle::platform::Transform<DeviceContext> trans;
    trans(dev_ctx_, x_, x_ + nx_, y_, z_, func_);
  }

  inline void RunRowWise(int n, int pre) const {
    paddle::platform::Transform<DeviceContext> trans;
    if (is_xsize_larger_) {
      trans(
          dev_ctx_,
          x_,
          x_ + nx_,
          paddle::operators::RowwiseTransformIterator<T, DeviceContext>(y_, n),
          z_,
          func_);
    } else {
      trans(
          dev_ctx_,
          y_,
          y_ + nx_,
          paddle::operators::RowwiseTransformIterator<T, DeviceContext>(x_, n),
          z_,
          func_);
    }
  }

  inline void RunMidWise(int n, int pre, int post) const {
    paddle::platform::Transform<DeviceContext> trans;
    if (is_xsize_larger_) {
      trans(dev_ctx_,
            x_,
            x_ + nx_,
            paddle::operators::MidWiseTransformIterator<T, DeviceContext>(
                y_, n, post),
            z_,
            func_);
    } else {
      trans(dev_ctx_,
            y_,
            y_ + nx_,
            paddle::operators::MidWiseTransformIterator<T, DeviceContext>(
                x_, n, post),
            z_,
            func_);
    }
  }

 private:
  const T* x_;
  const T* y_;
  OutType* z_;
  int64_t nx_;
  const DeviceContext& dev_ctx_;
  Functor func_;
  bool is_xsize_larger_;
};

template <typename Functor, typename T, typename OutType = T>
void CommonForwardBroadcastCPU(
    const DenseTensor& x,
    const DenseTensor& y,
    DenseTensor* z,
    int* x_dims_array,
    int* y_dims_array,
    int* out_dims_array,
    int max_dim,
    const paddle::platform::CPUDeviceContext& dev_ctx,
    Functor func,
    const bool is_xsize_larger = true) {
  std::vector<int> index_array(max_dim, 0);
  const T* x_data = x.data<T>();
  const T* y_data = y.data<T>();
  PADDLE_ENFORCE_NOT_NULL(x_data,
                          paddle::platform::errors::InvalidArgument(
                              "The input X should not be empty."));
  PADDLE_ENFORCE_NOT_NULL(y_data,
                          paddle::platform::errors::InvalidArgument(
                              "The input Y should not be empty."));
  OutType* out_data = z->mutable_data<OutType>();

  const int out_size = std::accumulate(
      out_dims_array, out_dims_array + max_dim, 1, std::multiplies<int>());
  int x_index, y_index;
  for (int out_index = 0; out_index < out_size; ++out_index) {
    x_index = paddle::operators::GetElementwiseIndex(
        x_dims_array, max_dim, index_array.data());
    y_index = paddle::operators::GetElementwiseIndex(
        y_dims_array, max_dim, index_array.data());
    if (is_xsize_larger) {
      out_data[out_index] = func(x_data[x_index], y_data[y_index]);
    } else {
      out_data[out_index] = func(y_data[y_index], x_data[x_index]);
    }

    paddle::operators::UpdateElementwiseIndexArray(
        out_dims_array, max_dim, index_array.data());
  }
}

template <typename Functor,
          typename DeviceContext,
          typename T,
          typename OutType = T>
void CommonElementwiseBroadcastForward(const DeviceContext& dev_ctx,
                                       const DenseTensor& x,
                                       const DenseTensor& y,
                                       DenseTensor* z,
                                       const paddle::framework::DDim& x_dims,
                                       const paddle::framework::DDim& y_dims,
                                       Functor func,
                                       int axis,
                                       const bool is_xsize_larger = true) {
  int max_dim = (std::max)(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis,
      0,
      paddle::platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis,
                    max_dim,
                    paddle::platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim,
                        axis));
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  paddle::operators::GetBroadcastDimsArrays(x_dims,
                                            y_dims,
                                            x_dims_array.data(),
                                            y_dims_array.data(),
                                            out_dims_array.data(),
                                            max_dim,
                                            axis);

  CommonForwardBroadcastCPU<Functor, T, OutType>(x,
                                                 y,
                                                 z,
                                                 x_dims_array.data(),
                                                 y_dims_array.data(),
                                                 out_dims_array.data(),
                                                 max_dim,
                                                 dev_ctx,
                                                 func,
                                                 is_xsize_larger);
}

// It is a common implementation to compute binary calculation with the support
// of broadcast, supporting both CPU and GPU.
// - CPU implementation cannot support the case when x needs broadcast, thus
//   this function need to be called with XxxFunctor and XxxInverseFunctor,
//   like paddle/fluid/operators/elementwise/elementwise_add_op.h#L49 - L55.
// - GPU implementation supports all the broadcast cases, thus there is no need
//   to define and call with XxxInverseFunctor.
// TODO(liuyiqun): optimize the CPU implementation to support all broadcast
// cases and avoid the need of XxxInverseFunctor.
template <typename Functor,
          typename DeviceContext,
          typename T,
          typename OutType = T>
void ElementwiseComputeEx(const DeviceContext& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          int axis,
                          Functor func,
                          DenseTensor* z) {
  if (paddle::platform::is_gpu_place(dev_ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
    std::vector<const DenseTensor*> ins = {&x, &y};
    std::vector<DenseTensor*> outs = {z};
    z->mutable_data<OutType>();

    LaunchElementwiseCudaKernel<paddle::operators::ElementwiseType::kBinary,
                                T,
                                OutType>(dev_ctx, ins, &outs, axis, func);
#endif
    return;
  }

  auto x_dims = x.dims();
  auto y_dims = y.dims();
  bool is_xsize_larger = true;
  int max_dim = x_dims.size();
  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
  }
  TransformFunctor<Functor, T, DeviceContext, OutType> functor(
      x, y, z, dev_ctx, func, is_xsize_larger);

  if (x_dims == y_dims) {
    functor.Run();
    return;
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis,
      0,
      paddle::platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis,
                    max_dim,
                    paddle::platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim,
                        axis));

  int pre, n, post, is_run_common_broadcast, axis_trim = 0;
  if (is_xsize_larger) {
    auto y_dims_trimed = paddle::operators::trim_trailing_singular_dims(y_dims);
    axis_trim = (y_dims_trimed.size() == 0) ? x_dims.size() : axis;
    paddle::operators::get_mid_dims(x_dims,
                                    y_dims_trimed,
                                    axis_trim,
                                    &pre,
                                    &n,
                                    &post,
                                    &is_run_common_broadcast);
  } else {
    auto x_dims_trimed = paddle::operators::trim_trailing_singular_dims(x_dims);
    axis_trim = (x_dims_trimed.size() == 0) ? y_dims.size() : axis;
    paddle::operators::get_mid_dims(y_dims,
                                    x_dims_trimed,
                                    axis_trim,
                                    &pre,
                                    &n,
                                    &post,
                                    &is_run_common_broadcast);
  }
  // special case for common implementation.
  // case 1: x=[2,3,1,5], y=[2,1,4,1]
  // case 2: x=[2,3,4], y=[1,1,4]
  if (is_run_common_broadcast == 1) {
    CommonElementwiseBroadcastForward<Functor, DeviceContext, T, OutType>(
        dev_ctx, x, y, z, x_dims, y_dims, func, axis, is_xsize_larger);
    return;
  }

  if (post == 1) {
    functor.RunRowWise(n, pre);
    return;
  } else {
    functor.RunMidWise(n, pre, post);
    return;
  }
}

template <typename DeviceContext, typename T>
void LaunchBroadcastElementwiseCpuKernel(const DeviceContext& dev_ctx,
                                         const DenseTensor& x,
                                         const DenseTensor& y,
                                         int axis,
                                         DenseTensor* z) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  if (x_dims.size() >= y_dims.size()) {
    ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(
        dev_ctx, x, y, axis, AddFunctor<T>(), z);
  } else {
    ElementwiseComputeEx<InverseAddFunctor<T>, DeviceContext, T>(
        dev_ctx, x, y, axis, InverseAddFunctor<T>(), z);
  }
}

template <typename DeviceContext, typename T, class Enable = void>
struct SameDimsElemwiseAdd {
  void operator()(const DeviceContext& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z);
};

template <typename T>
struct SameDimsElemwiseAdd<
    paddle::platform::CPUDeviceContext,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const paddle::platform::CPUDeviceContext& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    auto blas =
        paddle::operators::math::GetBlas<paddle::platform::CPUDeviceContext, T>(
            dev_ctx);
    blas.VADD(x.numel(), x.data<T>(), y.data<T>(), z->mutable_data<T>());
  }
};

template <typename T>
struct SameDimsElemwiseAdd<
    paddle::platform::CPUDeviceContext,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const paddle::platform::CPUDeviceContext& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    auto eigen_x = pten::EigenVector<T>::Flatten(x);
    auto eigen_y = pten::EigenVector<T>::Flatten(y);
    auto eigen_z = pten::EigenVector<T>::Flatten(*z);
    auto& place = *dev_ctx.eigen_device();
    eigen_z.device(place) = eigen_x + eigen_y;
  }
};

template <typename T>
void ElementwiseAddFunction(const CPUContext& dev_ctx,
                            const DenseTensor& X,
                            const DenseTensor& Y,
                            int axis,
                            DenseTensor* Out) {
  Out->mutable_data<T>();
  if (X.dims() == Y.dims()) {
    SameDimsElemwiseAdd<CPUContext, T> LaunchElementwiseCpuKernel;
    LaunchElementwiseCpuKernel(dev_ctx, X, Y, Out);
  } else {
    LaunchBroadcastElementwiseCpuKernel<CPUContext, T>(
        dev_ctx, X, Y, axis, Out);
  }
}

}  // namespace math
}  // namespace pten
