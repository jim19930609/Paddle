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
#include "paddle/fluid/platform/for_range.h"

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/functions/eigen/common.h"

#include "paddle/pten/kernels/functions/math/matmul_function.h"
#include "paddle/pten/kernels/functions/math/reduce_function.h"
#include "paddle/pten/kernels/functions/math/transform_function.h"

namespace pten {
namespace math {

struct IdentityFunctor {
  HOSTDEVICE explicit inline IdentityFunctor() {}

  template <typename U>
  HOSTDEVICE inline U operator()(const U& x) const {
    return x;
  }
};

template <typename DeviceContext, typename T>
void ReduceSumForMatmulGrad(const DeviceContext& dev_ctx,
                            const DenseTensor& input,
                            DenseTensor* output,
                            const std::vector<int>& reduce_dims) {
#if defined(__NVCC__) || defined(__HIPCC__)
  TensorReduce<T, T, cub::Sum, IdentityFunctor>(dev_ctx,
                                                input,
                                                output,
                                                reduce_dims,
                                                static_cast<T>(0),
                                                cub::Sum(),
                                                IdentityFunctor());
#else
  ReduceKernelFunctor<DeviceContext, T, SumFunctor>(
      dev_ctx, input, reduce_dims, true, false, output);
#endif
}

template <typename DeviceContext, typename T>
void MatMul(const DeviceContext& dev_ctx,
            const DenseTensor& mat_a,
            const paddle::operators::math::MatDescriptor& dim_a,
            const DenseTensor& mat_b,
            const paddle::operators::math::MatDescriptor& dim_b,
            T alpha,
            DenseTensor* mat_out,
            T beta) {
  PADDLE_ENFORCE_EQ(
      dim_a.width_,
      dim_b.height_,
      paddle::platform::errors::InvalidArgument(
          "The fisrt matrix width should be same as second matrix height,"
          "but received fisrt matrix width %d"
          ", second matrix height %d",
          dim_a.width_,
          dim_b.height_));

  CBLAS_TRANSPOSE transA = !dim_a.trans_ ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !dim_b.trans_ ? CblasNoTrans : CblasTrans;

  auto blas = paddle::operators::math::GetBlas<DeviceContext, T>(dev_ctx);
  if (dim_a.batch_size_ == 0 && dim_b.batch_size_ == 0) {
    blas.GEMM(transA,
              transB,
              dim_a.height_,
              dim_b.width_,
              dim_a.width_,
              alpha,
              mat_a.data<T>(),
              mat_b.data<T>(),
              beta,
              mat_out->mutable_data<T>());

  } else {
    PADDLE_ENFORCE_EQ(
        dim_a.batch_size_ == dim_b.batch_size_ || dim_a.batch_size_ == 0 ||
            dim_b.batch_size_ == 0,
        true,
        paddle::platform::errors::InvalidArgument(
            "dim_a.batch_size should be equal to dim_b.batch_size, or "
            "one of dim_a.batch_size and dim_b.batch_size should be 0. "
            "But got dim_a.batch_size = %d, dim_b.batch_size = %d.",
            dim_a.batch_size_,
            dim_b.batch_size_));
    blas.BatchedGEMM(
        transA,
        transB,
        dim_a.height_,
        dim_b.width_,
        dim_a.width_,
        alpha,
        mat_a.data<T>(),
        mat_b.data<T>(),
        beta,
        mat_out->mutable_data<T>(),
        dim_a.batch_size_ == 0 ? dim_b.batch_size_ : dim_a.batch_size_,
        dim_a.stride_,
        dim_b.stride_);
  }
}

template <typename DeviceContext, typename T>
void MatMul(const DeviceContext& dev_ctx,
            const DenseTensor& a,
            bool trans_a,
            const DenseTensor& b,
            bool trans_b,
            DenseTensor* out) {
  out->mutable_data<T>();
  auto mat_dim_a =
      paddle::operators::math::CreateMatrixDescriptor(a.dims(), 0, trans_a);
  auto mat_dim_b =
      paddle::operators::math::CreateMatrixDescriptor(b.dims(), 0, trans_b);
  if (a.dims().size() == 3 && b.dims().size() <= 2) {
    // the transpose_X must be false, if is true, the transpose cost much time
    if (!trans_a) {
      mat_dim_a.height_ *= mat_dim_a.batch_size_;
      mat_dim_a.batch_size_ = 0;
    }
  }
  MatMul<DeviceContext, T>(dev_ctx,
                           a,
                           mat_dim_a,
                           b,
                           mat_dim_b,
                           static_cast<T>(1),
                           out,
                           static_cast<T>(0));
}

template <typename DeviceContext, typename T, int Rank>
struct Transpose {
  void operator()(const DeviceContext& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* out,
                  const std::vector<int>& axis) {
    Eigen::array<int, Rank> permute;
    for (int i = 0; i < Rank; i++) {
      permute[i] = axis[i];
    }
    auto eigen_in = pten::EigenTensor<T, Rank>::From(in);
    auto eigen_out = pten::EigenTensor<T, Rank>::From(*out);
    auto* dev = dev_ctx.eigen_device();
    // use 32bit index to speed up computation
    bool use_32bit_index = eigen_out.size() < Eigen::NumTraits<int>::highest();
    bool is_gpu_place = paddle::platform::is_gpu_place(dev_ctx.GetPlace());
    if (use_32bit_index && is_gpu_place) {
      To32BitIndex(eigen_out).device(*dev) =
          To32BitIndex(eigen_in).shuffle(permute);
    } else {
      eigen_out.device(*dev) = eigen_in.shuffle(permute);
    }
  }
};

// Reshape a rank-3 tensor from P x M x N to M x (P * N).
// (Warning: This requires transposing data and writes into new memory.)
// Identity op if the tensor is not of rank 3.
template <typename DeviceContext, typename T>
void FoldHeadAndLastDims(const DeviceContext& dev_ctx,
                         const DenseTensor& input,
                         DenseTensor* output) {
  auto in_dims = input.dims();
  if (in_dims.size() != 3) {
    return;
  }

  output->Resize({in_dims[1], in_dims[0], in_dims[2]});
  output->mutable_data<T>();
  std::vector<int> axis = {1, 0, 2};
  Transpose<DeviceContext, T, 3> trans;
  trans(dev_ctx, input, output, axis);
  output->Resize({in_dims[1], in_dims[0] * in_dims[2]});
}

// Reshape a rank-3 tensor from P x M x N to (P * M) x N.
// Identity op if the tensor is not of rank 3.
inline void FoldInitDims(const DenseTensor& input, DenseTensor* output) {
  auto in_dims = input.dims();
  if (in_dims.size() == 3) {
    output->Resize({in_dims[0] * in_dims[1], in_dims[2]});
  }
}

template <typename DeviceContext, typename T>
void CalcInputGrad(const DeviceContext& dev_ctx,
                   const DenseTensor& a,
                   bool trans_a,
                   bool is_fold_init_dims_a,
                   const DenseTensor& b,
                   bool trans_b,
                   bool is_fold_init_dims_b,
                   DenseTensor* out) {
  if (out == nullptr) return;
  bool need_combine = (a.dims().size() == 3 || b.dims().size() == 3) &&
                      (out->dims()).size() == 2;
  if (!need_combine) {
    MatMul<DeviceContext, T>(dev_ctx, a, trans_a, b, trans_b, out);
  } else {
    DenseTensor a_tmp(a.meta(), TensorStatus());
    a_tmp.ShareAllocation(a.allocation());
    if (is_fold_init_dims_a)
      FoldInitDims(a, &a_tmp);
    else
      FoldHeadAndLastDims<DeviceContext, T>(dev_ctx, a, &a_tmp);

    DenseTensor b_tmp(b.meta(), TensorStatus());
    b_tmp.ShareAllocation(b.allocation());
    if (is_fold_init_dims_b)
      FoldInitDims(b, &b_tmp);
    else
      FoldHeadAndLastDims<DeviceContext, T>(dev_ctx, b, &b_tmp);

    MatMul<DeviceContext, T>(dev_ctx, a_tmp, trans_a, b_tmp, trans_b, out);
  }
}

inline void SyncDenseTensors(DenseTensor& src, DenseTensor& dst) {
  dst.Resize(src.dims());

  const TensorMeta& src_meta = src.meta();
  TensorMeta* dst_meta = dst.mutable_meta();

  dst_meta->layout = src_meta.layout;
  dst_meta->type = src_meta.type;
  dst_meta->backend = src_meta.backend;

  src.CheckMemorySize();
  dst.ShareAllocation(src.allocation());
}

template <typename DeviceContext, typename T>
struct ConjHelper {
  explicit ConjHelper(const DeviceContext& dev_ctx) : dev_ctx_(dev_ctx) {}

  HOSTDEVICE void operator()(DenseTensor& src, DenseTensor& dst) {
    SyncDenseTensors(src, dst);

    return;
  }

  const DeviceContext& dev_ctx_;
};

template <typename DeviceContext>
struct ConjHelper<DeviceContext, paddle::platform::complex<float>> {
  explicit ConjHelper(const DeviceContext& dev_ctx) : dev_ctx_(dev_ctx) {}

  HOSTDEVICE void operator()(DenseTensor& src, DenseTensor& dst) {
    dst.Resize(src.dims());
    auto* src_data = src.data<paddle::platform::complex<float>>();
    auto* dst_data = dst.mutable_data<paddle::platform::complex<float>>();

    paddle::platform::ForRange<DeviceContext> for_range(dev_ctx_, src.numel());
    paddle::operators::math::ConjFunctor<paddle::platform::complex<float>>
        functor(src_data, src.numel(), dst_data);
    for_range(functor);
    return;
  }
  const DeviceContext& dev_ctx_;
};

template <typename DeviceContext>
struct ConjHelper<DeviceContext, paddle::platform::complex<double>> {
  explicit ConjHelper(const DeviceContext& dev_ctx) : dev_ctx_(dev_ctx) {}

  HOSTDEVICE void operator()(DenseTensor& src, DenseTensor& dst) {
    dst.Resize(src.dims());
    auto* src_data = src.data<paddle::platform::complex<double>>();
    auto* dst_data = dst.mutable_data<paddle::platform::complex<double>>();

    paddle::platform::ForRange<DeviceContext> for_range(dev_ctx_, src.numel());
    paddle::operators::math::ConjFunctor<paddle::platform::complex<double>>
        functor(src_data, src.numel(), dst_data);
    for_range(functor);
    return;
  }
  const DeviceContext& dev_ctx_;
};

inline paddle::framework::DDim RowMatrixFromVector(
    const paddle::framework::DDim& x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return paddle::framework::make_ddim({1, x_dim[0]});
}

/**
 * Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
 * original y_dim is returned.
 */
inline paddle::framework::DDim ColumnMatrixFromVector(
    const paddle::framework::DDim& y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return paddle::framework::make_ddim({y_dim[0], 1});
}

inline void ReshapeTensorIntoMatrixSequence(
    DenseTensor* x, const paddle::operators::math::MatDescriptor& descriptor) {
  int64_t h, w;
  h = descriptor.height_;
  w = descriptor.width_;
  if (descriptor.trans_) {
    std::swap(w, h);
  }
  if (descriptor.batch_size_) {
    x->Resize({descriptor.batch_size_, h, w});
  } else {
    x->Resize({h, w});
  }
}

inline void ReshapeXYOutIntoMatrixSequence(DenseTensor* x,
                                           DenseTensor* y,
                                           DenseTensor* out,
                                           bool trans_x,
                                           bool trans_y) {
  auto x_dim = RowMatrixFromVector(x->dims());
  auto y_dim = ColumnMatrixFromVector(y->dims());
  auto mat_dim_x =
      paddle::operators::math::CreateMatrixDescriptor(x_dim, 0, trans_x);
  auto mat_dim_y =
      paddle::operators::math::CreateMatrixDescriptor(y_dim, 0, trans_y);
  if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
    out->Resize({mat_dim_x.height_, mat_dim_y.width_});
  } else {
    out->Resize({(std::max)(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
                 mat_dim_x.height_,
                 mat_dim_y.width_});
  }

  ReshapeTensorIntoMatrixSequence(x, mat_dim_x);
  ReshapeTensorIntoMatrixSequence(y, mat_dim_y);
}

template <typename DeviceContext, typename T, typename Enabel = void>
struct DotGradFunction {
  void operator()(const DeviceContext& dev_ctx,
                  const DenseTensor* tensor_x,
                  const DenseTensor* tensor_y,
                  const DenseTensor* tensor_dout,
                  DenseTensor* tensor_dx,
                  DenseTensor* tensor_dy);
};

template <typename DeviceContext, typename T>
struct DotGradFunction<DeviceContext,
                       T,
                       paddle::operators::math::EnableComplex<T>> {
  void operator()(const DeviceContext& dev_ctx,
                  const DenseTensor* tensor_x,
                  const DenseTensor* tensor_y,
                  const DenseTensor* tensor_dout,
                  DenseTensor* tensor_dx,
                  DenseTensor* tensor_dy) {
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 == tensor_dout->dims().size()) {
      auto dout = pten::EigenVector<T>::Flatten(*tensor_dout);

      if (tensor_dx) {
        auto y = pten::EigenVector<T>::Flatten(*tensor_y);
        auto& dev = *dev_ctx.eigen_device();
        Eigen::DSizes<int, 1> size(tensor_dx->numel());

        paddle::platform::ForRange<DeviceContext> for_range(dev_ctx,
                                                            tensor_y->numel());
        paddle::operators::math::ConjFunctor<T> functor(
            tensor_y->data<T>(),
            tensor_y->numel(),
            tensor_dx->mutable_data<T>());
        for_range(functor);
        auto dx = pten::EigenVector<T>::Flatten(*tensor_dx);

        dx.device(dev) = dx * dout.broadcast(size);
      }

      if (tensor_dy) {
        auto x = pten::EigenVector<T>::Flatten(*tensor_x);
        auto& dev = *dev_ctx.eigen_device();
        Eigen::DSizes<int, 1> size(tensor_dy->numel());

        paddle::platform::ForRange<DeviceContext> for_range(dev_ctx,
                                                            tensor_y->numel());
        paddle::operators::math::ConjFunctor<T> functor(
            tensor_x->data<T>(),
            tensor_x->numel(),
            tensor_dy->mutable_data<T>());
        for_range(functor);
        auto dy = pten::EigenVector<T>::Flatten(*tensor_dy);

        dy.device(dev) = dy * dout.broadcast(size);
      }
    } else {
      auto dout = pten::EigenMatrix<T>::From(*tensor_dout);

      if (tensor_dx) {
        tensor_dx->mutable_data<T>();
        auto y = pten::EigenMatrix<T>::From(*tensor_y);
        auto& dev = *dev_ctx.eigen_device();
        Eigen::DSizes<int, 2> size(1, tensor_dx->dims()[1]);

        paddle::platform::ForRange<DeviceContext> for_range(dev_ctx,
                                                            tensor_y->numel());
        paddle::operators::math::ConjFunctor<T> functor(
            tensor_y->data<T>(),
            tensor_y->numel(),
            tensor_dx->mutable_data<T>());
        for_range(functor);
        auto dx = pten::EigenMatrix<T>::From(*tensor_dx);

        dx.device(dev) = dx * dout.broadcast(size);
      }

      if (tensor_dy) {
        tensor_dy->mutable_data<T>();
        auto x = pten::EigenMatrix<T>::From(*tensor_x);
        auto& dev = *dev_ctx.eigen_device();
        Eigen::DSizes<int, 2> size(1, tensor_dy->dims()[1]);

        paddle::platform::ForRange<DeviceContext> for_range(dev_ctx,
                                                            tensor_x->numel());
        paddle::operators::math::ConjFunctor<T> functor(
            tensor_x->data<T>(),
            tensor_x->numel(),
            tensor_dy->mutable_data<T>());
        for_range(functor);

        auto dy = pten::EigenMatrix<T>::From(*tensor_dy);

        dy.device(dev) = dy * dout.broadcast(size);
      }
    }
#else
    const auto* data_dout = tensor_dout->data<T>();

    if (tensor_dx) {
      auto* data_dx = tensor_dx->mutable_data<T>();
      const auto* data_y = tensor_y->data<T>();
      const paddle::framework::DDim& dim = tensor_x->dims();
      size_t N = static_cast<size_t>(paddle::framework::product(dim));

      auto step = dim[dim.size() - 1];

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dx[i] = T(data_y[i].real, -data_y[i].imag) * data_dout[s];
      }
    }

    if (tensor_dy) {
      auto* data_dy = tensor_dy->mutable_data<T>();
      const auto* data_x = tensor_x->data<T>();
      const paddle::framework::DDim& dim = tensor_y->dims();
      size_t N = static_cast<size_t>(paddle::framework::product(dim));

      auto step = dim[dim.size() - 1];

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dy[i] = T(data_x[i].real, -data_x[i].imag) * data_dout[s];
      }
    }
#endif
  }
};

template <typename DeviceContext, typename T>
struct DotGradFunction<DeviceContext,
                       T,
                       paddle::operators::math::DisableComplex<T>> {
  void operator()(const DeviceContext& dev_ctx,
                  const DenseTensor* tensor_x,
                  const DenseTensor* tensor_y,
                  const DenseTensor* tensor_dout,
                  DenseTensor* tensor_dx,
                  DenseTensor* tensor_dy) {
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 == tensor_dout->dims().size()) {
      auto dout = pten::EigenVector<T>::Flatten(*tensor_dout);

      if (tensor_dx) {
        auto y = pten::EigenVector<T>::Flatten(*tensor_y);
        auto dx = pten::EigenVector<T>::Flatten(*tensor_dx);
        auto& dev = *dev_ctx.eigen_device();
        Eigen::DSizes<int, 1> size(tensor_dx->numel());
        dx.device(dev) = y * dout.broadcast(size);
      }

      if (tensor_dy) {
        auto x = pten::EigenVector<T>::Flatten(*tensor_x);
        auto dy = pten::EigenVector<T>::Flatten(*tensor_dy);
        auto& dev = *dev_ctx.eigen_device();
        Eigen::DSizes<int, 1> size(tensor_dy->numel());
        dy.device(dev) = x * dout.broadcast(size);
      }
    } else {
      auto dout = pten::EigenMatrix<T>::From(*tensor_dout);

      if (tensor_dx) {
        tensor_dx->mutable_data<T>();
        auto y = pten::EigenMatrix<T>::From(*tensor_y);
        auto dx = pten::EigenMatrix<T>::From(*tensor_dx);
        auto& dev = *dev_ctx.eigen_device();
        Eigen::DSizes<int, 2> size(1, tensor_dx->dims()[1]);
        dx.device(dev) = y * dout.broadcast(size);
      }

      if (tensor_dy) {
        tensor_dy->mutable_data<T>();
        auto x = pten::EigenMatrix<T>::From(*tensor_x);
        auto dy = pten::EigenMatrix<T>::From(*tensor_dy);
        auto& dev = *dev_ctx.eigen_device();
        Eigen::DSizes<int, 2> size(1, tensor_dy->dims()[1]);
        dy.device(dev) = x * dout.broadcast(size);
      }
    }
#else
    auto const *x = tensor_x->data<T>(), *y = tensor_y->data<T>(),
               *dz = tensor_dout->data<T>();
    auto&& d = tensor_x->dims();
    auto const N = tensor_x->numel();
    auto const B = d[d.size() - 1];

    if (tensor_dx) {
      auto* dx = tensor_dx->mutable_data<T>();
      for (auto j = 0; j < N / B; ++j) {
        auto const ss = dz[j];
        for (auto i = 0; i < B; ++i) *dx++ = *y++ * ss;
      }
    }

    if (tensor_dy) {
      auto* dy = tensor_dy->mutable_data<T>();
      for (auto j = 0; j < N / B; ++j) {
        auto const ss = dz[j];
        for (auto i = 0; i < B; i++) *dy++ = *x++ * ss;
      }
    }
#endif
  }
};

template <typename DeviceContext, typename T>
void GradMatMulFunction(const DeviceContext& dev_ctx,
                        const DenseTensor& X_in,
                        const DenseTensor& Y_in,
                        const DenseTensor& GradOut_in,
                        bool transpose_x,
                        bool transpose_y,
                        DenseTensor* GradX,
                        DenseTensor* GradY) {
  DenseTensor X(X_in.meta(), TensorStatus());
  X.ShareAllocation(X_in.allocation());

  DenseTensor Y(Y_in.meta(), TensorStatus());
  Y.ShareAllocation(Y_in.allocation());

  DenseTensor GradOut(GradOut_in.meta(), TensorStatus());
  GradOut.ShareAllocation(GradOut_in.allocation());

  pten::DenseTensor x_conj(X.meta(), pten::TensorStatus());
  pten::DenseTensor y_conj(Y.meta(), pten::TensorStatus());

  // get dims
  std::vector<std::int64_t> x_dims = vectorize(X.dims());
  std::vector<std::int64_t> y_dims = vectorize(Y.dims());
  std::vector<std::int64_t> dout_dims = vectorize(GradOut.dims());

  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int ndim = dout_dims.size();

  // Case1 : x's or y's dim = 1
  if (x_ndim == 1 && y_ndim == 1) {
    if (GradX) GradX->mutable_data<T>();
    if (GradY) GradY->mutable_data<T>();
    if (GradOut.numel() == 1) {
      DotGradFunction<DeviceContext, T>()(
          dev_ctx, &X, &Y, &GradOut, GradX, GradY);
      return;
    }
  }

  bool is_broadcast = true;
  if (x_ndim <= 2 || y_ndim <= 2) {
    is_broadcast = false;
  } else if (x_ndim != y_ndim) {
    is_broadcast = true;
  } else {
    is_broadcast = !std::equal(
        x_dims.cbegin(), x_dims.cbegin() + x_ndim - 2, y_dims.cbegin());
  }

  // Case2: no broadcast or no batch size, it aims to speed and it is same as
  // matmul in old version.
  if (!is_broadcast) {
    ReshapeXYOutIntoMatrixSequence(&X, &Y, &GradOut, transpose_x, transpose_y);
    paddle::framework::DDim dx_dims;
    if (GradX) {
      dx_dims = GradX->dims();
      if (dx_dims != X.dims()) {
        GradX->Resize(X.dims());
      }

      // for complex
      ConjHelper<DeviceContext, T> conj_helper(dev_ctx);
      conj_helper(Y, y_conj);
    }

    paddle::framework::DDim dy_dims;
    if (GradY) {
      dy_dims = GradY->dims();
      if (dy_dims != Y.dims()) {
        GradY->Resize(Y.dims());
      }

      // for complex
      ConjHelper<DeviceContext, T> conj_helper(dev_ctx);
      conj_helper(X, x_conj);
    }

    if (transpose_x && transpose_y) {
      CalcInputGrad<DeviceContext, T>(
          dev_ctx, y_conj, true, true, GradOut, true, false, GradX);
      CalcInputGrad<DeviceContext, T>(
          dev_ctx, GradOut, true, true, x_conj, true, false, GradY);
    } else if (transpose_x) {
      CalcInputGrad<DeviceContext, T>(
          dev_ctx, y_conj, false, false, GradOut, true, false, GradX);
      CalcInputGrad<DeviceContext, T>(
          dev_ctx, x_conj, false, false, GradOut, false, true, GradY);
    } else if (transpose_y) {
      CalcInputGrad<DeviceContext, T>(
          dev_ctx, GradOut, false, false, y_conj, false, true, GradX);
      CalcInputGrad<DeviceContext, T>(
          dev_ctx, GradOut, true, true, x_conj, false, true, GradY);
    } else {
      CalcInputGrad<DeviceContext, T>(
          dev_ctx, GradOut, false, false, y_conj, true, false, GradX);
      CalcInputGrad<DeviceContext, T>(
          dev_ctx, x_conj, true, true, GradOut, false, true, GradY);
    }

    if (GradX) {
      if (dx_dims != X.dims()) {
        GradX->Resize(dx_dims);
      }
    }
    if (GradY) {
      if (dy_dims != Y.dims()) {
        GradY->Resize(dy_dims);
      }
    }
  } else {
    // Case3: broadcast. It need cost much time to reduce sum for the
    // broadcast and wastes the memory.
    // So we should avoid the case in reality.
    VLOG(3) << "It need cost much time to reduce sum for the broadcast and "
               "wastes the memory. So we should avoid the case in reality";
    pten::DenseTensor dx_help(GradX->meta(), pten::TensorStatus());
    pten::DenseTensor dy_help(GradY->meta(), pten::TensorStatus());

    ConjHelper<DeviceContext, T> conj_helper(dev_ctx);
    conj_helper(X, x_conj);
    conj_helper(Y, y_conj);
    if (transpose_x) {
      if (transpose_y) {
        // X'Y': dA = Y'G', dB = G'X'
        if (GradX)
          MatMulFunction<DeviceContext, T>(dev_ctx,
                                           y_conj,
                                           GradOut,
                                           y_dims,
                                           dout_dims,
                                           &dx_help,
                                           true,
                                           true);
        if (GradY)
          MatMulFunction<DeviceContext, T>(dev_ctx,
                                           GradOut,
                                           x_conj,
                                           dout_dims,
                                           x_dims,
                                           &dy_help,
                                           true,
                                           true);
      } else {
        // X'Y: dX = YG', dY = XG
        if (GradX)
          MatMulFunction<DeviceContext, T>(dev_ctx,
                                           y_conj,
                                           GradOut,
                                           y_dims,
                                           dout_dims,
                                           &dx_help,
                                           false,
                                           true);
        if (GradY)
          MatMulFunction<DeviceContext, T>(dev_ctx,
                                           x_conj,
                                           GradOut,
                                           x_dims,
                                           dout_dims,
                                           &dy_help,
                                           false,
                                           false);
      }
    } else {
      if (transpose_y) {
        // XY': dX = GY, dY = G'X
        if (GradX)
          MatMulFunction<DeviceContext, T>(dev_ctx,
                                           GradOut,
                                           y_conj,
                                           dout_dims,
                                           y_dims,
                                           &dx_help,
                                           false,
                                           false);
        if (GradY)
          MatMulFunction<DeviceContext, T>(dev_ctx,
                                           GradOut,
                                           x_conj,
                                           dout_dims,
                                           x_dims,
                                           &dy_help,
                                           true,
                                           false);
      } else {
        // XY: dX = GY', dY = X'G
        if (GradX)
          MatMulFunction<DeviceContext, T>(dev_ctx,
                                           GradOut,
                                           y_conj,
                                           dout_dims,
                                           y_dims,
                                           &dx_help,
                                           false,
                                           true);
        if (GradY)
          MatMulFunction<DeviceContext, T>(dev_ctx,
                                           x_conj,
                                           GradOut,
                                           x_dims,
                                           dout_dims,
                                           &dy_help,
                                           true,
                                           false);
      }
    }

    // get help dims
    const std::vector<std::int64_t> dx_help_dims = vectorize(dx_help.dims());
    const std::vector<std::int64_t> dy_help_dims = vectorize(dy_help.dims());

    std::vector<std::int64_t> dx_broadcast_dims(ndim);
    std::vector<std::int64_t> dy_broadcast_dims(ndim);

    std::fill(
        dx_broadcast_dims.data(), dx_broadcast_dims.data() + ndim - x_ndim, 1);
    std::fill(
        dy_broadcast_dims.data(), dy_broadcast_dims.data() + ndim - y_ndim, 1);
    std::copy(x_dims.data(),
              x_dims.data() + x_ndim,
              dx_broadcast_dims.data() + ndim - x_ndim);
    std::copy(y_dims.data(),
              y_dims.data() + y_ndim,
              dy_broadcast_dims.data() + ndim - y_ndim);

    std::vector<int> dx_reduce_dims;
    std::vector<int> dy_reduce_dims;
    for (int idx = 0; idx <= ndim - 3; idx++) {
      if (dx_help_dims[idx] != 1 && dx_broadcast_dims[idx] == 1) {
        dx_reduce_dims.push_back(idx);
      }
      if (dy_help_dims[idx] != 1 && dy_broadcast_dims[idx] == 1) {
        dy_reduce_dims.push_back(idx);
      }
    }

    // reduce sum to get grad by ReduceSum
    if (GradX) {
      if (dx_reduce_dims.empty()) {
        SyncDenseTensors(dx_help, *GradX);
      } else {
        ReduceSumForMatmulGrad<DeviceContext, T>(
            dev_ctx, dx_help, GradX, dx_reduce_dims);
      }
      GradX->Resize(X.dims());
    }
    if (GradY) {
      if (dy_reduce_dims.empty()) {
        SyncDenseTensors(dy_help, *GradY);
      } else {
        ReduceSumForMatmulGrad<DeviceContext, T>(
            dev_ctx, dy_help, GradY, dy_reduce_dims);
      }
      GradY->Resize(Y.dims());
    }
  }
}

}  // namespace math
}  // namespace pten
