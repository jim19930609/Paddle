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

#if defined(__NVCC__) || defined(__HIPCC__)

#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/pten/kernels/cuda/utils.h"
#include "paddle/pten/kernels/functions/math/transform_function.h"

namespace pten {
namespace math {

namespace kps = paddle::operators::kernel_primitives;

template <typename Tx, typename Ty = Tx>
struct IdentityFunctor {
  HOSTDEVICE inline IdentityFunctor() {}

  HOSTDEVICE explicit inline IdentityFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x);
  }
};

// for cub::Reduce
template <typename Tx, typename Ty = Tx>
struct CustomSum {
  using Transformer = IdentityFunctor<Tx, Ty>;

  inline Ty initial() { return static_cast<Ty>(0.0f); }

  __device__ __forceinline__ Ty operator()(const Ty& a, const Ty& b) const {
    return b + a;
  }
};

template <typename Tx,
          typename Ty,
          template <typename, typename> class ReduceOp>
void TensorReduceFunctorImpl(const CUDAContext& dev_ctx,
                             const DenseTensor& x,
                             DenseTensor* y,
                             std::vector<int> origin_reduce_dims) {
  cudaStream_t stream = dev_ctx.stream();

  auto x_dim = paddle::framework::vectorize<int>(x.dims());
  auto config = paddle::operators::ReduceConfig<Ty>(origin_reduce_dims, x_dim);
  config.Run();  // get the parameters of LaunchReduceKernel
  int numel = x.numel();
  // after config.run()
  // SetOutputData for ReduceHigherDim when should_reduce_again is true,
  // temp_output should be stored temp_data in output_data space or stored in
  // y_data;
  paddle::framework::Tensor tmp;
  auto x_data = x.data<Tx>();
  auto y_data = y->mutable_data<Ty>();
  if (config.reduce_num == 1) {
    auto out_dims = y->dims();
    if (x.data_type() == y->data_type()) {
      pten::Copy(dev_ctx, x, y);
      y->Resize(out_dims);
    } else {
      paddle::framework::VisitDataType(
          static_cast<paddle::framework::proto::VarType::Type>(y->data_type()),
          CastOpFunctor<CUDAContext, Tx>(x, y, dev_ctx));
    }
    return;
  }
  config.SetOutputData(y_data, x.place(), &tmp);

  bool use_cub_reduce = (config.reduce_num == numel) &&
                        (!std::is_same<Tx, paddle::platform::float16>::value);
  if (use_cub_reduce) {
    // launch CUB::Reduce
    using TransformOp = typename ReduceOp<Tx, Ty>::Transformer;
    auto reducer = ReduceOp<Tx, Ty>();
    cub::TransformInputIterator<Ty, TransformOp, const Tx*> trans_x(
        x_data, TransformOp(config.reduce_num));
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(nullptr,
                              temp_storage_bytes,
                              trans_x,
                              y_data,
                              config.reduce_num,
                              reducer,
                              reducer.initial(),
                              stream);
    paddle::framework::Tensor tmp;
    auto* temp_storage = tmp.mutable_data<uint8_t>(
        paddle::framework::make_ddim(
            {static_cast<int64_t>(temp_storage_bytes)}),
        x.place());
    cub::DeviceReduce::Reduce(temp_storage,
                              temp_storage_bytes,
                              trans_x,
                              y_data,
                              config.reduce_num,
                              reducer,
                              reducer.initial(),
                              stream);

    return;
  }

  using MPType = typename paddle::operators::details::MPTypeTrait<Ty>::Type;
  auto reducer = ReduceOp<Tx, MPType>();
  // launch ReduceHigherDimKernel
  // when reduce_dim.size() == 1 and reduce_dim[0] != x_dim.size() - 1, this
  // function will be used
  // eg: x_dim = {nz, ny, nx}, nx != 1, axis can be 0 or 1
  //     if axis = 1 then grid.z = nz, grid.y = ny / block_size, grid.x = nx /
  //     32
  //     else grid.z = 1, grid.y = ny / block_size, grid.x = nx /32

  if (config.reduce_type == paddle::operators::ReduceType::kReduceHigherDim) {
    using TransformOp = typename ReduceOp<Tx, MPType>::Transformer;

    paddle::operators::ReduceHigherDimKernel<
        Tx,
        Ty,
        MPType,
        ReduceOp<Tx, MPType>,
        TransformOp><<<config.grid, config.block, 0, stream>>>(
        x_data,
        config.output_data,
        reducer,
        TransformOp(config.reduce_num),
        reducer.initial(),
        config.reduce_num,
        config.left_num,
        config.blocking_size);

    if (config.should_reduce_again) {
      dim3 block = dim3(config.block.x, 1, 1);
      dim3 grid = dim3(config.grid.x, 1, config.grid.z);
      paddle::operators::ReduceHigherDimKernel<
          Ty,
          Ty,
          MPType,
          ReduceOp<Tx, MPType>,
          IdentityFunctor<Ty, MPType>><<<grid, block, 0, stream>>>(
          config.output_data,
          y_data,
          reducer,
          IdentityFunctor<Ty, MPType>(config.grid.y),
          reducer.initial(),
          config.grid.y,
          config.left_num,
          config.grid.y);
    }
    return;
  }

  // when reduce_dim.size() == 1 and reduce_dim[0] == x_dim.size() - 1, or
  // when reduce_dim.size() != 1 and reduce_dim.size() != x_dim.size(), this
  // function will be used
  paddle::operators::LaunchReduceKernel<Tx, Ty, MPType, ReduceOp<Tx, MPType>>(
      x_data, y_data, reducer, reducer.initial(), stream, config);
}

template <typename T, typename DX_OP, typename DY_OP>
void CommonGradBroadcastCUDA(const DenseTensor& x,
                             const DenseTensor& y,
                             const DenseTensor& out,
                             const DenseTensor& dout,
                             DenseTensor* dx,
                             DenseTensor* dy,
                             int* x_dims_array,
                             int* y_dims_array,
                             int* out_dims_array,
                             int max_dim,
                             const paddle::platform::CUDADeviceContext& ctx,
                             DX_OP dx_op,
                             DY_OP dy_op) {
  const auto gplace =
      BOOST_GET_CONST(paddle::platform::CUDAPlace, ctx.GetPlace());
  auto cplace = paddle::platform::CPUPlace();
  const T* x_data = x.data<T>();
  const T* y_data = y.data<T>();
  const T* out_data = out.data<T>();
  const T* dout_data = dout.data<T>();
  T* dx_data = dx == nullptr ? nullptr : dx->mutable_data<T>();
  T* dy_data = dy == nullptr ? nullptr : dy->mutable_data<T>();

  std::vector<int> x_one_indexs;
  std::vector<int> y_one_indexs;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] != y_dims_array[i]) {
      if (x_dims_array[i] == 1) {
        x_one_indexs.push_back(i);
      }
      if (y_dims_array[i] == 1) {
        y_one_indexs.push_back(i);
      }
    }
  }

  std::vector<int> x_trans_indexs(max_dim);
  std::vector<int> y_trans_indexs(max_dim);
  paddle::operators::ComputeBroadcastTranspositionArray(
      x_one_indexs.data(), x_trans_indexs.data(), max_dim, x_one_indexs.size());
  paddle::operators::ComputeBroadcastTranspositionArray(
      y_one_indexs.data(), y_trans_indexs.data(), max_dim, y_one_indexs.size());

  // compute array stride for cuda kernel;
  // e.g. x.dims=[2,3,4], x_stride=[12,4,1]
  std::vector<int> x_strides_array(max_dim);
  std::vector<int> y_strides_array(max_dim);
  std::vector<int> out_strides_array(max_dim);
  int x_stride = 1;
  int y_stride = 1;
  int z_stride = 1;
  for (int i = max_dim - 1; i >= 0; i--) {
    x_strides_array[i] = x_dims_array[i] == 1 ? 0 : x_stride;
    y_strides_array[i] = y_dims_array[i] == 1 ? 0 : y_stride;
    out_strides_array[i] = z_stride;
    x_stride *= x_dims_array[i];
    y_stride *= y_dims_array[i];
    z_stride *= out_dims_array[i];
  }

  std::vector<int> x_strides_order(max_dim);
  std::vector<int> y_strides_order(max_dim);
  std::vector<int> x_dims_order(max_dim);
  std::vector<int> y_dims_order(max_dim);
  for (int i = 0; i < max_dim; ++i) {
    x_strides_order[i] = out_strides_array[x_trans_indexs[i]];
    y_strides_order[i] = out_strides_array[y_trans_indexs[i]];
    x_dims_order[i] = out_dims_array[x_trans_indexs[i]];
    y_dims_order[i] = out_dims_array[y_trans_indexs[i]];
  }
  std::vector<int> x_broadcast_pos;
  std::vector<int> y_broadcast_pos;

  int bytes = max_dim * sizeof(int);

  for (int i = 0; i < max_dim; ++i) {
    if (x_dims_array[i] != out_dims_array[i] && x_dims_array[i] == 1) {
      x_broadcast_pos.emplace_back(i);
    }
    if (y_dims_array[i] != out_dims_array[i] && y_dims_array[i] == 1) {
      y_broadcast_pos.emplace_back(i);
    }
  }

  auto stream = ctx.stream();
  bool can_split_x = false;
  bool can_split_y = false;

  auto FastCommonCUDAF = [&](const std::vector<int>& broadcast_pos, bool is_y) {
    int h = std::accumulate(out_dims_array,
                            out_dims_array + broadcast_pos.size(),
                            1,
                            std::multiplies<int>());
    int w = std::accumulate(out_dims_array + broadcast_pos.size(),
                            out_dims_array + max_dim,
                            1,
                            std::multiplies<int>());

    VLOG(3) << "FastCommonCUDAF elementwise w:" << w << " h:" << h
            << " is_y:" << is_y;

    int split_h;
    int split_w;
    int kh = h;
    int kw = w;

    if (is_y) {
      split_h = std::accumulate(x_dims_array,
                                x_dims_array + broadcast_pos.size(),
                                1,
                                std::multiplies<int>());
      split_w = std::accumulate(x_dims_array + broadcast_pos.size(),
                                x_dims_array + max_dim,
                                1,
                                std::multiplies<int>());

    } else {
      split_h = std::accumulate(y_dims_array,
                                y_dims_array + broadcast_pos.size(),
                                1,
                                std::multiplies<int>());
      split_w = std::accumulate(y_dims_array + broadcast_pos.size(),
                                y_dims_array + max_dim,
                                1,
                                std::multiplies<int>());
    }

    if (h > split_h) kh = split_h;
    if (w > split_w) kw = split_w;

    if (is_y) {
      if (w < 16 || h < 16) {
        int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
        int grid_size = w;
        paddle::operators::CommonGradBroadcast1CUDAKernelHeight<<<grid_size,
                                                                  block_size,
                                                                  0,
                                                                  stream>>>(
            x_data,
            y_data,
            out_data,
            dout_data,
            h,
            w,
            dy_op,
            dy_data,
            kh,
            kw,
            is_y);
      } else {
        dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
        int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
        paddle::operators::FastCommonGradBroadcastCUDAKernelHeight<<<grid_size,
                                                                     block_size,
                                                                     0,
                                                                     stream>>>(
            x_data,
            y_data,
            out_data,
            dout_data,
            h,
            w,
            dy_op,
            dy_data,
            kh,
            kw,
            is_y);
      }
    } else {
      if (w < 16 || h < 16) {
        int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
        int grid_size = w;
        paddle::operators::CommonGradBroadcast1CUDAKernelHeight<<<grid_size,
                                                                  block_size,
                                                                  0,
                                                                  stream>>>(
            x_data,
            y_data,
            out_data,
            dout_data,
            h,
            w,
            dx_op,
            dx_data,
            kh,
            kw,
            is_y);
      } else {
        dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
        int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
        paddle::operators::FastCommonGradBroadcastCUDAKernelHeight<<<grid_size,
                                                                     block_size,
                                                                     0,
                                                                     stream>>>(
            x_data,
            y_data,
            out_data,
            dout_data,
            h,
            w,
            dx_op,
            dx_data,
            kh,
            kw,
            is_y);
      }
    }
  };

  auto FastBroadCastHeightCUDAF = [&](const std::vector<int>& broadcast_pos,
                                      bool x_large) {
    int h = std::accumulate(out_dims_array,
                            out_dims_array + broadcast_pos.size(),
                            1,
                            std::multiplies<int>());
    int w = std::accumulate(out_dims_array + broadcast_pos.size(),
                            out_dims_array + max_dim,
                            1,
                            std::multiplies<int>());

    VLOG(3) << "FastBroadCastHeightCUDAF w:" << w << " h:" << h;

    if (w < 16 || h < 16) {
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
      int grid_size = w;
      paddle::operators::ElemwiseGradBroadcast1CUDAKernel<<<grid_size,
                                                            block_size,
                                                            0,
                                                            stream>>>(x_data,
                                                                      y_data,
                                                                      out_data,
                                                                      dout_data,
                                                                      h,
                                                                      w,
                                                                      x_large,
                                                                      dx_op,
                                                                      dy_op,
                                                                      dx_data,
                                                                      dy_data);
    } else {
      dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
      int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
      paddle::operators::FastElemwiseGradBroadcast1CUDAKernel<<<grid_size,
                                                                block_size,
                                                                0,
                                                                stream>>>(
          x_data,
          y_data,
          out_data,
          dout_data,
          h,
          w,
          x_large,
          dx_op,
          dy_op,
          dx_data,
          dy_data);
    }
  };

  auto FastBroadCastAllCUDAF = [&](
      const std::vector<int>& broadcast_pos, int max_dim, bool is_x_large) {
    int axis = broadcast_pos[0];
    int pre = std::accumulate(
        out_dims_array, out_dims_array + axis, 1, std::multiplies<int>());
    int mid = 1;
    int post = 1;

    if (broadcast_pos.size() == 1) {
      mid = out_dims_array[axis];
      post = std::accumulate(out_dims_array + axis + 1,
                             out_dims_array + max_dim,
                             1,
                             std::multiplies<int>());
    } else {
      mid = std::accumulate(out_dims_array + axis,
                            out_dims_array + broadcast_pos.back() + 1,
                            1,
                            std::multiplies<int>());
      post = std::accumulate(out_dims_array + broadcast_pos.back() + 1,
                             out_dims_array + max_dim,
                             1,
                             std::multiplies<int>());
    }

    VLOG(3) << "FastBroadCastAllCUDAF pre:" << pre << " mid:" << mid
            << " post:" << post;

    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
    int grid_size = pre * post;

    paddle::operators::FastCommonGradBroadcastAllCUDAKernel<<<grid_size,
                                                              block_size,
                                                              0,
                                                              stream>>>(
        x_data,
        y_data,
        out_data,
        dout_data,
        pre,
        mid,
        post,
        is_x_large,
        dx_op,
        dy_op,
        dx_data,
        dy_data);
  };

  auto FastBroadCastOneCUDAF = [&](
      const std::vector<int>& broadcast_pos, int max_dim, bool is_x) {
    int axis = broadcast_pos[0];
    int pre = std::accumulate(
        out_dims_array, out_dims_array + axis, 1, std::multiplies<int>());
    int mid = out_dims_array[axis];
    int post = std::accumulate(out_dims_array + axis + 1,
                               out_dims_array + max_dim,
                               1,
                               std::multiplies<int>());

    int k_pre;
    int k_mid;
    int k_post;

    if (is_x) {
      k_pre = std::accumulate(
          y_dims_array, y_dims_array + axis, 1, std::multiplies<int>());
      k_mid = y_dims_array[axis];
      k_post = std::accumulate(y_dims_array + axis + 1,
                               y_dims_array + max_dim,
                               1,
                               std::multiplies<int>());
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
      int grid_size = pre * post;
      // we need to calc y offset with blockid, so do x_pre/y_pre to get left
      // size.
      if (k_pre != pre) k_pre = pre / k_pre;

      paddle::operators::FastCommonGradBroadcastOneCUDAKernel<<<grid_size,
                                                                block_size,
                                                                0,
                                                                stream>>>(
          x_data,
          y_data,
          out_data,
          dout_data,
          pre,
          mid,
          post,
          k_pre,
          k_mid,
          k_post,
          true,
          dx_op,
          dx_data);
    } else {
      k_pre = std::accumulate(
          x_dims_array, x_dims_array + axis, 1, std::multiplies<int>());
      k_mid = x_dims_array[axis];
      k_post = std::accumulate(x_dims_array + axis + 1,
                               x_dims_array + max_dim,
                               1,
                               std::multiplies<int>());
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
      int grid_size = pre * post;
      if (k_pre != pre) k_pre = pre / k_pre;

      paddle::operators::FastCommonGradBroadcastOneCUDAKernel<<<grid_size,
                                                                block_size,
                                                                0,
                                                                stream>>>(
          x_data,
          y_data,
          out_data,
          dout_data,
          pre,
          mid,
          post,
          k_pre,
          k_mid,
          k_post,
          false,
          dy_op,
          dy_data);
    }
    VLOG(3) << "FastBroadCastOneCUDAF pre:" << pre << " mid:" << mid
            << " post:" << post;
  };

  // do fast elementwise if: 1. only one input need to do broadcast, we can
  // fallback
  // to old fast path.
  // 2. if both x and y need broadcast, then do it one by one.
  bool fast_broadcast = false;
  if (x_broadcast_pos.empty() && !y_broadcast_pos.empty()) {
    can_split_y = paddle::operators::SplitDims(y_broadcast_pos, max_dim);
    if (can_split_y) {
      // only y need to do broadcast on h
      if (y_broadcast_pos[0] == 0) {
        FastBroadCastHeightCUDAF(y_broadcast_pos, true);
        fast_broadcast = true;
      }
    } else if (y_broadcast_pos.size() == 1 ||
               paddle::operators::CheckContiguousDims(
                   y_broadcast_pos)) {  // for only one dim and
                                        // contiguous broadcast.
      // If cannot split,  which means input has 3 parts
      FastBroadCastAllCUDAF(y_broadcast_pos, max_dim, true);
      fast_broadcast = true;
    }
  } else if (y_broadcast_pos.empty() && !x_broadcast_pos.empty()) {
    // only x need broadcast
    can_split_x = paddle::operators::SplitDims(x_broadcast_pos, max_dim);
    if (can_split_x) {
      if (x_broadcast_pos[0] == 0) {
        FastBroadCastHeightCUDAF(x_broadcast_pos, false);
        fast_broadcast = true;
      }
    } else if (x_broadcast_pos.size() == 1 ||
               paddle::operators::CheckContiguousDims(x_broadcast_pos)) {
      FastBroadCastAllCUDAF(x_broadcast_pos, max_dim, false);
      fast_broadcast = true;
    }
  } else if (!x_broadcast_pos.empty() && !y_broadcast_pos.empty()) {
    // do x and y broadcast each.
    can_split_y = paddle::operators::SplitDims(y_broadcast_pos, max_dim);
    bool fast_broadcast_x = false;
    bool fast_broadcast_y = false;
    if (can_split_y) {
      // begin at start.
      if (y_broadcast_pos[0] == 0) {
        FastCommonCUDAF(y_broadcast_pos, true);
        fast_broadcast_y = true;
      }
    } else if (y_broadcast_pos.size() == 1) {
      FastBroadCastOneCUDAF(y_broadcast_pos, max_dim, false);
      can_split_y = true;
      fast_broadcast_y = true;
    }
    can_split_x = paddle::operators::SplitDims(x_broadcast_pos, max_dim);
    if (can_split_x) {
      if (x_broadcast_pos[0] == 0) {
        FastCommonCUDAF(x_broadcast_pos, false);
        fast_broadcast_x = true;
      }
    } else if (x_broadcast_pos.size() == 1) {
      FastBroadCastOneCUDAF(x_broadcast_pos, max_dim, true);
      can_split_x = true;
      fast_broadcast_x = true;
    }
    VLOG(3) << "CommonBroadcast can_split_y:" << can_split_y
            << " can_split_x:" << can_split_x;
    // if both x and y into fast path then return
    if (fast_broadcast_x && fast_broadcast_y) {
      fast_broadcast = true;
    }
    if (can_split_y && can_split_x && fast_broadcast) return;
  }

  // Should remove memory copy, use reg instead.
  if (fast_broadcast) {
    return;
  }
  int x_blocks = 0;
  int x_threads = 0;
  paddle::operators::ComputeBroadcastKernelSize(
      x_dims_array, out_dims_array, &x_blocks, &x_threads, max_dim);
  int y_blocks = 0;
  int y_threads = 0;
  paddle::operators::ComputeBroadcastKernelSize(
      y_dims_array, out_dims_array, &y_blocks, &y_threads, max_dim);

  auto x_strides_array_tmp = paddle::memory::Alloc(ctx, bytes);
  int* x_strides_array_gpu = reinterpret_cast<int*>(x_strides_array_tmp->ptr());
  paddle::memory::Copy(gplace,
                       x_strides_array_gpu,
                       cplace,
                       x_strides_array.data(),
                       bytes,
                       ctx.stream());

  auto y_strides_array_tmp = paddle::memory::Alloc(ctx, bytes);
  int* y_strides_array_gpu = reinterpret_cast<int*>(y_strides_array_tmp->ptr());
  paddle::memory::Copy(gplace,
                       y_strides_array_gpu,
                       cplace,
                       y_strides_array.data(),
                       bytes,
                       ctx.stream());

  auto out_dims_array_tmp = paddle::memory::Alloc(ctx, bytes);
  int* out_dims_array_gpu = reinterpret_cast<int*>(out_dims_array_tmp->ptr());
  paddle::memory::Copy(
      gplace, out_dims_array_gpu, cplace, out_dims_array, bytes, ctx.stream());

  const int out_size = std::accumulate(
      out_dims_array, out_dims_array + max_dim, 1, std::multiplies<int>());
  int x_block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, x_threads);
  int y_block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, y_threads);
  if (dx) {
    auto x_strides_order_tmp = paddle::memory::Alloc(ctx, bytes);
    int* x_strides_order_gpu =
        reinterpret_cast<int*>(x_strides_order_tmp->ptr());
    paddle::memory::Copy(gplace,
                         x_strides_order_gpu,
                         cplace,
                         x_strides_order.data(),
                         bytes,
                         ctx.stream());

    auto x_dims_order_tmp = paddle::memory::Alloc(ctx, bytes);
    int* x_dims_order_gpu = reinterpret_cast<int*>(x_dims_order_tmp->ptr());
    paddle::memory::Copy(gplace,
                         x_dims_order_gpu,
                         cplace,
                         x_dims_order.data(),
                         bytes,
                         ctx.stream());
    paddle::operators::CommonGradBroadcastCUDAKernel<
        T,
        DX_OP><<<x_blocks, x_block_size, 0, ctx.stream()>>>(x_strides_array_gpu,
                                                            y_strides_array_gpu,
                                                            out_dims_array_gpu,
                                                            x_strides_order_gpu,
                                                            x_dims_order_gpu,
                                                            x_data,
                                                            y_data,
                                                            out_data,
                                                            dout_data,
                                                            dx_data,
                                                            out_size,
                                                            max_dim,
                                                            x_threads,
                                                            dx_op);
  }
  if (dy) {
    auto y_strides_order_tmp = paddle::memory::Alloc(ctx, bytes);
    int* y_strides_order_gpu =
        reinterpret_cast<int*>(y_strides_order_tmp->ptr());
    paddle::memory::Copy(gplace,
                         y_strides_order_gpu,
                         cplace,
                         y_strides_order.data(),
                         bytes,
                         ctx.stream());

    auto y_dims_order_tmp = paddle::memory::Alloc(ctx, bytes);
    int* y_dims_order_gpu = reinterpret_cast<int*>(y_dims_order_tmp->ptr());
    paddle::memory::Copy(gplace,
                         y_dims_order_gpu,
                         cplace,
                         y_dims_order.data(),
                         bytes,
                         ctx.stream());
    paddle::operators::CommonGradBroadcastCUDAKernel<
        T,
        DY_OP><<<y_blocks, y_block_size, 0, ctx.stream()>>>(x_strides_array_gpu,
                                                            y_strides_array_gpu,
                                                            out_dims_array_gpu,
                                                            y_strides_order_gpu,
                                                            y_dims_order_gpu,
                                                            x_data,
                                                            y_data,
                                                            out_data,
                                                            dout_data,
                                                            dy_data,
                                                            out_size,
                                                            max_dim,
                                                            y_threads,
                                                            dy_op);
  }
}

// reduce config
template <typename Ty>
struct ReduceConfig {
  ReduceConfig(const std::vector<int>& origin_reduce_dims,
               const std::vector<int>& origin_x_dim)
      : reduce_dims_origin(origin_reduce_dims), x_dim(origin_x_dim) {}

  // get the parameters of reduceKernel
  void Run() {
    // step1: update the reduce_dim left_dim and x_dim
    SetReduceDim();

    // step2: get the strides of dim for reduceAny and reduceLastDim
    SetStrides();

    // step3: get the type of reduce
    SetReduceType();

    // step4: set the block and grid for launch kernel
    SetBlockDim();
  }

  // when should_reduce_again is true, we need malloc temp space for temp data
  void SetOutputData(Ty* y_data, const TensorMeta& meta) {
    if (should_reduce_again) {
      TensorMeta tmp_meta(paddle::framework::make_ddim({static_cast<int64_t>(
                              left_num * grid.z * grid.y * sizeof(Ty))}),
                          meta.backend,
                          meta.type,
                          meta.layout,
                          meta.offset);
      DenseTensor tmp_tensor(tmp_meta, TensorStatus());

      output_data = tmp_tensor.mutable_data<Ty>();
    } else {
      output_data = y_data;
    }
  }

 private:
  // set reduce_dim, left_dim and update x_dim
  // eg: x_dim = [2, 4, 6] origin_reduce_dims = [0, 1]
  //     --SetReduceDim--> x_dim = [8,6], reduce_dim = [0], left_dim = [1]
  void SetReduceDim() {
    std::set<int> reduce_set;
    for (auto e : reduce_dims_origin) {
      auto pos = e >= 0 ? e : e + x_dim.size();
      reduce_set.insert(pos);
    }

    std::vector<int> reduce_dim_temp(reduce_set.begin(), reduce_set.end());
    std::sort(reduce_dim_temp.begin(), reduce_dim_temp.end());

    // update reduce_dim and x_dim
    std::vector<int> x_new_dim;

    reduce_dim.push_back(reduce_dim_temp[0]);
    x_new_dim.push_back(x_dim[0]);

    int idx_reduce = 1;
    int num = 0;

    if (reduce_dim_temp.size() > 1) {
      for (int i = 1; i < x_dim.size(); i++) {
        if ((idx_reduce < reduce_dim_temp.size()) &&
            (i == reduce_dim_temp[idx_reduce])) {
          int result =
              reduce_dim_temp[idx_reduce] - reduce_dim[reduce_dim.size() - 1];
          bool is_equal = ((result - num) == 1);
          if (is_equal) {
            x_new_dim[x_new_dim.size() - 1] *= x_dim[i];
            num++;
          } else {
            reduce_dim.push_back(reduce_dim_temp[idx_reduce] - num);
            x_new_dim.push_back(x_dim[i]);
          }
          idx_reduce++;
        } else {
          x_new_dim.push_back(x_dim[i]);
        }
      }
    } else {
      x_new_dim = x_dim;
    }

    // update x_dim
    x_dim = x_new_dim;
    std::vector<int>().swap(x_new_dim);

    std::vector<int> reduce_dim_new;
    int is_reduced = 0;
    for (auto e : reduce_dim) {
      is_reduced |= 1 << e;
    }

    std::vector<int>().swap(reduce_dim);

    for (int i = 0; i < x_dim.size(); i++) {
      if ((i == 0) || (((is_reduced >> i) ^ (is_reduced >> (i - 1))) & 1)) {
        x_new_dim.push_back(x_dim[i]);
        if ((is_reduced >> i) & 1)
          reduce_dim_new.push_back(x_new_dim.size() - 1);
      } else {
        x_new_dim[x_new_dim.size() - 1] *= x_dim[i];
      }
    }

    x_dim = x_new_dim;
    reduce_dim = reduce_dim_new;

    int x_rank = static_cast<int>(x_dim.size());
    std::set<int> left_set;

    for (int i = 0; i < x_rank; ++i) {
      left_set.insert(i);
    }

    for (auto e : reduce_dim) {
      left_set.erase(e);
    }

    left_dim.assign(left_set.begin(), left_set.end());

    // if the last dim gets involved in reduction
    reduce_last_dim = (reduce_dim.back() == x_dim.size() - 1);
  }

  // set x_strides, reduce_strides, left_strides for reduceLastDim and reduceAny
  // eg: x_dim = [8, 6], reduce_dim = [0], left_dim = [1]
  //     --SetStrides--> x_strides= [6,1], reduce_strides = [1],
  //     left_strides = [1]
  void SetStrides() {
    std::vector<int> idx_dim;
    for (int i = 0; i < x_dim.size(); i++) {
      idx_dim.push_back(i);
    }

    x_strides = paddle::operators::details::GetDimStrides(x_dim, idx_dim);
    reduce_strides =
        paddle::operators::details::GetDimStrides(x_dim, reduce_dim);
    left_strides = paddle::operators::details::GetDimStrides(x_dim, left_dim);
    reduce_num = reduce_strides[0] * x_dim[reduce_dim[0]];

    left_num = 1;
    if (left_dim.size()) {
      left_num = left_strides[0] * x_dim[left_dim[0]];
    }
  }

  // get the reduceType
  // eg: x_dim = [8, 6] reduce_dim = [0] --> ReduceHigherDim -->reduceFirstDim
  //     x_dim = [8, 6] reduce_dim = [1] --> reduceLastDim
  //     x_dim = [8] reduce_dim = [0] --> reduceAll
  //     x_dim = [8, 6, 4, 2] reduce_dim = [0, 2] --> reduceAny
  void SetReduceType() {
    int rank = x_dim.size();
    int reduce_rank = reduce_dim.size();
    bool is_last_dim =
        (rank == 2) && (reduce_rank == 1) && (reduce_dim[0] == 1);
    if (rank == reduce_rank || is_last_dim) {
      reduce_type =
          static_cast<int>(paddle::operators::ReduceType::kReduceLastDim);
    } else if (reduce_rank == 1) {
      // ReduceFirstDim and reduceSecondDim
      reduce_type =
          static_cast<int>(paddle::operators::ReduceType::kReduceHigherDim);
    } else {
      reduce_type = static_cast<int>(paddle::operators::ReduceType::kReduceAny);
    }
  }

  void SetBlockDimForReduceAny(dim3* block_dim, dim3* grid_dim) {
    constexpr int min_reduce_num_per_thread = 16;
    constexpr int max_reduce_num_per_thread = 256;
    constexpr int max_num_threads = kps::details::kReduceMaxThread;

    // set block size.
    // 1. If reduce_last_dim == true, all the threads whose threadIdx.y are same
    //    will process the reduction for one output.
    //    The number of output for one block is blockDim.y;
    // 2. If reduce_last_dim == false, different threadIdx.x will process
    //    different reduction and gets the output separately. If it is
    //    necessary, it should reduce in block y.
    //    The number of output for one block is blockDim.x;
    int block_x, block_y;
    int grid_num, reduce_num_per_thread;
    if (reduce_last_dim) {
      block_x = paddle::operators::details::GetBlockDim(reduce_num);
      block_y = paddle::operators::details::GetBlockDim(left_num);
      block_dim->x = block_x;
      block_dim->y =
          std::min(block_y, static_cast<int>(max_num_threads / block_dim->x));
      grid_num = paddle::operators::details::AlignUp(left_num, block_dim->y);
      reduce_num_per_thread =
          paddle::operators::details::AlignUp(reduce_num, block_dim->x);
    } else {
      block_x = paddle::operators::details::GetBlockDim(left_num);
      block_y = paddle::operators::details::GetBlockDim(reduce_num);
      block_dim->x = std::min(block_x, 32);
      block_dim->y =
          std::min(block_y, static_cast<int>(max_num_threads / block_dim->x));
      block_dim->x =
          std::min(block_x, static_cast<int>(max_num_threads / block_dim->y));
      grid_num = paddle::operators::details::AlignUp(left_num, block_dim->x);
      reduce_num_per_thread =
          paddle::operators::details::AlignUp(reduce_num, block_dim->y);
    }
    int device_id = paddle::platform::GetCurrentDeviceId();
    int max_mp = paddle::platform::GetCUDAMultiProcessors(device_id);
    int max_threads_per_mp =
        paddle::platform::GetCUDAMaxThreadsPerMultiProcessor(device_id);
    int max_threads = max_threads_per_mp * max_mp;
    int num_threads = block_dim->x * block_dim->y;
    int max_num_blocks = max_threads / num_threads;

    // set grid size.
    // Whether to set grid.y larger than 1, there are 3 following rules:
    // 1. The number that each thread process should no less than
    //    min_reduce_num_per_threadbut no more than max_reduce_num_per_thread;
    // 2. It should maximize the utilization of SM.
    // So we choose the minimum between input_split_num_1 and input_split_num_3
    // to make each thread process as mush data as possible. Meanwhile,
    // the number cannot be larger than max_reduce_num_per_thread, so we
    // choose the maximum between the result above and input_split_num_2.
    int input_split_num_1 = paddle::operators::details::AlignUp(
        reduce_num_per_thread, min_reduce_num_per_thread);
    int input_split_num_2 = paddle::operators::details::AlignUp(
        reduce_num_per_thread, max_reduce_num_per_thread);
    int input_split_num_3 =
        paddle::operators::details::AlignUp(max_num_blocks, grid_num);

    grid_dim->x = grid_num;
    grid_dim->y = std::max(std::min(input_split_num_1, input_split_num_3),
                           input_split_num_2);
    // if grid.y > 1, we need launch reduce kernel again.
    if (grid_dim->y > 1) {
      should_reduce_again = true;
    }
  }

  // set block and grid for launch kernel
  // for ReduceHigherDim: if block is enough -> splite reduce_num
  //                     else init block(32, 1) grid(block_num, 1)
  // for others: block(block_num, 1) , grid(left_num, 1)
  void SetBlockDim() {
    // init
    int block_num = paddle::operators::details::GetBlockDim(reduce_num);
    should_reduce_again = false;

    dim3 block_dim(block_num, 1);
    dim3 grid_dim(left_num, 1);
    blocking_size = reduce_num;

    if (reduce_type == paddle::operators::ReduceType::kReduceHigherDim) {
      int last_dim_num = x_dim.back();
      // update left_num
      int grid_z = left_num / last_dim_num;
      left_num = last_dim_num;

      block_dim.z = 1;
      grid_dim.z = grid_z;

      int device_id = paddle::platform::GetCurrentDeviceId();
      int max_mp = paddle::platform::GetCUDAMultiProcessors(device_id);
      int max_threads_per_mp =
          paddle::platform::GetCUDAMaxThreadsPerMultiProcessor(device_id);
      int max_threads = max_threads_per_mp * max_mp;

      // init
      int num_block = (max_threads / left_num);

      if (num_block > 1 && reduce_num >= REDUCE_SPLIT_BOUNDARY) {
        blocking_size =
            paddle::operators::details::GetLastPow2(reduce_num / num_block);

        if (blocking_size <= 1) {
          blocking_size =
              paddle::operators::details::GetLastPow2(sqrt(reduce_num));
        } else if (blocking_size * 2 < reduce_num) {
          blocking_size *= 2;
        }

        should_reduce_again = true;

        block_dim.x = paddle::operators::details::GetBlockDim(left_num);
        block_dim.y = 1;
        grid_dim.x = (left_num + block_dim.x - 1) / block_dim.x;
        grid_dim.y = (reduce_num + blocking_size - 1) / blocking_size;

      } else {
        block_dim.x = paddle::operators::details::GetBlockDim(left_num);
        block_dim.y = 1;
        blocking_size = reduce_num;
        grid_dim.x = (left_num + block_dim.x - 1) / block_dim.x;
        grid_dim.y = 1;
      }
    } else {
      SetBlockDimForReduceAny(&block_dim, &grid_dim);
    }

    block = block_dim;
    grid = grid_dim;
  }

 public:
  std::vector<int> reduce_dims_origin;
  std::vector<int> reduce_dim;
  std::vector<int> x_dim;
  std::vector<int> left_dim;
  std::vector<int> x_strides;
  std::vector<int> left_strides;
  std::vector<int> reduce_strides;

  int reduce_type;
  int reduce_num;
  int left_num;
  int blocking_size;
  bool should_reduce_again;
  bool reduce_last_dim;

  Ty* output_data;

  dim3 block;
  dim3 grid;
};

template <typename T>
static __global__ void SimpleElemwiseAddGradCUDAKernel(
    const T* __restrict__ dout, int size, int vec_size, T* dx, T* dy) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int loop = size / vec_size;
  int remainder = size % vec_size;
  const float4* dout_vec = reinterpret_cast<const float4*>(dout);
  float4* dx_vec = reinterpret_cast<float4*>(dx);
  float4* dy_vec = reinterpret_cast<float4*>(dy);
  float4 tmp_loop;

  for (int i = tid; i < loop; i += stride) {
    tmp_loop = dout_vec[i];
    dx_vec[i] = tmp_loop;
    dy_vec[i] = tmp_loop;
  }

  if (tid == loop && remainder != 0) {
    T tmp_rem;
    while (remainder) {
      int idx = size - remainder;
      remainder--;
      tmp_rem = dout[idx];
      dx[idx] = tmp_rem;
      dy[idx] = tmp_rem;
    }
  }
}

// cuda definition
template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext,
                 paddle::platform::CUDADeviceContext>::value>::type
elementwise_add_grad(const DeviceContext& ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const DenseTensor& out,
                     const DenseTensor& dout,
                     int axis,
                     DenseTensor* dx,
                     DenseTensor* dy) {
  auto* dx_data = dx->mutable_data<T>();
  auto* dy_data = dy->mutable_data<T>();
  auto* dout_data = dout.data<T>();
  if (dx_data == dout_data && dy_data != dout_data) {
    VLOG(4) << "Special case when dx_data is the same as dout_data, "
               "only need copy dout to dy";
    pten::Copy(ctx, dout, dy);

  } else if (dx_data != dout_data && dy_data == dout_data) {
    VLOG(4) << "Special case when dy_data is the same as dout_data, "
               "only need copy dout to dx";
    pten::Copy(ctx, dout, dx);

  } else if (dx_data != dout_data && dy_data != dout_data) {
    auto size = x.numel();
    int vec_size = max(static_cast<int>(sizeof(float4) / sizeof(T)), 1);
    dim3 block_size = dim3(ELEMENTWISE_BLOCK_SIZE, 1);
    dim3 grid_size =
        dim3(((size + vec_size - 1) / vec_size + ELEMENTWISE_BLOCK_SIZE - 1) /
                 ELEMENTWISE_BLOCK_SIZE,
             1);
    SimpleElemwiseAddGradCUDAKernel<
        T><<<grid_size, block_size, 0, ctx.stream()>>>(dout.data<T>(),
                                                       size,
                                                       vec_size,
                                                       dx->mutable_data<T>(),
                                                       dy->mutable_data<T>());
  } else {
    VLOG(4) << "Special case when dy_data is the same as dout_data, "
               "and dx_data is the same as dout_data, do not need "
               "any operator";
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext,
                 paddle::platform::CUDADeviceContext>::value>::type
default_elementwise_add_grad(const DeviceContext& ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             const DenseTensor& out,
                             const DenseTensor& dout,
                             int axis,
                             DenseTensor* dx,
                             DenseTensor* dy) {
  auto* dout_data = dout.data<T>();

  // dx
  if (dx != nullptr) {
    auto* dx_data = dx->mutable_data<T>();
    if (dx->dims() == dout.dims()) {
      if (dx_data != dout_data) {
        pten::Copy(ctx, dout, dx);
      }
    } else {
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.

      /* Skip Inplace Strategy for now
      if (dx->IsSharedBufferWith(dout)) {
        dx->clear();
        dx->mutable_data<T>(x.dims(), ctx.GetPlace());
      }
      */
      std::vector<int> reduce_dims =
          paddle::operators::GetReduceDim(x.dims(), out.dims(), axis);
      TensorReduceFunctorImpl<T, T, CustomSum>(ctx, dout, dx, reduce_dims);
    }
  }
  // dy
  if (dy != nullptr) {
    auto* dy_data = dy->mutable_data<T>();
    if (dy->dims() == dout.dims()) {
      if (dy_data != dout_data) {
        pten::Copy(ctx, dout, dy);
      }
    } else {
      std::vector<int> reduce_dims =
          paddle::operators::GetReduceDim(y.dims(), out.dims(), axis);
      TensorReduceFunctorImpl<T, T, CustomSum>(ctx, dout, dy, reduce_dims);
    }
  }
}

}  // namespace math
}  // namespace pten

#endif
