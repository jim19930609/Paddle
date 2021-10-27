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

#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/kernels/functions/eigen/common.h"
#include "paddle/pten/kernels/cuda/reduce.h"
#include "paddle/pten/kernels/cuda/utils.h"

#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/platform/complex.h"

#include "paddle/pten/kernels/functions/math/reduce_function.h"
#include "paddle/pten/kernels/functions/math/transform_function.h"

namespace pten {

template <typename Tx, typename Ty = Tx>
struct IdentityFunctor {
  HOSTDEVICE inline IdentityFunctor() {}

  HOSTDEVICE explicit inline IdentityFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x);
  }
};

/* --------- Functors --------- */
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
          math::CastOpFunctor<CUDAContext, Tx>(x, y, dev_ctx));
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

static std::vector<int> GetReduceDim(const std::vector<int>& dims,
                                     int dim_size,
                                     bool reduce_all) {
  std::vector<int> reduce_dims;
  if (reduce_all) {
    reduce_dims.resize(dim_size);
    int reduce_size = reduce_dims.size();
    for (int i = 0; i < reduce_size; ++i) {
      reduce_dims[i] = i;
    }
  } else {
    for (auto e : dims) {
      PADDLE_ENFORCE_LT(e,
                        dim_size,
                        paddle::platform::errors::InvalidArgument(
                            "ReduceOp: invalid axis, when x_dims is %d, "
                            "axis[i] should less than x_dims, but got %d.",
                            dim_size,
                            e));
      reduce_dims.push_back(e >= 0 ? e : e + dim_size);
    }
  }
  return reduce_dims;
}

template <typename Tx, template <typename, typename> class ReduceOp>
struct TensorReduceFunc {
  const CUDAContext& dev;
  const DenseTensor& x;
  DenseTensor* y;
  std::vector<int> origin_reduce_dims;
  TensorReduceFunc(const CUDAContext& dev_ctx,
                   const DenseTensor& x,
                   DenseTensor* y,
                   std::vector<int> origin_reduce_dims)
      : dev(dev_ctx), x(x), y(y), origin_reduce_dims(origin_reduce_dims) {}

  template <typename Ty>
  void apply() const {
    TensorReduceFunctorImpl<Tx, Ty, ReduceOp>(dev, x, y, origin_reduce_dims);
  }
};

template <typename T, template <typename, typename> class ReduceOp>
void ReduceSumCudaKernel(const CUDAContext& dev_ctx,
                         const DenseTensor& x,
                         bool reduce_all,
                         const std::vector<int>& dim,
                         bool keep_dim,
                         int out_dtype,
                         DenseTensor* out) {
  std::vector<int> reduce_dims = GetReduceDim(dim, x.dims().size(), reduce_all);

  if (out_dtype >= 0) {
    paddle::framework::VisitDataTypeSmall(
        static_cast<paddle::framework::proto::VarType::Type>(out_dtype),
        TensorReduceFunc<T, ReduceOp>(dev_ctx, x, out, reduce_dims));
  } else {
    TensorReduceFunctorImpl<T, T, ReduceOp>(dev_ctx, x, out, reduce_dims);
  }
}

template <typename T>
void ReduceSum(const CUDAContext& dev_ctx,
               const DenseTensor& x,
               bool reduce_all,
               const std::vector<int>& dim,
               bool keep_dim,
               int out_dtype,
               DenseTensor* out) {
  ReduceSumCudaKernel<T, CustomSum>(
      dev_ctx, x, reduce_all, dim, keep_dim, out_dtype, out);
}

}  // namespace pt

// TODO(chenweihang): replace by better impl
PT_REGISTER_MODULE(ReduceCUDA);

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_KERNEL("reduce_sum",
                   CUDA,
                   ANY,
                   pten::ReduceSum,
                   bool,
                   float,
                   double,
                   paddle::platform::float16,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
