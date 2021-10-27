/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// See Note [ Why still include the fluid headers? ]
#include "paddle/pten/infershape/unary.h"

namespace pten {

TensorMeta UnchangedInferShape(const TensorMeta& x_meta) { return x_meta; }

TensorMeta ReductionInferShape(const TensorMeta& x_meta) {
  const auto& out_dims = paddle::framework::make_ddim({1});
  TensorMeta return_meta(
      out_dims, x_meta.backend, x_meta.type, x_meta.layout, x_meta.offset);
  return return_meta;
}

TensorMeta GenericReductionInferShape(const TensorMeta& x_meta,
                                      bool reduce_all,
                                      const std::vector<int>& dim,
                                      bool keep_dim,
                                      int out_dtype) {
  auto x_dims = x_meta.dims;
  auto x_rank = x_dims.size();
  std::vector<int> dims = dim;
  PADDLE_ENFORCE_GT(dims.size(),
                    0,
                    paddle::platform::errors::InvalidArgument(
                        "The input dim dimensions of ReduceOp "
                        "should be greater than 0. But received the dim "
                        "dimesions of Reduce = %d.",
                        dims.size()));

  for (size_t i = 0; i < dims.size(); ++i) {
    PADDLE_ENFORCE_LT(dims[i],
                      x_rank,
                      paddle::platform::errors::InvalidArgument(
                          "The reduce dim index %d should be in the "
                          "range [-dimension(X), dimension(X)] "
                          "which dimesion = %d. But received dim index = %d.",
                          i,
                          x_rank,
                          dims[i]));
    PADDLE_ENFORCE_GE(dims[i],
                      -x_rank,
                      paddle::platform::errors::InvalidArgument(
                          "The reduce dim index %d should be in the "
                          "range [-dimension(X), dimension(X)] "
                          "which dimesion = %d. But received dim index = %d.",
                          i,
                          x_rank,
                          dims[i]));
    if (dims[i] < 0) dims[i] = x_rank + dims[i];
  }

  sort(dims.begin(), dims.end());

  TensorMeta return_meta(
      x_meta.dims, x_meta.backend, x_meta.type, x_meta.layout, x_meta.offset);

  if (reduce_all) {
    if (keep_dim)
      return_meta.dims =
          paddle::framework::make_ddim(std::vector<int64_t>(x_rank, 1));
    else
      return_meta.dims = paddle::framework::make_ddim({1});

  } else {
    auto dims_vector = vectorize(x_dims);
    if (keep_dim) {
      for (size_t i = 0; i < dims.size(); ++i) {
        dims_vector[dims[i]] = 1;
      }
    } else {
      const int kDelFlag = -2;
      for (size_t i = 0; i < dims.size(); ++i) {
        dims_vector[dims[i]] = kDelFlag;
      }
      dims_vector.erase(
          remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
          dims_vector.end());
    }
    if (!keep_dim && dims_vector.size() == 0) {
      dims_vector.push_back(1);
    }
    auto out_dims = paddle::framework::make_ddim(dims_vector);
    return_meta.dims = out_dims;
    if (dims.size() > 0 && dims[0] != 0) {
      // Only pass LoD when not reducing on the first dim.
      return_meta.lod = x_meta.lod;
    }
  }

  return return_meta;
}

TensorMeta FlattenInferShape(const TensorMeta& x_meta,
                             int start_axis,
                             int stop_axis) {
  auto& x_dims = x_meta.dims;
  int in_dims_size = x_dims.size();
  if (start_axis < 0) {
    start_axis = start_axis + in_dims_size;
  }
  if (stop_axis < 0) {
    stop_axis = stop_axis + in_dims_size;
  }
  PADDLE_ENFORCE_GE(stop_axis,
                    start_axis,
                    paddle::platform::errors::InvalidArgument(
                        "The stop_axis should be greater"
                        "than or equal to start_axis."));

  int64_t outer = 1;
  std::vector<int32_t> out_shape;
  out_shape.reserve(in_dims_size - stop_axis + start_axis);

  for (int i = 0; i < start_axis; ++i) {
    out_shape.push_back(x_dims[i]);
  }
  for (int i = start_axis; i <= stop_axis; i++) {
    if (x_dims[i] == -1 || outer == -1) {
      outer = -1;
    } else {
      outer *= x_dims[i];
    }
  }
  out_shape.push_back(outer);
  for (int i = stop_axis + 1; i < in_dims_size; i++) {
    out_shape.push_back(x_dims[i]);
  }
  const auto& out_dims = paddle::framework::make_ddim(out_shape);
  TensorMeta return_meta(
      out_dims, x_meta.backend, x_meta.type, x_meta.layout, x_meta.offset);

  if (x_dims[0] == return_meta.dims[0]) {
    // Only pass LoD when the first dimension of output and Input(X)
    // are the same.
    return_meta.lod = x_meta.lod;
  }

  return return_meta;
}

}  // namespace pten
