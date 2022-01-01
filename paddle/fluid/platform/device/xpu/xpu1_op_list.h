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
#pragma once

#ifdef PADDLE_WITH_XPU
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/framework/op_kernel_type.h"

namespace paddle {
namespace platform {

using vartype = paddle::framework::proto::VarType;
using pOpKernelType = paddle::framework::OpKernelType;
using XPUKernelSet =
    std::unordered_set<pOpKernelType, paddle::framework::OpKernelType::Hash>;
using XPUOpMap = std::unordered_map<std::string, XPUKernelSet>;

XPUOpMap& get_kl1_ops() {
  // KL1支持的op，通过op_name, data_type, place来索引
  static XPUOpMap s_xpu1_kernels{
      {"abs", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"accuracy", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"adam", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"adamw", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"affine_channel_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"affine_channel",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"arg_max", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"assign", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                               pOpKernelType(vartype::FP64, XPUPlace()),
                               pOpKernelType(vartype::INT32, XPUPlace()),
                               pOpKernelType(vartype::INT64, XPUPlace()),
                               pOpKernelType(vartype::BOOL, XPUPlace())})},
      {"batch_norm_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"batch_norm", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"bilinear_interp",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"bilinear_interp_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"bilinear_interp_v2",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"bilinear_interp_v2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"broadcast", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                                  pOpKernelType(vartype::FP64, XPUPlace()),
                                  pOpKernelType(vartype::INT32, XPUPlace()),
                                  pOpKernelType(vartype::INT64, XPUPlace())})},
      {"cast", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                             pOpKernelType(vartype::INT64, XPUPlace()),
                             pOpKernelType(vartype::INT32, XPUPlace())})},
      {"clip_by_norm",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"coalesce_tensor",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                     pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace())})},
      {"concat", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"concat_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"conv2d", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"conv2d_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"deformable_conv",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"deformable_conv_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"depthwise_conv2d",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"depthwise_conv2d_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"dropout", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"dropout_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"c_allreduce_sum",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"c_reduce_sum",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_add",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_add_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_div_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_div",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_floordiv",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_max_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_max",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_min_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_min",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_mul_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_mul",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_pow",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_sub_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"elementwise_sub",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"equal", XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace())})},
      {"expand_as_v2",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::FP16, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"expand_v2", XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                                  pOpKernelType(vartype::INT64, XPUPlace()),
                                  pOpKernelType(vartype::BOOL, XPUPlace()),
                                  pOpKernelType(vartype::FP16, XPUPlace()),
                                  pOpKernelType(vartype::FP32, XPUPlace())})},
      {"fill_any_like",
       XPUKernelSet({pOpKernelType(vartype::INT64, XPUPlace())})},
      {"fill_constant",
       XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"gather_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"gather", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"gaussian_random",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"gelu_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"gelu", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"hard_switch_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"hard_switch", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"iou_similarity",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"lamb", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"layer_norm_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"layer_norm", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"leaky_relu_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"leaky_relu", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"load", XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                             pOpKernelType(vartype::INT8, XPUPlace()),
                             pOpKernelType(vartype::INT32, XPUPlace()),
                             pOpKernelType(vartype::INT64, XPUPlace()),
                             pOpKernelType(vartype::FP32, XPUPlace())})},
      {"logicaland", XPUKernelSet({pOpKernelType(vartype::BOOL, XPUPlace()),
                                   pOpKernelType(vartype::INT8, XPUPlace()),
                                   pOpKernelType(vartype::INT16, XPUPlace()),
                                   pOpKernelType(vartype::INT32, XPUPlace()),
                                   pOpKernelType(vartype::INT64, XPUPlace()),
                                   pOpKernelType(vartype::FP32, XPUPlace())})},
      {"logicalnot", XPUKernelSet({pOpKernelType(vartype::BOOL, XPUPlace()),
                                   pOpKernelType(vartype::INT8, XPUPlace()),
                                   pOpKernelType(vartype::INT16, XPUPlace()),
                                   pOpKernelType(vartype::INT32, XPUPlace()),
                                   pOpKernelType(vartype::INT64, XPUPlace()),
                                   pOpKernelType(vartype::FP32, XPUPlace())})},
      {"logicalor", XPUKernelSet({pOpKernelType(vartype::BOOL, XPUPlace()),
                                  pOpKernelType(vartype::INT8, XPUPlace()),
                                  pOpKernelType(vartype::INT16, XPUPlace()),
                                  pOpKernelType(vartype::INT32, XPUPlace()),
                                  pOpKernelType(vartype::INT64, XPUPlace()),
                                  pOpKernelType(vartype::FP32, XPUPlace())})},
      {"log_loss_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"log_loss", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"logsumexp", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"log", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"lookup_table_v2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"lookup_table_v2",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"matmul_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"matmul_v2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"matmul_v2", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"matmul", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"mean_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"mean", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"momentum", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"mul_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"mul", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"nearest_interp_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"nearest_interp_v2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"nearest_interp_v2",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"nearest_interp",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"one_hot_v2", XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                                   pOpKernelType(vartype::INT64, XPUPlace())})},
      {"one_hot", XPUKernelSet({pOpKernelType(vartype::INT32, XPUPlace()),
                                pOpKernelType(vartype::INT64, XPUPlace())})},
      {"pool2d_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"pool2d", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"pow", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"range", XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                              pOpKernelType(vartype::INT64, XPUPlace()),
                              pOpKernelType(vartype::INT32, XPUPlace()),
                              pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_max_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_max", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_mean", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_mean_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_prod", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_sum_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reduce_sum", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"relu_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"relu", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reshape2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"reshape2", XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                                 pOpKernelType(vartype::INT64, XPUPlace()),
                                 pOpKernelType(vartype::INT32, XPUPlace()),
                                 pOpKernelType(vartype::BOOL, XPUPlace()),
                                 pOpKernelType(vartype::FP32, XPUPlace())})},
      {"rmsprop", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"rnn_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"rnn", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"roi_align_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"roi_align", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"scale", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sgd", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"shape", XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                              pOpKernelType(vartype::INT64, XPUPlace()),
                              pOpKernelType(vartype::INT32, XPUPlace()),
                              pOpKernelType(vartype::BOOL, XPUPlace()),
                              pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sigmoid_cross_entropy_with_logits_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sigmoid_cross_entropy_with_logits",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sigmoid_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sigmoid", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sign", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"slice_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"slice", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                              pOpKernelType(vartype::INT32, XPUPlace())})},
      {"softmax_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"softmax_with_cross_entropy",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"softmax_with_cross_entropy_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"softmax", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"split", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace()),
                              pOpKernelType(vartype::INT32, XPUPlace())})},
      {"sqrt_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sqrt", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"square_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"square", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"squeeze2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"squeeze2", XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                                 pOpKernelType(vartype::INT64, XPUPlace()),
                                 pOpKernelType(vartype::INT32, XPUPlace()),
                                 pOpKernelType(vartype::BOOL, XPUPlace()),
                                 pOpKernelType(vartype::INT8, XPUPlace()),
                                 pOpKernelType(vartype::UINT8, XPUPlace()),
                                 pOpKernelType(vartype::FP32, XPUPlace())})},
      {"squeeze_grad",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"squeeze", XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                                pOpKernelType(vartype::INT64, XPUPlace()),
                                pOpKernelType(vartype::INT32, XPUPlace()),
                                pOpKernelType(vartype::BOOL, XPUPlace()),
                                pOpKernelType(vartype::INT8, XPUPlace()),
                                pOpKernelType(vartype::UINT8, XPUPlace()),
                                pOpKernelType(vartype::FP32, XPUPlace())})},
      {"stack", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"sum", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"tanh_grad", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"tanh", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"top_k", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"transpose2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"transpose2", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"transpose_grad",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"transpose", XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"truncated_gaussian_random",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"uniform_random",
       XPUKernelSet({pOpKernelType(vartype::FP32, XPUPlace())})},
      {"unsqueeze2_grad",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"unsqueeze2", XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                                   pOpKernelType(vartype::INT64, XPUPlace()),
                                   pOpKernelType(vartype::INT32, XPUPlace()),
                                   pOpKernelType(vartype::BOOL, XPUPlace()),
                                   pOpKernelType(vartype::INT8, XPUPlace()),
                                   pOpKernelType(vartype::UINT8, XPUPlace()),
                                   pOpKernelType(vartype::FP32, XPUPlace())})},
      {"unsqueeze_grad",
       XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                     pOpKernelType(vartype::INT64, XPUPlace()),
                     pOpKernelType(vartype::INT32, XPUPlace()),
                     pOpKernelType(vartype::BOOL, XPUPlace()),
                     pOpKernelType(vartype::INT8, XPUPlace()),
                     pOpKernelType(vartype::UINT8, XPUPlace()),
                     pOpKernelType(vartype::FP32, XPUPlace())})},
      {"unsqueeze", XPUKernelSet({pOpKernelType(vartype::FP64, XPUPlace()),
                                  pOpKernelType(vartype::INT64, XPUPlace()),
                                  pOpKernelType(vartype::INT32, XPUPlace()),
                                  pOpKernelType(vartype::BOOL, XPUPlace()),
                                  pOpKernelType(vartype::INT8, XPUPlace()),
                                  pOpKernelType(vartype::UINT8, XPUPlace()),
                                  pOpKernelType(vartype::FP32, XPUPlace())})},
      {"where_index", XPUKernelSet({pOpKernelType(vartype::BOOL, XPUPlace())})},
      // AddMore
  };

  return s_xpu1_kernels;
}

}  // namespace platform
}  // namespace paddle
#endif
