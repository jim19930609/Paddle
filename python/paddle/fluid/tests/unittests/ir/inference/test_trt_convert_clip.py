# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest


class TrtConvertClipTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(dims, batch, attrs: List[Dict[str, Any]]):
            if dims == 1:
                return np.ones([64]).astype(np.float32)
            elif dims == 2:
                return np.ones([3, 64]).astype(np.float32)
            elif dims == 3:
                return np.ones([3, 64, 64]).astype(np.float32)
            else:
                return np.ones([batch, 3, 64, 64]).astype(np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            return np.array([np.random.uniform(1, 10)]).astype("float32")

        def generate_weight2(attrs: List[Dict[str, Any]]):
            return np.array([np.random.uniform(10, 20)]).astype("float32")

        for dims in [1, 2, 3, 4]:
            for batch in [1, 2, 4]:
                for op_inputs in [{
                        "X": ["input_data"]
                }, {
                        "X": ["input_data"],
                        "Min": ["Min_"],
                        "Max": ["Max_"]
                }]:
                    self.input_num = len(op_inputs)
                    self.dims = dims
                    dics = [{
                        "min": np.random.uniform(1, 10),
                        "max": np.random.uniform(10, 20)
                    }, {
                        "op_inputs": op_inputs
                    }]
                    ops_config = [{
                        "op_type": "clip",
                        "op_inputs": op_inputs,
                        "op_outputs": {
                            "Out": ["output_data"]
                        },
                        "op_attrs": dics[0]
                    }]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={
                            "Min_": TensorConfig(data_gen=partial(
                                generate_weight1, dics)),
                            "Max_": TensorConfig(data_gen=partial(
                                generate_weight2, dics))
                        },
                        inputs={
                            "input_data": TensorConfig(data_gen=partial(
                                generate_input1, dims, batch, dics))
                        },
                        outputs=["output_data"])

                    yield program_config

    def sample_predictor_configs(self, program_config):
        def generate_dynamic_shape(attrs):
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {"input_data": [1]}
                self.dynamic_shape.max_input_shape = {"input_data": [128]}
                self.dynamic_shape.opt_input_shape = {"input_data": [64]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 32]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 64]}
                self.dynamic_shape.opt_input_shape = {"input_data": [3, 64]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 32, 32]}
                self.dynamic_shape.max_input_shape = {
                    "input_data": [10, 64, 64]
                }
                self.dynamic_shape.opt_input_shape = {"input_data": [3, 64, 64]}
            else:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 3, 32, 32]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 3, 64, 64]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [1, 3, 64, 64]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if self.input_num == 3 or self.dims == 1:
                return 0, 3
            else:
                return 1, 2

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if len(
                    program_config.inputs['input_data'].shape
            ) == 2 and not predictor_config.tensorrt_dynamic_shape_enabled():
                return True
            return False

        self.add_skip_case(
            teller1, SkipReasons.TRT_NOT_IMPLEMENTED,
            "The output shape has diff, but we can add shuffle layer to resolve it."
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
