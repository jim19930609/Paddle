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

from auto_scan_test import PassAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


class TestConvConcatReluMkldnnFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]
        # If the problem has been fixed, the judgment 
        # needs to be deleted!!!
        if attrs[0]['data_format'] == "NHWC":
            return False

        return True

    def sample_program_config(self, draw):
        data_format = draw(st.sampled_from(["NCHW", "NHWC"]))
        dilations = draw(st.sampled_from([[1, 1], [2, 2], [1, 2]]))
        padding_algorithm = draw(st.sampled_from(["EXPLICIT", "SAME", "VALID"]))
        groups = draw(st.sampled_from([1, 2, 4]))
        paddings = draw(st.sampled_from([[0, 3], [1, 2, 3, 4]]))
        strides = draw(st.sampled_from([[1, 1], [2, 2], [1, 2]]))
        axis = draw(st.sampled_from([0]))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input(attrs):
            if attrs[0]['data_format'] == "NCHW":
                return np.random.random(
                    [attrs[2]['batch_size'], 48, 64, 64]).astype(np.float32)
            else:
                return np.random.random(
                    [attrs[2]['batch_size'], 64, 64, 48]).astype(np.float32)

        def generate_weight():
            return np.random.random(
                [16, int(48 / groups), 3, 3]).astype(np.float32)

        attrs = [{
            "data_format": data_format,
            "dilations": dilations,
            "padding_algorithm": padding_algorithm,
            "groups": groups,
            "paddings": paddings,
            "strides": strides
        }, {
            "axis": axis
        }, {
            'batch_size': batch_size
        }]

        ops_config = [{
            "op_type": "conv2d",
            "op_inputs": {
                "Input": ["input_data1"],
                "Filter": ["input_weight"]
            },
            "op_outputs": {
                "Output": ["conv1_output"]
            },
            "op_attrs": {
                "data_format": attrs[0]['data_format'],
                "dilations": attrs[0]['dilations'],
                "padding_algorithm": attrs[0]['padding_algorithm'],
                "groups": attrs[0]['groups'],
                "paddings": attrs[0]['paddings'],
                "strides": attrs[0]['strides']
            }
        }, {
            "op_type": "conv2d",
            "op_inputs": {
                "Input": ["input_data2"],
                "Filter": ["input_weight"]
            },
            "op_outputs": {
                "Output": ["conv2_output"]
            },
            "op_attrs": {
                "data_format": attrs[0]['data_format'],
                "dilations": attrs[0]['dilations'],
                "padding_algorithm": attrs[0]['padding_algorithm'],
                "groups": attrs[0]['groups'],
                "paddings": attrs[0]['paddings'],
                "strides": attrs[0]['strides']
            }
        }, {
            "op_type": "concat",
            "op_inputs": {
                "X": ["conv1_output", "conv2_output"]
            },
            "op_outputs": {
                "Out": ["concat_output"]
            },
            "op_attrs": {
                'axis': attrs[1]['axis']
            }
        }, {
            "op_type": "relu",
            "op_inputs": {
                "X": ["concat_output"]
            },
            "op_outputs": {
                "Out": ["relu_output"]
            },
            "op_attrs": {}
        }]

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "input_weight": TensorConfig(data_gen=partial(generate_weight))
            },
            inputs={
                "input_data1":
                TensorConfig(data_gen=partial(generate_input, attrs)),
                "input_data2":
                TensorConfig(data_gen=partial(generate_input, attrs))
            },
            outputs=["relu_output"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ["conv2d", "conv2d", "concat"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False, passes=["conv_concat_relu_mkldnn_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
