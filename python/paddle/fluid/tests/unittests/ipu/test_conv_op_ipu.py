#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.optimizer
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_feed()
        self.set_attrs()

    def set_feed(self):
        self.feed_shape = []
        self.feed_shape.append([1, 3, 10, 10])

        self.feed = {}
        self.feed["in_0"] = np.random.uniform(
            size=self.feed_shape[0]).astype(np.float32)

        self.feed_list = list(self.feed.keys())

    def set_attrs(self):
        self.attrs = {}
        self.attrs['num_filters'] = 3
        self.attrs['filter_size'] = 3
        self.attrs['stride'] = 1
        self.attrs['padding'] = 0
        self.attrs['dilation'] = 1
        self.attrs['groups'] = 1
        self.attrs['data_format'] = 'NCHW'

    def _test_base(self, run_ipu=True):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = self.SEED
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                image = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype='float32')
                out = paddle.fluid.layers.conv2d(image, **self.attrs)

                fetch_list = [out.name]

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = self.feed_list
                ipu_strategy = compiler.get_ipu_strategy()
                ipu_strategy.is_training = self.is_training
                program = compiler.IPUCompiledProgram(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog

            result = exe.run(program, feed=self.feed, fetch_list=fetch_list)
            return result[0]

    def test_base(self):
        res0 = self._test_base(True)
        res1 = self._test_base(False)

        self.assertTrue(
            np.allclose(
                res0.flatten(), res1.flatten(), atol=self.atol))


class TestCase1(TestBase):
    def set_attrs(self):
        super().set_attrs()
        self.attrs['num_filters'] = 1


class TestCase2(TestBase):
    def set_attrs(self):
        super().set_attrs()
        self.attrs['filter_size'] = [3, 3]


class TestCase2_1(TestBase):
    def set_attrs(self):
        super().set_attrs()
        self.attrs['filter_size'] = [3, 2]


class TestCase3(TestBase):
    def set_attrs(self):
        super().set_attrs()
        self.attrs['stride'] = [2, 3]


class TestCase4(TestBase):
    def set_attrs(self):
        super().set_attrs()
        self.attrs['dilation'] = [2, 2]


class TestCase5(TestBase):
    def set_attrs(self):
        super().set_attrs()
        self.attrs['groups'] = 3


class TestCase6(TestBase):
    def set_attrs(self):
        super().set_attrs()
        self.attrs['padding'] = 2


class TestCase7(TestBase):
    def set_attrs(self):
        super().set_attrs()
        self.attrs['padding'] = [2, 3]


class TestCase8(TestBase):
    def set_attrs(self):
        super().set_attrs()
        self.attrs['padding'] = [1, 2, 2, 3]


if __name__ == "__main__":
    unittest.main()
