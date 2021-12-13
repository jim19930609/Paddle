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

import paddle.fluid.core as core
import paddle.fluid.eager.eager_tensor_patch_methods as eager_tensor_patch_methods
import paddle
import numpy as np
from paddle.fluid import eager_guard
import unittest


class EagerScaleTestCase(unittest.TestCase):
    def test_scale_base(self):
        with eager_guard():
            paddle.set_device("cpu")
            arr = np.ones([4, 16, 16, 32]).astype('float32')
            tensor = paddle.to_tensor(arr, 'float32', core.CPUPlace())
            #print(tensor)
            tensor = core.eager.scale(tensor, 2.0, 0.9, True, False)
            for i in range(0, 100):
                tensor = core.eager.scale(tensor, 2.0, 0.9, True, False)
            #print(tensor)
            self.assertEqual(tensor.shape, [4, 16, 16, 32])
            self.assertEqual(tensor.stop_gradient, True)
            tensor.stop_gradient = False
            self.assertEqual(tensor.stop_gradient, False)
            tensor.stop_gradient = True
            self.assertEqual(tensor.stop_gradient, False)
            tensor.name = 'tensor_name_test'
            self.assertEqual(tensor.name, 'tensor_name_test')
            self.assertEqual(tensor.persistable, False)
            tensor.persistable = True
            self.assertEqual(tensor.persistable, True)
            tensor.persistable = False
            self.assertEqual(tensor.persistable, False)
            self.assertTrue(tensor.place.is_cpu_place())


class EagerMatmulTestCase(unittest.TestCase):
    def test_matmul_base(self):
        with eager_guard():
            paddle.set_device("cpu")

            arrX = np.ones([4, 16]).astype('float32')
            X = paddle.to_tensor(arrX, 'float32', core.CPUPlace())
            arrY = np.ones([16, 32]).astype('float32')
            Y = paddle.to_tensor(arrY, 'float32', core.CPUPlace())
            #print(X)
            #print(Y)

            Out = core.eager.matmul(X, Y, False, False, False)
            #print(Out)
            OutNumpy = Out.numpy()
            #print(OutNumpy)
            self.assertEqual(Out.shape, [4, 32])
            self.assertEqual(Out.stop_gradient, True)
            self.assertEqual(OutNumpy[0, 0], 16.0)


class EagerEWAddTestCase(unittest.TestCase):
    def test_elementwise_add_base(self):
        with eager_guard():
            paddle.set_device("cpu")

            arrX = np.ones([4, 16]).astype('float32')
            X = paddle.to_tensor(arrX, 'float32', core.CPUPlace())
            arrY = np.ones([4, 16]).astype('float32')
            Y = paddle.to_tensor(arrY, 'float32', core.CPUPlace())
            #print(X)
            #print(Y)

            Out = core.eager.elementwise_add(X, Y, -1, False)
            #print(Out)
            OutNumpy = Out.numpy()
            self.assertEqual(Out.shape, [4, 16])
            self.assertEqual(Out.stop_gradient, True)
            self.assertEqual(OutNumpy[0, 0], 2.0)


if __name__ == "__main__":
    unittest.main()
