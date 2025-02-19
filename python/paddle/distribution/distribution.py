#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define the distribution functions
# __all__ = ['Categorical',
#            'MultivariateNormalDiag',
#            'Normal',
#            'sampling_id',
#            'Uniform']

from __future__ import print_function

import math
import warnings

import numpy as np
from paddle import _C_ops

from ..fluid import core
from ..fluid.data_feeder import (check_dtype, check_type,
                                 check_variable_and_dtype, convert_dtype)
from ..fluid.framework import in_dygraph_mode
from ..fluid.layers import (control_flow, elementwise_add, elementwise_div,
                            elementwise_mul, elementwise_sub, nn, ops, tensor)
from ..tensor import arange, concat, gather_nd, multinomial


class Distribution(object):
    """
    The abstract base class for probability distributions. Functions are 
    implemented in specific distributions.

    Args:
        batch_shape(Sequence[int], optional):  independent, not identically 
            distributed draws, aka a "collection" or "bunch" of distributions.
        event_shape(Sequence[int], optional): the shape of a single 
            draw from the distribution; it may be dependent across dimensions. 
            For scalar distributions, the event shape is []. For n-dimension 
            multivariate distribution, the event shape is [n].
    """

    def __init__(self, batch_shape=(), event_shape=()):

        self._batch_shape = batch_shape if isinstance(
            batch_shape, tuple) else tuple(batch_shape)
        self._event_shape = event_shape if isinstance(
            event_shape, tuple) else tuple(event_shape)

        super(Distribution, self).__init__()

    @property
    def batch_shape(self):
        """Returns batch shape of distribution

        Returns:
            Sequence[int]: batch shape
        """
        return self._batch_shape

    @property
    def event_shape(self):
        """Returns event shape of distribution

        Returns:
            Sequence[int]: event shape
        """
        return self._event_shape

    def sample(self, shape=()):
        """Sampling from the distribution."""
        raise NotImplementedError

    def entropy(self):
        """The entropy of the distribution."""
        raise NotImplementedError

    def kl_divergence(self, other):
        """The KL-divergence between self distributions and other."""
        raise NotImplementedError

    def prob(self, value):
        """Probability density/mass function evaluated at value.

        Args:
            value (Tensor): value which will be evaluated
        """
        raise NotImplementedError

    def log_prob(self, value):
        """Log probability density/mass function."""
        raise NotImplementedError

    def probs(self, value):
        """Probability density/mass function."""
        raise NotImplementedError

    def _extend_shape(self, sample_shape):
        """compute shape of the sample 

        Args:
            sample_shape (Tensor): sample shape

        Returns:
            Tensor: generated sample data shape
        """
        return sample_shape + self._batch_shape + self._event_shape

    def _validate_args(self, *args):
        """
        Argument validation for distribution args
        Args:
            value (float, list, numpy.ndarray, Tensor)
        Raises
            ValueError: if one argument is Tensor, all arguments should be Tensor
        """
        is_variable = False
        is_number = False
        for arg in args:
            if isinstance(arg, tensor.Variable):
                is_variable = True
            else:
                is_number = True

        if is_variable and is_number:
            raise ValueError(
                'if one argument is Tensor, all arguments should be Tensor')

        return is_variable

    def _to_tensor(self, *args):
        """
        Argument convert args to Tensor

        Args:
            value (float, list, numpy.ndarray, Tensor)
        Returns:
            Tensor of args.
        """
        numpy_args = []
        variable_args = []
        tmp = 0.

        for arg in args:
            if isinstance(arg, float):
                arg = [arg]
            if not isinstance(arg, (list, tuple, np.ndarray, tensor.Variable)):
                raise TypeError(
                    "Type of input args must be float, list, numpy.ndarray or Tensor, but received type {}".
                    format(type(arg)))

            arg_np = np.array(arg)
            arg_dtype = arg_np.dtype
            if str(arg_dtype) != 'float32':
                if str(arg_dtype) != 'float64':
                    # "assign" op doesn't support float64. if dtype is float64, float32 variable will be generated
                    #  and converted to float64 later using "cast".
                    warnings.warn(
                        "data type of argument only support float32 and float64, your argument will be convert to float32."
                    )
                arg_np = arg_np.astype('float32')
            # tmp is used to support broadcast, it summarizes shapes of all the args and get the mixed shape.
            tmp = tmp + arg_np
            numpy_args.append(arg_np)

        dtype = tmp.dtype
        for arg in numpy_args:
            arg_broadcasted, _ = np.broadcast_arrays(arg, tmp)
            arg_variable = tensor.create_tensor(dtype=dtype)
            tensor.assign(arg_broadcasted, arg_variable)
            variable_args.append(arg_variable)

        return tuple(variable_args)

    def _check_values_dtype_in_probs(self, param, value):
        """
        Log_prob and probs methods have input ``value``, if value's dtype is different from param,
        convert value's dtype to be consistent with param's dtype.

        Args:
            param (Tensor): low and high in Uniform class, loc and scale in Normal class.
            value (Tensor): The input tensor.

        Returns:
            value (Tensor): Change value's dtype if value's dtype is different from param.
        """
        if in_dygraph_mode():
            if value.dtype != param.dtype and convert_dtype(
                    value.dtype) in ['float32', 'float64']:
                warnings.warn(
                    "dtype of input 'value' needs to be the same as parameters of distribution class. dtype of 'value' will be converted."
                )
                return _C_ops.cast(value, 'in_dtype', value.dtype, 'out_dtype',
                                   param.dtype)
            return value

        check_variable_and_dtype(value, 'value', ['float32', 'float64'],
                                 'log_prob')
        if value.dtype != param.dtype:
            warnings.warn(
                "dtype of input 'value' needs to be the same as parameters of distribution class. dtype of 'value' will be converted."
            )
            return tensor.cast(value, dtype=param.dtype)
        return value
