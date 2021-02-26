"""
@Description: Cross in Deep & Cross Network (DCN)
@version: https://arxiv.org/abs/1708.05123
@License: MIT
@Author: Wang Yao
@Date: 2020-08-06 18:44:25
@LastEditors: Wang Yao
@LastEditTime: 2021-02-26 15:09:18
"""
from typing import Optional, Union, Text

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class Cross(tf.keras.layers.Layer):
    """ Cross net in Deep & Cross Network (DCN) """

    def __init__(self, 
                 projection_dim: Optional[int] = None, 
                 diag_scale: Optional[float] = 0.0, 
                 use_bias: bool = True, 
                 kernel_init: Union[Text, tf.keras.initializers.Initializer] = "truncated_normal",
                 kernel_regu: Union[Text, None, tf.keras.regularizers.Regularizer] = None,
                 bias_init: Union[Text, tf.keras.initializers.Initializer] = "zeros",
                 bias_regu: Union[Text, None, tf.keras.regularizers.Regularizer] = None,
                 **kwargs):

        super(Cross, self).__init__(**kwargs)

        self._projection_dim = projection_dim
        self._diag_scale = diag_scale
        self._use_bias = use_bias
        self._kernel_init = tf.keras.initializers.get(kernel_init)
        self._kernel_regu = tf.keras.regularizers.get(kernel_regu)
        self._bias_init = tf.keras.initializers.get(bias_init)
        self._bias_regu = tf.keras.regularizers.get(bias_regu)

        assert self._diag_scale >= 0, \
            ValueError("diag scale must be non-negative, got {}".format(self._diag_scale))

        
    def build(self, input_shape):
        last_dim = input_shape[-1]

        if self._projection_dim is None:
            self._dense = tf.keras.layers.Dense(
                last_dim,
                kernel_initializer=self._kernel_init,
                kernel_regularizer=self._kernel_regu,
                bias_initializer=self._bias_init,
                bias_regularizer=self._bias_regu,
                use_bias=self._use_bias
            )
        else:
            if self._projection_dim < 0 or self._projection_dim > last_dim / 2:
                raise ValueError(
                    "`projection_dim` should be smaller than last_dim / 2 to improve "
                    "the model efficiency, and should be positive. Got "
                    "`projection_dim` {}, and last dimension of input {}".format(
                        self._projection_dim, last_dim))
            self._dense_u = tf.keras.layers.Dense(
                self._projection_dim,
                kernel_initializer=self._kernel_init,
                kernel_regularizer=self._kernel_regu,
                use_bias=False,
            )
            self._dense_v = tf.keras.layers.Dense(
                last_dim,
                kernel_initializer=self._kernel_init,
                bias_initializer=self._bias_init,
                kernel_regularizer=self._kernel_regu,
                bias_regularizer=self._bias_regu,
                use_bias=self._use_bias,
            )
        super(Cross, self).build(input_shape)

    def call(self, x0: tf.Tensor, x: Optional[tf.Tensor] = None):

        if x is None:
            x = x0
        
        if x0.shape[-1] != x.shape[-1]:
            raise ValueError("`x0` and `x` dim mismatch. " 
                             "Got `x0` dim = {} and `x` dim = {}".format(
                                x0.shape[-1], x.shape[-1]))
        
        if self._projection_dim is None:
            prod_output = self._dense(x)
        else:
            prod_output = self._dense_v(self._dense_u(x))

        if self._diag_scale:
            prod_output = prod_output + self._diag_scale * x

        return x0 * prod_output + x

    def get_config(self):
        config = {
            "projection_dim":
                self._projection_dim,
            "diag_scale":
                self._diag_scale,
            "use_bias":
                self._use_bias,
            "kernel_init":
                tf.keras.initializers.serialize(self._kernel_init),
            "kernel_regu":
                tf.keras.regularizers.serialize(self._kernel_regu),
            "bias_init":
                tf.keras.initializers.serialize(self._bias_init),
            "bias_regu":
                tf.keras.regularizers.serialize(self._bias_regu),
        }
        base_config = super(Cross, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


