"""
@Description: Factorization Machines
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-12-03 18:01:05
@LastEditors: Wang Yao
@LastEditTime: 2021-03-10 19:14:47
"""
from typing import Optional, Union, Text

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class FM(tf.keras.layers.Layer):
    """ Factorization Machine """

    def __init__(self, 
                 factors: Union[int, None] = 10,
                 kernel_init: Union[Text, tf.keras.initializers.Initializer] = "truncated_normal",
                 kernel_regu: Union[Text, None, tf.keras.regularizers.Regularizer] = None,
                 **kwargs):

        super(FM, self).__init__(**kwargs)
        
        self._factors = factors
        self._kernel_init = tf.keras.initializers.get(kernel_init)
        self._kernel_regu = tf.keras.regularizers.get(kernel_regu)

        if (self._factors is not None) and (self._factors <= 0):
            raise ValueError("`factors` should be bigger than 0. "
                    "Got `factors` = {}".format(self._factors))

    def build(self, input_shape):
        last_dim = input_shape[-1]

        if self._factors is not None:
            self._kernel = self.add_weight(
                shape=(last_dim, self._factors),
                initializer=self._kernel_init,
                regularizer=self._kernel_regu,
                trainable=True,
                name="kernel"
            )
        self.built = True

    def call(self, x: tf.Tensor):

        if self._factors is None:

            if tf.keras.backend.ndim(x) != 3:
                raise ValueError("When `factors` is None, `x` dim should be 3. "
                                 "Got `x` dim = {}".format(tf.keras.backend.ndim(x)))

            x_sum = tf.reduce_sum(x, axis=1)
            x_square_sum = tf.reduce_sum(tf.pow(x, 2), axis=1)
        else:
            if tf.keras.backend.ndim(x) != 2:
                raise ValueError("When `factors` is not None, `x` dim should be 2. "
                                 "Got `x` dim = {}".format(tf.keras.backend.ndim(x)))

            x_sum = tf.linalg.matmul(x, self._kernel, a_is_sparse=True)
            x_square_sum = tf.linalg.matmul(
                tf.pow(x, 2), tf.pow(self._kernel, 2), a_is_sparse=True)

        outputs = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.pow(x_sum, 2),
                x_square_sum
            ), axis=1, keepdims=True)

        return outputs

    def get_config(self):
        config = {
            "factors":
                self._factors,
            "kernel_init":
                tf.keras.initializers.serialize(self._kernel_init),
            "kernel_regu":
                tf.keras.regularizers.serialize(self._kernel_regu),
        }
        base_config = super(FM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

