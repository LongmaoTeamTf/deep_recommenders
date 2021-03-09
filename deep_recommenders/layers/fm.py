"""
@Description: Factorization Machines
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-12-03 18:01:05
@LastEditors: Wang Yao
@LastEditTime: 2021-02-26 15:17:34
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
        self._kernel_init = kernel_init
        self._kernel_regu = kernel_regu

        if (self._factors is not None) and (self._factors <= 0):
            raise ValueError("`factors` should be bigger than 0. "
                    "Got `factors` = {}".format(self._factors))

    def build(self, input_shape):
        last_dim = input_shape[-1]

        if self._factors is not None:
            self._kernel = self.add_weight(
                shape=(last_dim, self._factors),
                initializer=tf.keras.initializers.get(self._kernel_init),
                regularizer=tf.keras.regularizers.get(self._kernel_regu),
                trainable=True,
                name="kernel"
            )
        self.built = True

    def call(self, x: tf.Tensor):
        
        if tf.keras.backend.ndim(x) != 3:
            raise ValueError("`x` dim should be 3. Got `x` dim = {}".format(
                    tf.keras.backend.ndim(x)))

        if self._factors is None:
            embed_x = x
            square_embed_x = tf.pow(x, 2)
        else:
            embed_x = tf.matmul(x, self._kernel)
            square_embed_x = tf.matmul(tf.pow(x, 2), tf.pow(self._kernel, 2))

        outputs = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.pow(tf.reduce_sum(embed_x, axis=1, keepdims=True), 2),
                tf.reduce_sum(square_embed_x, axis=1, keepdims=True)
            ), axis=-1, keepdims=False
        )
        return outputs

    def get_config(self):
        if self._factors is None:
            config = {
                "factors":
                    self._factors,
            }
        else:
            config = {
                "factors":
                    self._factors,
                "kernel_initializer":
                    tf.keras.initializers.serialize(self._kernel_init),
                "kernel_regularizer":
                    tf.keras.regularizers.serialize(self._kernel_regu),
            }
        base_config = super(FM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

