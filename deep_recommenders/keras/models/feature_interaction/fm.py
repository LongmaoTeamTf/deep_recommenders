#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class FM(tf.keras.layers.Layer):
    """ Factorization Machine """

    def __init__(self, **kwargs):
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):

        self._linear = tf.keras.layers.Dense(
            units=1,
            kernel_initializer="zeros",
            name="linear"
        )
        self.built = True

    def call(self, sparse_inputs, embedding_inputs=None, **kwargs):

        if embedding_inputs is None:
            return self._linear(sparse_inputs)

        x_sum = tf.reduce_sum(embedding_inputs, axis=1)
        x_square_sum = tf.reduce_sum(tf.pow(embedding_inputs, 2), axis=1)

        interaction = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.pow(x_sum, 2),
                x_square_sum
            ), axis=1, keepdims=True)

        return self._linear(sparse_inputs) + interaction




