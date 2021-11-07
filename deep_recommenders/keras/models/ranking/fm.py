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


class FactorizationMachine(tf.keras.Model):

    def __init__(self, indicator_columns, embedding_columns, **kwargs):
        super(FactorizationMachine, self).__init__(**kwargs)
        self._indicator_columns = indicator_columns
        self._embedding_columns = embedding_columns

        self._sparse_features_layer = tf.keras.layers.DenseFeatures(self._indicator_columns)
        self._embedding_features_layer = {
            c.categorical_column.key: tf.keras.layers.DenseFeatures(c)
            for c in self._embedding_columns
        }
        self._kernel = FM()

    def call(self, inputs, training=None, mask=None):
        sparse_features = self._sparse_features_layer(inputs)
        embeddings = []
        for column_name, column_input in inputs.items():
            dense_features = self._embedding_features_layer.get(column_name)
            if dense_features is not None:
                embedding = dense_features({column_name: column_input})
                embeddings.append(embedding)
        stack_embeddings = tf.stack(embeddings, axis=1)
        outputs = self._kernel(sparse_features, stack_embeddings)
        return tf.nn.sigmoid(outputs)

    def get_config(self):
        config = {
            "indicator_columns": self._indicator_columns,
            "embedding_columns": self._embedding_columns
        }
        base_config = super(FactorizationMachine, self).get_config()
        return {**base_config, **config}
