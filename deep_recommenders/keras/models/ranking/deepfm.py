#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from deep_recommenders.keras.models.ranking import FM


class DeepFM(tf.keras.Model):

    def __init__(self,
                 indicator_columns,
                 embedding_columns,
                 dnn_units_size,
                 dnn_activation="relu",
                 **kwargs):

        super(DeepFM, self).__init__(**kwargs)
        self._indicator_columns = indicator_columns
        self._embedding_columns = embedding_columns
        self._dnn_units_size = dnn_units_size
        self._dnn_activation = dnn_activation

        self._sparse_features_layer = tf.keras.layers.DenseFeatures(self._indicator_columns)
        self._embedding_features_layer = {
            c.categorical_column.key: tf.keras.layers.DenseFeatures(c)
            for c in self._embedding_columns
        }
        self._fm = FM()
        self._dnn = tf.keras.Sequential([
            tf.keras.layers.Dense(units, activation=self._dnn_activation)
            for units in self._dnn_units_size
            ] + [tf.keras.layers.Dense(1)]
        )

    def call(self, inputs, **kwargs):
        sparse_features = self._sparse_features_layer(inputs)
        embeddings = []
        for column_name, column_input in inputs.items():
            dense_features = self._embedding_features_layer.get(column_name)
            if dense_features is not None:
                embedding = dense_features({column_name: column_input})
                embeddings.append(embedding)
        stack_embeddings = tf.stack(embeddings, axis=1)
        concat_embeddings = tf.concat(embeddings, axis=1)
        outputs = self._fm(sparse_features, stack_embeddings) + self._dnn(concat_embeddings)
        return tf.keras.activations.sigmoid(outputs)

    def get_config(self):
        config = {
            "dnn_units_size": self._dnn_units_size,
            "dnn_activation": self._dnn_activation
        }
        base_config = super(DeepFM, self).get_config()
        return {**base_config, **config}
