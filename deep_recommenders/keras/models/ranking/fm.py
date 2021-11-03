#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf


from deep_recommenders.keras.models.feature_interaction import FM


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
