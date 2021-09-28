#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from deep_recommenders.estimator.models.feature_interaction import dnn


class WDL(object):

    def __init__(self,
                 indicator_columns,
                 embedding_columns,
                 dnn_hidden_units
                 ):
        self._indicator_columns = indicator_columns
        self._embedding_columns = embedding_columns
        self._dnn_hidden_units = dnn_hidden_units

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, features):
        with tf.variable_scope("wide"):
            linear_outputs = tf.feature_column.linear_model(features,
                                                            self._indicator_columns)
        with tf.variable_scope("deep"):
            embeddings = []
            for embedding_column in self._embedding_columns:
                feature_name = embedding_column.name.replace("_embedding", "")
                feature = {feature_name: features.get(feature_name)}
                embedding = tf.feature_column.input_layer(feature, embedding_column)
                embeddings.append(embedding)
            concat_embeddings = tf.concat(embeddings, axis=1)
            dnn_outputs = dnn(concat_embeddings,
                              self._dnn_hidden_units,
                              output_activation=None)
        return tf.nn.sigmoid(linear_outputs + dnn_outputs)
