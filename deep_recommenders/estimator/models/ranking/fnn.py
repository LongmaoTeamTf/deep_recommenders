#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from deep_recommenders.estimator.models.feature_interaction import dnn


class FNN(object):

    def __init__(self,
                 indicator_columns,
                 embedding_columns,
                 warmup_from_fm,
                 dnn_units,
                 dnn_activation=tf.nn.relu,
                 dnn_batch_normalization=False,
                 dnn_dropout=None,
                 **dnn_kwargs):
        self._indicator_columns = indicator_columns
        self._embedding_columns = embedding_columns
        self._warmup_from_fm = warmup_from_fm
        self._dnn_hidden_units = dnn_units
        self._dnn_activation = dnn_activation
        self._dnn_batch_norm = dnn_batch_normalization
        self._dnn_dropout = dnn_dropout
        self._dnn_kwargs = dnn_kwargs

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def warm_up(self):
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.load(sess, ["serve"], self._warmup_from_fm)
            linear_variables = tf.get_collection(
                tf.GraphKeys.MODEL_VARIABLES, "linear")
            linear_variables = {
                var.name.split("/")[2].replace("_indicator", "")
                if "bias" not in var.name else "bias": sess.run(var)
                for var in linear_variables
            }
            factorized_variables = tf.get_collection(
                tf.GraphKeys.MODEL_VARIABLES, "factorized")
            factorized_variables = {
                var.name.split("/")[2].replace("_embedding", ""): sess.run(var)
                for var in factorized_variables
            }
            return linear_variables, factorized_variables

    def call(self, features):
        linear_variables, factorized_variables = self.warm_up()

        weights = []
        for indicator_column in self._indicator_columns:
            feature_name = indicator_column.categorical_column.key
            feature = {feature_name: features.get(feature_name)}
            sparse = tf.feature_column.input_layer(feature, indicator_column)
            weights_initializer = tf.constant_initializer(linear_variables.get(feature_name))
            weight = tf.layers.dense(sparse,
                                     units=1,
                                     use_bias=False,
                                     kernel_initializer=weights_initializer)
            weights.append(weight)
        concat_weights = tf.concat(weights, axis=1)

        embeddings = []
        for embedding_column in self._embedding_columns:
            feature_name = embedding_column.categorical_column.key
            feature = {feature_name: features.get(feature_name)}
            embedding_column = tf.feature_column.embedding_column(
                embedding_column.categorical_column,
                embedding_column.dimension,
                initializer=tf.constant_initializer(factorized_variables.get(feature_name))
            )
            embedding = tf.feature_column.input_layer(feature, embedding_column)
            embeddings.append(embedding)
        concat_embeddings = tf.concat(embeddings, axis=1)

        bias = tf.expand_dims(linear_variables.get("bias"), axis=0)
        bias = tf.tile(bias, [tf.shape(concat_weights)[0], 1])

        dnn_inputs = tf.concat([bias, concat_weights, concat_embeddings], axis=1)

        outputs = dnn(dnn_inputs,
                      self._dnn_hidden_units + [1],
                      activation=self._dnn_activation,
                      batch_normalization=self._dnn_batch_norm,
                      dropout=self._dnn_dropout,
                      **self._dnn_kwargs)
        return tf.nn.sigmoid(outputs)
