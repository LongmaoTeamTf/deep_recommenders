#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

if tf.__version__ >= "2.0.0":
    import tensorflow.compat.v1 as tf


def fm(x):
    """
    Second order interaction in Factorization Machine
    :param x:
        type: tf.Tensor
        shape: (batch_size, num_features, embedding_dim)
    :return: tf.Tensor
    """

    if x.shape.rank != 3:
        raise ValueError("The rank of `x` should be 3. Got rank = {}.".format(x.shape.rank))

    sum_square = tf.square(tf.reduce_sum(x, axis=1))
    square_sum = tf.reduce_sum(tf.square(x), axis=1)

    return 0.5 * tf.reduce_sum(
        tf.subtract(sum_square, square_sum), axis=1, keepdims=True)


class FM(object):
    """
    Factorization Machine
    """

    def __init__(self, indicator_columns, embedding_columns):
        self._indicator_columns = indicator_columns
        self._embedding_columns = embedding_columns

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, features):

        with tf.variable_scope("linear"):
            linear_outputs = tf.feature_column.linear_model(features, self._indicator_columns)

        with tf.variable_scope("factorized"):
            self.embeddings = []
            for embedding_column in self._embedding_columns:
                feature_name = embedding_column.name.replace("_embedding", "")
                feature = {feature_name: features.get(feature_name)}
                embedding = tf.feature_column.input_layer(feature, embedding_column)
                self.embeddings.append(embedding)
            stack_embeddings = tf.stack(self.embeddings, axis=1)
            factorized_outputs = fm(stack_embeddings)

        return linear_outputs + factorized_outputs
