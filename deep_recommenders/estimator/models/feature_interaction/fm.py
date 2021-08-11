#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

if tf.__version__ >= "2.3.0":
    import tensorflow.compat.v1 as tf


def fm(x: tf.Tensor, num_factors: int = None):
    """
    Factorization Machine
    :param x: tf.Tensor
    :param num_factors: the number of latent factors
    :return: tf.Tensor
    """

    x_rank = x.get_shape().rank

    if num_factors is None:
        assert x_rank == 3, ValueError("When `num_factors` is None , the rank of `x` should be 3."
                                       " Got {}.".format(x_rank))
        x_sum = tf.reduce_sum(x, axis=1)
        x_square_sum = tf.reduce_sum(tf.pow(x, 2), axis=1)
    else:
        assert x_rank == 2, ValueError("When `num_factors` is not None, the rank of `x` should be 2."
                                       " Got {}.".format(x_rank))
        initial = tf.random.truncated_normal(shape=(x.get_shape()[-1], num_factors))
        factors = tf.Variable(initial, name='factors')

        x_sum = tf.linalg.matmul(x, factors, a_is_sparse=True)
        x_square_sum = tf.linalg.matmul(tf.pow(x, 2), tf.pow(factors, 2), a_is_sparse=True)

    return 0.5 * tf.reduce_sum(
        tf.subtract(tf.pow(x_sum, 2), x_square_sum), axis=1, keepdims=True)
