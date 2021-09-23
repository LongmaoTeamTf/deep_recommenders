#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

if tf.__version__ >= "2.0.0":
    import tensorflow.compat.v1 as tf


def dnn(inputs,
        hidden_units,
        hidden_activation=tf.nn.relu,
        output_activation=tf.nn.sigmoid,
        hidden_dropout=None,
        initializer=None):

    x = inputs
    for units in hidden_units:
        x = tf.layers.dense(x,
                            units,
                            activation=hidden_activation,
                            kernel_initializer=initializer)

        if hidden_dropout is not None:
            x = tf.layers.dropout(x, rate=hidden_dropout)

    outputs = tf.layers.dense(x, 1, kernel_initializer=initializer)

    if output_activation is not None:
        outputs = output_activation(outputs)
    return outputs
