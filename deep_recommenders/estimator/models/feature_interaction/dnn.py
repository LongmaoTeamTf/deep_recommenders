#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

if tf.__version__ >= "2.0.0":
    import tensorflow.compat.v1 as tf


def dnn(inputs,
        hidden_units,
        activation=tf.nn.relu,
        batch_normalization=False,
        dropout=None,
        **kwargs):

    x = inputs
    for units in hidden_units[:-1]:
        x = tf.layers.dense(x,
                            units,
                            activation,
                            **kwargs)

        if batch_normalization is True:
            x = tf.nn.batch_normalization(x)

        if dropout is not None:
            x = tf.nn.dropout(x, rate=dropout)

    outputs = tf.layers.dense(x, hidden_units[-1], **kwargs)

    return outputs
