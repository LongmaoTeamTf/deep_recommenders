#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf


def task_network(inputs,
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


def multi_task(inputs,
               num_tasks,
               task_hidden_units,
               task_output_activations,
               **kwargs):

    outputs = []

    for i in range(num_tasks):

        task_inputs = inputs[i] if isinstance(inputs, list) else inputs

        output = task_network(task_inputs,
                              task_hidden_units,
                              output_activation=task_output_activations[i],
                              **kwargs)
        outputs.append(output)

    return outputs
