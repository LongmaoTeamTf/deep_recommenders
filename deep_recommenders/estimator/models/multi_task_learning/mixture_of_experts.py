#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

if tf.__version__ >= "2.0.0":
    import tensorflow.compat.v1 as tf

from deep_recommenders.estimator.models.feature_interaction import dnn


class MMoE(object):

    def __init__(self,
                 feature_columns,
                 num_tasks,
                 num_experts,
                 expert_hidden_units,
                 task_hidden_units,
                 task_hidden_activation=tf.nn.relu,
                 task_batch_normalization=False,
                 task_dropout=None,
                 expert_hidden_activation=tf.nn.relu,
                 expert_batch_normalization=False,
                 expert_dropout=None):

        self._columns = feature_columns

        self._num_tasks = num_tasks
        self._num_experts = num_experts
        self._expert_hidden_units = expert_hidden_units
        self._task_hidden_units = task_hidden_units

        self._task_hidden_activation = task_hidden_activation
        self._task_batch_norm = task_batch_normalization
        self._task_dropout = task_dropout

        self._expert_hidden_activation = expert_hidden_activation
        self._expert_batch_norm = expert_batch_normalization
        self._expert_dropout = expert_dropout

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def gating_network(self, inputs):
        """
        Gating network: y = SoftMax(W * inputs)
        """
        x = tf.layers.dense(inputs,
                            units=self._num_experts,
                            use_bias=False)

        return tf.nn.softmax(x)

    def call(self, features):

        inputs = tf.feature_column.input_layer(features, self._columns)

        with tf.variable_scope("mixture_of_experts"):
            experts_outputs = []
            for _ in range(self._num_experts):
                x = dnn(inputs,
                        self._expert_hidden_units,
                        activation=self._expert_hidden_activation,
                        batch_normalization=self._expert_batch_norm,
                        dropout=self._expert_dropout)
                experts_outputs.append(x)
            moe_outputs = tf.stack(experts_outputs, axis=1)

        with tf.variable_scope("multi_gate"):
            mg_outputs = []
            for _ in range(self._num_experts):
                gate = self.gating_network(inputs)
                gate = tf.expand_dims(gate, axis=1)
                output = tf.linalg.matmul(gate, moe_outputs)
                mg_outputs.append(tf.squeeze(output, axis=1))

        outputs = []
        for idx in range(self._num_tasks):
            with tf.variable_scope("task{}".format(idx)):
                x = dnn(mg_outputs[idx],
                        self._task_hidden_units + [1],
                        activation=self._task_hidden_activation,
                        batch_normalization=self._task_batch_norm,
                        dropout=self._task_dropout)

                outputs.append(x)

        return outputs
