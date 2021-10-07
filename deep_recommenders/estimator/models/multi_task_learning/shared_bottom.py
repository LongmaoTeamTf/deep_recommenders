#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

if tf.__version__ >= "2.0.0":
    import tensorflow.compat.v1 as tf

from deep_recommenders.estimator.models.feature_interaction import dnn
from deep_recommenders.estimator.models.multi_task_learning import multi_task


class SharedBottom(object):

    def __init__(self,
                 num_tasks,
                 task_units,
                 task_target_activations,
                 bottom_units,
                 bottom_activation=tf.nn.relu,
                 bottom_batch_norm=False,
                 bottom_dropout=None,
                 task_activation=tf.nn.relu,
                 task_batch_norm=False,
                 task_dropout=None):
        self._num_tasks = num_tasks
        self._task_units = task_units
        self._task_activations = task_target_activations
        self._bottom_units = bottom_units
        self._bottom_activation = bottom_activation
        self._bottom_batch_norm = bottom_batch_norm
        self._bottom_dropout = bottom_dropout
        self._task_activation = task_activation
        self._task_batch_norm = task_batch_norm
        self._task_dropout = task_dropout

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, features):

        bottom_outputs = dnn(features,
                             self._bottom_units,
                             activation=self._bottom_activation,
                             batch_normalization=self._bottom_batch_norm,
                             dropout=self._bottom_dropout)

        outputs = multi_task(bottom_outputs,
                             self._num_tasks,
                             self._task_activations,
                             self._task_units,
                             dnn_activation=self._task_activation,
                             dnn_batch_normalization=self._task_batch_norm,
                             dnn_dropout=self._task_dropout)
        return outputs


