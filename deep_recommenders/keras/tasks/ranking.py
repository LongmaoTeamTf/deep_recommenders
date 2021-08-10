#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Optional, List

import tensorflow as tf

from deep_recommenders.keras.tasks import base


class Ranking(tf.keras.layers.Layer, base.Task):
    """排序任务"""

    def __init__(self,
                 loss: Optional[tf.keras.losses.Loss] = None,
                 metrics: Optional[List[tf.keras.metrics.Metric]] = None,
                 prediction_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
                 label_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
                 loss_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
                 **kwargs):

        super(Ranking, self).__init__(**kwargs)

        self._loss = (tf.keras.losses.BinaryCrossentropy() if loss is None else loss)
        self._metrics = metrics or []
        self._prediction_metrics = prediction_metrics or []
        self._label_metrics = label_metrics or []
        self._loss_metrics = loss_metrics or []


    def call(self,
             labels: tf.Tensor,
             predictions: tf.Tensor,
             sample_weight: Optional[tf.Tensor] = None,
             compute_metrics: bool = True,
             training: bool = False):
        """Compute loss and metircs."""
        
        loss = self._loss(labels, predictions, sample_weight=sample_weight)

        if compute_metrics is False:
            return loss
        
        update_ops = []

        for metric in self._metrics:
            update_ops.append(metric.update_state(
                y_true=labels, y_pred=predictions, sample_weight=sample_weight))

        for metric in self._prediction_metrics:
            update_ops.append(
                metric.update_state(predictions, sample_weight=sample_weight))

        for metric in self._label_metrics:
            update_ops.append(
                metric.update_state(labels, sample_weight=sample_weight))

        for metric in self._loss_metrics:
            update_ops.append(
                metric.update_state(loss, sample_weight=sample_weight))

        # Custom metrics may not return update ops, unlike built-in Keras metrics.
        update_ops = [x for x in update_ops if x is not None]

        with tf.control_dependencies(update_ops):
            return tf.identity(loss)
