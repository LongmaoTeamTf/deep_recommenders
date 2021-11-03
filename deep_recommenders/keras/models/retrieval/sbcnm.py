#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf


MAX_FLOAT = np.finfo(np.float32).max / 100.0
MIN_FLOAT = np.finfo(np.float32).min / 100.0

from deep_recommenders.keras.models.retrieval import FactorizedTopK


def _gather_elements_along_row(data: tf.Tensor,
                               column_indices: tf.Tensor) -> tf.Tensor:
    """与factorized_top_k中_take_long_axis相同"""
    with tf.control_dependencies(
            [tf.assert_equal(tf.shape(data)[0], tf.shape(column_indices)[0])]):
        num_row = tf.shape(data)[0]
        num_column = tf.shape(data)[1]
        num_gathered = tf.shape(column_indices)[1]
        row_indices = tf.tile(
            tf.expand_dims(tf.range(num_row), -1),
            [1, num_gathered])
        flat_data = tf.reshape(data, [-1])
        flat_indices = tf.reshape(
            row_indices * num_column + column_indices, [-1])
        return tf.reshape(
            tf.gather(flat_data, flat_indices), [num_row, num_gathered])


class HardNegativeMining(tf.keras.layers.Layer):
    """Hard Negative"""

    def __init__(self, num_hard_negatives: int, **kwargs):
        super(HardNegativeMining, self).__init__(**kwargs)

        self._num_hard_negatives = num_hard_negatives

    def call(self, logits: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        num_sampled = tf.minimum(self._num_hard_negatives + 1, tf.shape(logits)[1])

        _, indices = tf.nn.top_k(logits + labels * MAX_FLOAT, k=num_sampled, sorted=False)

        logits = _gather_elements_along_row(logits, indices)
        labels = _gather_elements_along_row(labels, indices)

        return logits, labels


class RemoveAccidentalNegative(tf.keras.layers.Layer):

    def call(self,
             logits: tf.Tensor,
             labels: tf.Tensor,
             identifiers: tf.Tensor) -> tf.Tensor:
        """Zeros logits of accidental negatives
        Args:
            logits: [batch_size, num_candidates] 2D tensor
            labels: [batch_size, num_candidates] one-hot 2D tensor
            identifiers: [num_candidates] candidates identifiers tensor
        Returns:
            logits: Modified logits.
        """
        identifiers = tf.expand_dims(identifiers, 1)
        positive_indices = tf.math.argmax(labels, axis=1)
        positive_identifier = tf.gather(identifiers, positive_indices)

        duplicate = tf.equal(positive_identifier, tf.transpose(identifiers))
        duplicate = tf.cast(duplicate, labels.dtype)

        duplicate = duplicate - labels

        return logits + duplicate * MIN_FLOAT


class SamplingProbabilityCorrection(tf.keras.layers.Layer):
    """Sampling probability correction."""

    def call(self,
             logits: tf.Tensor,
             candidate_sampling_probability: tf.Tensor) -> tf.Tensor:
        """Corrects the input logits to account for candidate sampling probability."""

        return logits - tf.math.log(candidate_sampling_probability)


class Retrieval(tf.keras.layers.Layer):
    """检索任务"""

    def __init__(self,
                 loss: Optional[tf.keras.losses.Loss] = None,
                 metrics: Optional[FactorizedTopK] = None,
                 temperature: Optional[float] = None,
                 num_hard_negatives: Optional[int] = None,
                 **kwargs):
        super(Retrieval, self).__init__(**kwargs)

        self._loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM
        ) if loss is None else loss

        self._factorized_metrics = metrics
        self._temperature = temperature
        self._num_hard_negatives = num_hard_negatives

    @property
    def factorized_metrics(self) -> Optional[FactorizedTopK]:
        """The metrics object used to compute retrieval metrics."""

        return self._factorized_metrics

    @factorized_metrics.setter
    def factorized_metrics(self, value: Optional[FactorizedTopK]) -> None:
        """Sets factorized metrics."""

        self._factorized_metrics = value

    def call(self,
             query_embeddings: tf.Tensor,
             candidate_embeddings: tf.Tensor,
             sample_weight: Optional[tf.Tensor] = None,
             candidate_sampling_probability: Optional[tf.Tensor] = None,
             candidate_ids: Optional[tf.Tensor] = None,
             compute_metrics: bool = True) -> tf.Tensor:
        """Compute loss and metrics"""

        scores = tf.matmul(query_embeddings, candidate_embeddings, transpose_b=True)

        num_queries = tf.shape(scores)[0]
        num_candidates = tf.shape(scores)[1]

        labels = tf.eye(num_queries, num_candidates)

        if candidate_sampling_probability is not None:
            scores = deep_recommenders.keras.layers.embedding.loss.SamplingProbablityCorrection()(
                scores, candidate_sampling_probability)

        if candidate_ids is not None:
            scores = deep_recommenders.keras.layers.embedding.loss.RemoveAccidentalNegative()(
                scores, labels, candidate_ids)

        if self._num_hard_negatives is not None:
            scores, labels = deep_recommenders.keras.layers.embedding.loss.HardNegativeMining(
                self._num_hard_negatives)(scores, labels)

        if self._temperature is not None:
            scores = scores / self._temperature

        loss = self._loss(y_true=labels, y_pred=scores, sample_weight=sample_weight)

        if compute_metrics is False:
            return loss

        if not self._factorized_metrics:
            return loss

        update_op = self._factorized_metrics.update_state(
            query_embeddings, candidate_embeddings)

        with tf.control_dependencies([update_op]):
            return tf.identity(loss)
