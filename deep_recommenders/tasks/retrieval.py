"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2021-02-24 11:23:29
@LastEditors: Wang Yao
@LastEditTime: 2021-02-24 19:36:34
"""
from typing import Optional

import tensorflow as tf

from deep_recommenders import layers
from deep_recommenders import metrics
from deep_recommenders.tasks import base


class Retrieval(tf.keras.layers.Layer, base.Task):
    """检索任务"""

    def __init__(self,
                 loss: Optional[tf.keras.losses.Loss] = None,
                 metrics: Optional[metrics.FactorizedTopK] = None,
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
    def factorized_metrics(self) -> Optional[metrics.FactorizedTopK]:
        """The metrics object used to compute retrieval metrics."""

        return self._factorized_metrics

    @factorized_metrics.setter
    def factorized_metrics(self, value: Optional[metrics.FactorizedTopK]) -> None:
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
            scores = layers.loss.SamplingProbablityCorrection()(
                scores, candidate_sampling_probability)
        
        if candidate_ids is not None:
            scores = layers.loss.RemoveAccidentalNegative()(
                scores, labels, candidate_ids)

        if self._num_hard_negatives is not None:
            scores, labels = layers.loss.HardNegativeMining(
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
