"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2021-02-23 10:41:28
@LastEditors: Wang Yao
@LastEditTime: 2021-02-26 15:27:55
"""
from typing import List, Optional, Sequence, Union, Text

import tensorflow as tf
from deep_recommenders import layers


class FactorizedTopK(tf.keras.layers.Layer):
    """ Metric for a retrieval model. """

    def __init__(self, 
                 candidates: Union[layers.factorized_top_k.TopK, tf.data.Dataset], 
                 metrics: Optional[Sequence[tf.keras.metrics.Metric]] = None,
                 k: int = 100,
                 name: Text = "factorized_top_k",
                 **kwargs):
        super(FactorizedTopK, self).__init__(name=name, **kwargs)

        if metrics is None:
            metrics = [
                tf.keras.metrics.TopKCategoricalAccuracy(
                    k=n, name=f"{self.name}/top_{n}_categorical_accuracy")
                for n in [1, 5, 10, 50, 100]
            ]

        if isinstance(candidates, tf.data.Dataset):
            candidates = layers.factorized_top_k.Streaming(k=k).index(candidates)

        self._candidates = candidates
        self._metrics = metrics
        self._k = k

    def update_state(self, 
                     query_embeddings: tf.Tensor,
                     true_candidate_embeddings: tf.Tensor) -> tf.Operation:
        """Update metric"""
        
        positive_scores = tf.reduce_sum(
            query_embeddings * true_candidate_embeddings, axis=1, keepdims=True)
        
        top_k_predictions, _ = self._candidates(query_embeddings, k=self._k)

        y_true = tf.concat([
            tf.ones(tf.shape(positive_scores)),
            tf.zeros_like(top_k_predictions)
        ], axis=1)
        y_pred = tf.concat([
            positive_scores, 
            top_k_predictions
        ], axis=1)

        update_ops = []
        for metric in self._metrics:
            update_ops.append(metric.update_state(y_true=y_true, y_pred=y_pred))

        return tf.group(update_ops)

    def reset_states(self) -> None:
        """Resets the metrics."""
        for metric in self.metrics:
            metric.reset_states()

    def result(self) -> List[tf.Tensor]:
        """Returns a list of metric results."""

        return [metric.result() for metric in self.metrics]
        
