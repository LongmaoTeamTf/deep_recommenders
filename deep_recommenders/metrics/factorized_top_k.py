"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2021-02-23 10:41:28
@LastEditors: Wang Yao
@LastEditTime: 2021-02-24 10:54:56
"""
from absl.testing import parameterized
from typing import List, Optional, Sequence, Union

import numpy as np
import tensorflow as tf

from deep_recommenders import layers


class FactorizedTopK(tf.keras.layers.Layer):
    """ Metric for a retrieval model. """

    def __init__(self, 
                 candidates: Union[layers.factorized_top_k.TopK, tf.data.Dataset], 
                 metrics: Optional[Sequence[tf.keras.metrics.Metric]] = None,
                 k: int = 100,
                 **kwargs):
        super(FactorizedTopK, self).__init__(**kwargs)

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
            tf.ones_like(positive_scores),
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
        

class TestFactorizedTopKMetics(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters(
        layers.factorized_top_k.Streaming,
        layers.factorized_top_k.BruteForce,
        layers.factorized_top_k.Faiss,
        None)
    def test_factorized_topk_metrics(self, top_k_layer):

        rng = np.random.RandomState(42) # pylint: disable=no-member

        num_candidates, num_queries, embedding_dim = (100, 10, 4)

        candidates = rng.normal(size=(num_candidates, embedding_dim)).astype(np.float32)
        queries = rng.normal(size=(num_queries, embedding_dim)).astype(np.float32)
        true_candidates = rng.normal(size=(num_queries, embedding_dim)).astype(np.float32)

        positive_scores = (queries * true_candidates).sum(axis=1, keepdims=True)
        candidate_scores = queries @ candidates.T

        all_scores = np.concatenate([positive_scores, candidate_scores], axis=1)

        ks = [1, 5, 10, 50]

        candidates = tf.data.Dataset.from_tensor_slices(candidates).batch(32)

        if top_k_layer is not None:
            candidates = top_k_layer().index(candidates)

        metric = FactorizedTopK(
            candidates=candidates,
            metrics=[
                tf.keras.metrics.TopKCategoricalAccuracy(
                    k=x, name=f"top_{x}_categorical_accuracy") for x in ks
            ],
            k=max(ks),
        )
        
        metric.update_state(
            query_embeddings=queries, true_candidate_embeddings=true_candidates)
        
        for k, metric_value in zip(ks, metric.result()):
            in_top_k = tf.math.in_top_k(
                targets=np.zeros(num_queries).astype(np.int32),
                predictions=all_scores,
                k=k)
            self.assertAllClose(metric_value, in_top_k.numpy().mean())


if __name__ == "__main__":
    tf.test.main()
