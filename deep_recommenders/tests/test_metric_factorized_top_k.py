"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2021-02-26 15:24:31
@LastEditors: Wang Yao
@LastEditTime: 2021-02-26 15:29:05
"""
import sys
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf

from absl.testing import parameterized
from deep_recommenders import layers
from deep_recommenders.metrics import FactorizedTopK


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
