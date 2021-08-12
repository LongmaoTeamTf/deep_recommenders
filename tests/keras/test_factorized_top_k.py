#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True

import os
import numpy as np
import tensorflow as tf

from absl.testing import parameterized
from deep_recommenders.keras.models.retrieval import factorized_top_k
from deep_recommenders.keras.models.retrieval import FactorizedTopK


class TestFactorizedTopK(tf.test.TestCase, parameterized.TestCase):

    def test_take_long_axis(self):
        arr = tf.constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        indices = tf.constant([[0, 1], [2, 1]])
        out = factorized_top_k._take_long_axis(arr, indices)
        expected_out = tf.constant([[0.1, 0.2], [0.6, 0.5]])
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(out, expected_out)

    def test_exclude(self):
        scores = tf.constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        identifiers = tf.constant([[0, 1, 2], [3, 4, 5]])
        exclude = tf.constant([[1, 2], [3, 5]])
        k = 1
        x, y = factorized_top_k._exclude(scores, identifiers, exclude, k)
        expected_x = tf.constant([[0.1], [0.5]])
        expected_y = tf.constant([[0], [4]])
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose((x, y), (expected_x, expected_y))

    @parameterized.parameters(np.str, np.float32, np.float64, np.int32, np.int64)
    def test_faiss(self, identifier_dtype):
        num_candidates, num_queries = (5000, 4)

        rng = np.random.RandomState(42) # pylint: disable=no-member
        candidates = rng.normal(size=(num_candidates, 4)).astype(np.float32)
        query = rng.normal(size=(num_queries, 4)).astype(np.float32)
        candidate_names = np.arange(num_candidates).astype(identifier_dtype)

        faiss_topk = factorized_top_k.Faiss(k=10)
        faiss_topk.index(candidates, candidate_names)

        for _ in range(100):
            pre_serialization_results = faiss_topk(query[:2])

        path = os.path.join(self.get_temp_dir(), "query_model")
        faiss_topk.save(
            path,
            options=tf.saved_model.SaveOptions(namespace_whitelist=["Faiss"]))
        loaded = tf.keras.models.load_model(path)

        for _ in range(100):
            post_serialization_results = loaded(tf.constant(query[:2]))

        self.assertAllEqual(post_serialization_results, pre_serialization_results)
    
    @parameterized.parameters(np.float32, np.float64)
    def test_faiss_with_no_identifiers(self, candidate_dtype):
        """ 测试构建无唯一标识索引 """
        num_candidates = 5000

        candidates = np.random.normal(size=(num_candidates, 4)).astype(candidate_dtype)
        faiss_topk = factorized_top_k.Faiss(k=10)
        faiss_topk.index(candidates, identifiers=None)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(num_candidates, faiss_topk._searcher.ntotal)

    @parameterized.parameters(np.str, np.float32, np.float64, np.int32, np.int64)
    def test_faiss_with_dataset(self, identifier_dtype):
        num_candidates = 5000
        
        candidates = tf.data.Dataset.from_tensor_slices(
            np.random.normal(size=(num_candidates, 4)).astype(np.float32))
        identifiers = tf.data.Dataset.from_tensor_slices(
            np.arange(num_candidates).astype(identifier_dtype))
        faiss_topk = factorized_top_k.Faiss(k=10)
        faiss_topk.index(candidates.batch(100), identifiers=identifiers)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(num_candidates, faiss_topk._searcher.ntotal)

    @parameterized.parameters(
        factorized_top_k.Streaming,
        factorized_top_k.BruteForce,
        factorized_top_k.Faiss,
        None)
    def test_factorized_topk_metrics(self, top_k_layer):

        rng = np.random.RandomState(42)  # pylint: disable=no-member

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
