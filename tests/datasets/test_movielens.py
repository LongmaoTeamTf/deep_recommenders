#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from absl.testing import parameterized

if tf.__version__ < "2.0.0":
    tf.enable_eager_execution()

if tf.__version__ >= "2.0.0":
    import tensorflow.compat.v1 as tf

from deep_recommenders.datasets import MovieLens


class TestMovieLens(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters(16, 64, 256, 1024)
    def test_batch(self, batch_size):
        movielens = MovieLens()
        dataset = movielens.dataset(batch_size=batch_size)
        for x, y in dataset.take(1):
            self.assertAllEqual(x["UserID"].shape, (batch_size,))
            self.assertAllEqual(y.shape, (batch_size,))

    @parameterized.parameters(1, 2, 3)
    def test_repeat(self, epochs):
        movielens = MovieLens()
        dataset = movielens.dataset(epochs, 2048)
        steps = 0
        for _ in dataset:
            steps += 1
        expect_steps = (movielens.num_ratings * epochs) // 2048 + 1
        self.assertAllEqual(steps, expect_steps)

    def test_map(self):
        movielens = MovieLens()
        dataset = movielens.dataset()
        dataset = dataset.map(lambda _, y: tf.where(y > 3, tf.ones_like(y), tf.zeros_like(y)))
        for y in dataset.take(1):
            self.assertLess(tf.reduce_sum(y), 256)


if __name__ == '__main__':
    tf.test.main()
