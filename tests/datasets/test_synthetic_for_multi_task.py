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

from deep_recommenders.datasets import SyntheticForMultiTask


class TestSyntheticForMultiTask(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters(16, 64, 256, 1024)
    def test_input_fn(self, dim):
        synthetic = SyntheticForMultiTask(1000, example_dim=dim)
        dataset = synthetic.input_fn()
        for features, labels in dataset.take(1):
            self.assertAllEqual(len(features.keys()), dim)
            self.assertAllEqual(len(labels.keys()), 2)

    @parameterized.parameters(16, 64, 256, 512)
    def test_batch_size(self, batch_size):
        synthetic = SyntheticForMultiTask(1000)
        dataset = synthetic.input_fn(batch_size=batch_size)
        for features, labels in dataset.take(1):
            self.assertAllEqual(features["C0"].shape, (batch_size, 1))


if __name__ == '__main__':
    tf.test.main()
