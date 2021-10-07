#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf

if tf.__version__ >= "2.0.0":
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

from absl.testing import parameterized
from deep_recommenders.estimator.models.multi_task_learning.mixture_of_experts import synthetic_data
from deep_recommenders.estimator.models.multi_task_learning import OMoE
from deep_recommenders.estimator.models.multi_task_learning import MMoE


class TestMixtureOfExperts(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters(42, 256, 1024, 2021)
    def test_synthetic_data(self, random_seed):
        np.random.seed(random_seed)
        _, (y1, y2) = synthetic_data(1000, p=0.8)
        cor = np.corrcoef(y1, y2)
        print(cor)

    def test_one_gate(self):

        num_examples = 1000
        example_dim = 128

        inputs = tf.random.normal(shape=(num_examples, example_dim))

        outputs = OMoE(inputs,
                       num_tasks=2,
                       num_experts=3,
                       task_hidden_units=[10, 5],
                       task_output_activations=[None, None],
                       expert_hidden_units=[64, 32],
                       expert_hidden_activation=tf.nn.relu,
                       task_hidden_activation=tf.nn.relu,
                       task_dropout=None)

    def test_multi_gate(self):

        num_examples = 1000
        example_dim = 128

        inputs = tf.random.normal(shape=(num_examples, example_dim))

        outputs = MMoE(inputs,
                       num_tasks=2,
                       num_experts=3,
                       task_hidden_units=[10, 5],
                       task_output_activations=[None, None],
                       expert_hidden_units=[64, 32],
                       expert_hidden_activation=tf.nn.relu,
                       task_hidden_activation=tf.nn.relu,
                       task_dropout=None)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.test.main()
