#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf

from absl.testing import parameterized
from deep_recommenders.models.multi_task_learning.mixture_of_experts import _synthetic_data


class TestMixtureOfExperts(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters(42, 256, 1024, 2021)
    def test_synthetic_data(self, random_seed):
        np.random.seed(random_seed)
        _, (y1, y2) = _synthetic_data(1000, p=0.8)
        cor = np.corrcoef(y1, y2)
        print(cor)


if __name__ == '__main__':
    tf.test.main()
