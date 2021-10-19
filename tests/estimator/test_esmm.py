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
from deep_recommenders.datasets import SyntheticForMultiTask
from deep_recommenders.estimator.models.multi_task_learning import ESMM


class TestESMM(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters(32, 64, 128, 512)
    def test_mmoe(self, batch_size):

        def build_columns():
            return [
                tf.feature_column.numeric_column("C{}".format(i))
                for i in range(100)
            ]

        columns = build_columns()
        model = ESMM(columns, hidden_units=[32, 10])

        dataset = SyntheticForMultiTask(5000)

        with self.session() as sess:
            iterator = dataset.input_fn(batch_size=batch_size).make_one_shot_iterator()
            x, y = iterator.get_next()
            p_cvr, p_ctr, p_ctcvr = model(x)
            sess.run(tf.global_variables_initializer())
            p_cvr = sess.run(p_cvr)
            p_ctr = sess.run(p_ctr)
            p_ctcvr = sess.run(p_ctcvr)
            self.assertAllEqual(p_cvr.shape, (batch_size, 1))
            self.assertAllEqual(p_ctr.shape, (batch_size, 1))
            self.assertAllEqual(p_ctcvr.shape, (batch_size, 1))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.test.main()
