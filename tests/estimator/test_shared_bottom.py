#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True

import tempfile
import tensorflow as tf
from absl.testing import parameterized
from deep_recommenders.estimator.models.multi_task_learning import SharedBottom

if tf.__version__ >= "2.0.0":
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()


class TestSharedBottom(tf.test.TestCase, parameterized.TestCase):

    def test_shared_bottom(self):
        num_examples = 100
        example_dim = 10
        x = tf.random_normal(shape=(num_examples, example_dim))

        with self.session() as sess:
            shared_bottom = SharedBottom(num_tasks=2,
                                         task_units=[10, 5],
                                         task_target_activations=[tf.nn.sigmoid, tf.nn.sigmoid],
                                         bottom_units=[32, 16])
            y = shared_bottom(x)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            sess.run(y)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.test.main()
