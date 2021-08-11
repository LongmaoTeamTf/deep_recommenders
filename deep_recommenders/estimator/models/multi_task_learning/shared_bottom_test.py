#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True

import tensorflow as tf

if tf.__version__ >= "2.3.0":
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

import tempfile
from absl.testing import parameterized

from deep_recommenders.estimator.models.multi_task_learning import synthetic_data_input_fn
from deep_recommenders.estimator.models.multi_task_learning import shared_bottom
from deep_recommenders.estimator.models.multi_task_learning import shared_bottom_estimator


class TestSharedBottom(tf.test.TestCase, parameterized.TestCase):

    def test_shared_bottom(self):
        num_examples = 100
        example_dim = 10
        x = tf.random_normal(shape=(num_examples, example_dim))

        with self.session() as sess:
            y = shared_bottom(x, 2, [32, 16], [10, 5], [None, None])
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            sess.run(y)

    @parameterized.parameters(
        {
            "num_tasks": 2,
            "bottom_units": [32, 16],
            "task_units": [10, 5],
            "task_output_activations": [None, None],
            "task_losses": ["mse", "mse"],
            "lr": 0.001
        }
    )
    def test_shared_bottom_estimator(self, **params):

        def _map_fn(x, y):
            return {"inputs": x}, y

        with tempfile.TemporaryDirectory() as temp_dir:
            estimator = shared_bottom_estimator(temp_dir, 8, 8, params)
            estimator.train(input_fn=lambda: synthetic_data_input_fn(1000).map(map_func=_map_fn))

            features = {"inputs": tf.placeholder(tf.float32, (None, 100), name="inputs")}

            serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features)

            estimator.export_saved_model(temp_dir + "/saved_model", serving_input_receiver_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.test.main()
