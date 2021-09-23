#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True

import tensorflow as tf

if tf.__version__ >= "2.0.0":
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

from deep_recommenders.estimator.models.feature_interaction import fm


class TestFM(tf.test.TestCase):

    def test_fm(self):
        inputs = tf.random_normal(shape=(10, 2, 3))

        with self.session() as sess:
            y = fm(inputs)
            init = tf.global_variables_initializer()
            sess.run(init)
            pred = sess.run(y)
            self.assertAllEqual(pred.shape, (10, 1))


if __name__ == '__main__':
    tf.test.main()
