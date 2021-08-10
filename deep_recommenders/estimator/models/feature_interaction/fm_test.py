#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode = True

import tensorflow as tf
from deep_recommenders.estimator.models.feature_interaction import fm


class TestFM(tf.test.TestCase):

    def test_fm_with_factors(self):
        inputs = tf.random_normal(shape=(2, 3))

        with self.test_session() as sess:
            y = fm(inputs, num_factors=5)
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(y)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.test.main()
