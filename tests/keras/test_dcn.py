#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True

import os
import tempfile
import numpy as np
import tensorflow as tf

from deep_recommenders.keras.models.ranking.dcn import Cross


class TestDCN(tf.test.TestCase):

    def test_cross_full_matrix(self):
        x0 = np.asarray([[0.1, 0.2, 0.3]]).astype(np.float32)
        x = np.asarray([[0.4, 0.5, 0.6]]).astype(np.float32)
        
        cross = Cross(projection_dim=None, kernel_init="ones")
        output = cross(x0, x)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(np.asarray([[0.55, 0.8, 1.05]]), output)

    def test_cross_save_model(self):

        def get_model():
            x0 = tf.keras.layers.Input(shape=(13,))
            x1 = Cross(projection_dim=None)(x0, x0)
            x2 = Cross(projection_dim=None)(x0, x1)
            logits = tf.keras.layers.Dense(units=1)(x2)
            return tf.keras.Model(x0, logits)

        model = get_model()
        random_input = np.random.uniform(size=(10, 13))
        model_pred = model.predict(random_input)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dcn_model")
            model.save(path)
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict(random_input)
        for i in range(len(model.layers)):
            assert model.layers[i].get_config() == loaded_model.layers[i].get_config()
        self.assertAllClose(model_pred, loaded_pred)


if __name__ == "__main__":
    tf.test.main()
