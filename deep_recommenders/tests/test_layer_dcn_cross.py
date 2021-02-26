"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2021-02-26 15:08:52
@LastEditors: Wang Yao
@LastEditTime: 2021-02-26 15:15:38
"""
import sys
sys.dont_write_bytecode = True

import os
import tempfile
import numpy as np
import tensorflow as tf

from deep_recommenders.layers.dcn_cross import Cross


class TestCross(tf.test.TestCase):

    def test_full_matrix(self):
        x0 = np.asarray([[0.1, 0.2, 0.3]]).astype(np.float32)
        x = np.asarray([[0.4, 0.5, 0.6]]).astype(np.float32)
        
        layer = Cross(projection_dim=None, kernel_init="ones")
        output = layer(x0, x)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(np.asarray([[0.55, 0.8, 1.05]]), output)

    def test_save_model(self):

        def get_model():
            x0 = tf.keras.layers.Input(shape=(13,))
            x1 = Cross(projection_dim=None)(x0, x0)
            x2 = Cross(projection_dim=None)(x0, x1)
            logits = tf.keras.layers.Dense(units=1)(x2)
            model = tf.keras.Model(x0, logits)
            return model

        model = get_model()
        random_input = np.random.uniform(size=(10, 13))
        model_pred = model.predict(random_input)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dcn_model")
            model.save(
                path,
                options=tf.saved_model.SaveOptions(namespace_whitelist=["Addons"]))
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict(random_input)
        for i in range(3):
            assert model.layers[i].get_config() == loaded_model.layers[i].get_config()
        self.assertAllEqual(model_pred, loaded_pred)


if __name__ == "__main__":
    tf.test.main()