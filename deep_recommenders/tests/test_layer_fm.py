"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2021-02-26 15:15:08
@LastEditors: Wang Yao
@LastEditTime: 2021-03-11 15:19:01
"""
import sys
sys.dont_write_bytecode = True

import os
import tempfile
import numpy as np
import tensorflow as tf

from deep_recommenders.layers.fm import FM


class TestFM(tf.test.TestCase):

    def test_invalid_factors(self):
        """ 测试 factors <= 0 """
        with self.assertRaisesRegexp(ValueError,
                                 r"should be bigger than 0"):
            x = np.random.random((10, 5)).astype(np.float32) # pylint: disable=no-member
            layer = FM(factors=-1)
            layer(x)

    def test_x_invalid_dim(self):
        """ 测试 x dim invalid """
        with self.assertRaisesRegexp(ValueError,
                                r"`x` dim should be 2."):
            x = np.random.normal(size=(3, 10, 5))
            fm = FM(factors=2)
            fm(x)
        with self.assertRaisesRegexp(ValueError,
                                r"`x` dim should be 3."):
            x = np.random.normal(size=(10, 5))
            fm = FM(factors=None)
            fm(x)

    def test_factors(self):
        """ 测试 factors """
        x = np.asarray([
            [1.0, 1.0, 0.0], 
            [0.0, 1.0, 0.0],
        ]).astype(np.float32)

        factors = np.asarray([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0]
        ]).astype(np.float32)

        x_sum = x @ factors
        x_square_sum = np.power(x, 2) @ np.power(factors, 2)
        expected_outputs = 0.5 * np.sum(np.power(x_sum, 2) - x_square_sum, axis=1, keepdims=True)

        fm = FM(factors=2, kernel_init="ones")
        outputs = fm(x)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(outputs, expected_outputs)

    def test_none_factors(self):
        """ 测试 factors = None """

        x = np.asarray([
            [1.0, 1.0, 0.0], 
            [0.0, 1.0, 1.0]])

        fm_factors = FM(factors=2)
        fm_factors_outputs = fm_factors(x)

        embeddings_martix = fm_factors.get_weights()[0]

        fm_nofactors_x = tf.gather(embeddings_martix, [[0, 1], [1, 2]])

        fm_nofactors = FM(factors=None)
        fm_nofactors_outputs = fm_nofactors(fm_nofactors_x)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(fm_nofactors_outputs, fm_factors_outputs)

        
    def test_train_model(self):
        """ 测试训练模型 """

        def get_model():
            inputs = tf.keras.layers.Input(shape=(10, 5,))
            x = FM(factors=None)(inputs)
            print(x)
            logits = tf.keras.layers.Dense(1)(x)
            model = tf.keras.Model(inputs, logits)
            return model

        model = get_model()
        random_inputs = np.random.uniform(size=(32, 10, 5))
        random_outputs = np.random.uniform(size=(32,))
        model.compile(loss="mse")
        model.fit(random_inputs, random_outputs, verbose=0)

    def test_save_model(self):
        """ 测试保存模型 """

        def get_model():
            inputs = tf.keras.layers.Input(shape=(10, 5,))
            x = FM(factors=None)(inputs)
            logits = tf.keras.layers.Dense(1)(x)
            model = tf.keras.Model(inputs, logits)
            return model

        model = get_model()
        random_input = np.random.uniform(size=(32, 10, 5))
        model_pred = model.predict(random_input)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fm")
            model.save(
                path,
                options=tf.saved_model.SaveOptions(namespace_whitelist=["FM"]))
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict(random_input)
        for model_layer, loaded_layer in zip(model.layers, loaded_model.layers):
            assert model_layer.get_config() == loaded_layer.get_config()
        self.assertAllEqual(model_pred, loaded_pred)


if __name__ == "__main__":
    tf.test.main()
