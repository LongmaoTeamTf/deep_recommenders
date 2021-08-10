#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True

import os
import tempfile
import numpy as np
import tensorflow as tf

from deep_recommenders.keras.layers.xdeepfm import CIN


class TestCIN(tf.test.TestCase):
    
    def test_invalid_inputs_type(self):
        """ 测试输入类型 """
        with self.assertRaisesRegexp(ValueError,
                r"`CIN` layer's inputs type should be `tuple`."):
            inputs = np.random.normal(size=(2, 3, 5)).astype(np.float32)
            CIN(feature_map=3)(inputs)

    def test_invalid_inputs_ndim(self):
        """ 测试输入维度 """
        with self.assertRaisesRegexp(ValueError,
                r"`x0` and `x` dim should be 3."):
            inputs = np.random.normal(size=(2, 15)).astype(np.float32)
            CIN(feature_map=3)((inputs, inputs))

    def test_outputs(self):
        """ 测试输出是否正确 """
        x0 = np.asarray([[[0.1, 0.2, 0.3],[0.4, 0.5, 0.6]]]).astype(np.float32)
        x = np.asarray([[[0.1, 0.2, 0.3],[0.4, 0.5, 0.6]]]).astype(np.float32)
        outputs = CIN(
            feature_map=2, 
            activation="relu", 
            kernel_init="ones")((x0, x))
        expect_outputs = np.asarray([
            [[0.25, 0.49, 0.81],
            [0.25, 0.49, 0.81]]
        ]).astype(np.float32)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(outputs, expect_outputs)

    def test_bias(self):
        """ 测试bias """
        x0 = np.asarray([[[0.1, 0.2, 0.3],[0.4, 0.5, 0.6]]]).astype(np.float32)
        x = np.asarray([[[0.1, 0.2, 0.3],[0.4, 0.5, 0.6]]]).astype(np.float32)
        outputs = CIN(
            feature_map=2,
            use_bias=True,
            activation="relu", 
            kernel_init="ones",
            bias_init="ones")((x0, x))
        expect_outputs = np.asarray([
            [[1.25, 1.49, 1.81],
            [1.25, 1.49, 1.81]]
        ]).astype(np.float32)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(outputs, expect_outputs)

    def test_train_model(self):
        """ 测试模型训练 """

        def get_model():
            x0 = tf.keras.layers.Input(shape=(12, 10))
            x = CIN(feature_map=3)((x0, x0))
            x = CIN(feature_map=3)((x0, x))
            x = tf.keras.layers.Flatten()(x)
            outputs = tf.keras.layers.Dense(1)(x)
            model = tf.keras.Model(x0, outputs)
            return model
    
        x0 = np.random.uniform(size=(10, 12, 10))
        y = np.random.uniform(size=(10,))

        model = get_model()
        model.compile(loss="mse")
        model.fit(x0, y, verbose=0)

    def test_save_model(self):
        """ 测试模型保存 """

        def get_model():
            x0 = tf.keras.layers.Input(shape=(12, 10))
            x = CIN(feature_map=3)((x0, x0))
            x = CIN(feature_map=3)((x0, x))
            x = tf.keras.layers.Flatten()(x)
            logits = tf.keras.layers.Dense(1)(x)
            model = tf.keras.Model(x0, logits)
            return model
    
        x0 = np.random.uniform(size=(10, 12, 10))

        model = get_model()
        model_pred = model.predict(x0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "xDeepFM")
            model.save(
                path,
                options=tf.saved_model.SaveOptions(namespace_whitelist=["xDeepFm"]))
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict(x0)
        for model_layer, loaded_layer in zip(model.layers, loaded_model.layers):
            assert model_layer.get_config() == loaded_layer.get_config()
        self.assertAllEqual(model_pred, loaded_pred)


if __name__ == "__main__":
    tf.test.main()
