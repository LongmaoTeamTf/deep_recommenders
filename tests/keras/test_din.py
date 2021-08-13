#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True

import os
import tempfile
import numpy as np
import tensorflow as tf

from absl.testing import parameterized
from deep_recommenders.keras.models.ranking import din
from deep_recommenders.keras.models.ranking import DIN


class TestDIN(tf.test.TestCase, parameterized.TestCase):

    def test_activation_unit_noiteract(self):
        
        x = np.random.normal(size=(3, 5))
        y = np.random.normal(size=(3, 5))

        activation_unit = din.ActivationUnit(10, kernel_init="ones")
        outputs = activation_unit(x, y)

        dense = tf.keras.layers.Dense(10, activation="relu", kernel_initializer="ones")
        expected_outputs = tf.math.reduce_sum(
            dense(np.concatenate([x, y], axis=1)), axis=1, keepdims=True)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(outputs, expected_outputs)

    def test_activation_unit_iteract(self):

        x = np.random.normal(size=(3, 5))
        y = np.random.normal(size=(3, 5))
        
        interacter = tf.keras.layers.Subtract()

        activation_unit = din.ActivationUnit(10,
                                             interacter=interacter, kernel_init="ones")
        outputs = activation_unit(x, y)

        dense = tf.keras.layers.Dense(10, activation="relu", kernel_initializer="ones")
        expected_outputs = tf.math.reduce_sum(
            dense(np.concatenate([x, y, x - y], axis=1)), axis=1, keepdims=True)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(outputs, expected_outputs)

    @parameterized.parameters(1e-7, 1e-8, 1e-9, 1e-10)
    def test_dice(self, epsilon):
        
        inputs = np.asarray([[-0.2, -0.1, 0.1, 0.2]]).astype(np.float32)
        
        outputs = din.Dice(epsilon=epsilon)(inputs)

        p = (inputs - inputs.mean()) / np.math.sqrt(inputs.std() + epsilon)
        p = 1 / (1 + np.exp(-p))
       
        x = tf.where(inputs > 0, x=inputs, y=tf.zeros_like(inputs))
        expected_outputs = tf.where(x > 0, x=p*x, y=(1-p)*x)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(outputs, expected_outputs)

    def test_din(self):

        def build_model():
            x = tf.keras.layers.Input(shape=(5,))
            y = tf.keras.layers.Input(shape=(5,))
            interacter = tf.keras.layers.Subtract()
            activation_unit = din.ActivationUnit(10, interacter=interacter)
            outputs = activation_unit(x, y)
            return tf.keras.Model([x, y], outputs)
        
        x_embeddings = np.random.normal(size=(10, 5))
        y_embeddings = np.random.normal(size=(10, 5))
        labels = np.random.normal(size=(10,))

        model = build_model()
        model.compile(loss="mse")
        model.fit([x_embeddings, y_embeddings], labels, verbose=0)

        model_pred = model.predict([x_embeddings, y_embeddings])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "din_model")
            model.save(
                path,
                options=tf.saved_model.SaveOptions(namespace_whitelist=["din"]))
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict([x_embeddings, y_embeddings])
        
        self.assertAllEqual(model_pred, loaded_pred)


if __name__ == "__main__":
    tf.test.main()
