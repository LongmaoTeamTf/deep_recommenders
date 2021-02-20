"""
@Description: Factorization Machines
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-12-03 18:01:05
@LastEditors: Wang Yao
@LastEditTime: 2021-02-19 15:08:40
"""
from typing import Optional, Union, Text

import os
import tempfile
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class FM(tf.keras.layers.Layer):
    """ Factorization Machine """

    def __init__(self, 
                 factors: Union[int, None] = 10,
                 kernel_init: Union[Text, tf.keras.initializers.Initializer] = "truncated_normal",
                 kernel_regu: Union[Text, None, tf.keras.regularizers.Regularizer] = None,
                 **kwargs):

        super(FM, self).__init__(**kwargs)
        
        self._factors = factors
        self._kernel_init = kernel_init
        self._kernel_regu = kernel_regu

        if (self._factors is not None) and (self._factors <= 0):
            raise ValueError("`factors` should be bigger than 0. "
                    "Got `factors` = {}".format(self._factors))

    def build(self, input_shape):
        last_dim = input_shape[-1]

        if self._factors is not None:
            self._kernel = self.add_weight(
                shape=(last_dim, self._factors),
                initializer=tf.keras.initializers.get(self._kernel_init),
                regularizer=tf.keras.regularizers.get(self._kernel_regu),
                trainable=True,
                name="kernel"
            )
        self.built = True

    def call(self, x: tf.Tensor):
        
        if tf.keras.backend.ndim(x) != 3:
            raise ValueError("`x` dim should be 3. Got `x` dim = {}".format(
                    tf.keras.backend.ndim(x)))

        if self._factors is None:
            embed_x = x
            square_embed_x = tf.pow(x, 2)
        else:
            embed_x = tf.matmul(x, self._kernel)
            square_embed_x = tf.matmul(tf.pow(x, 2), tf.pow(self._kernel, 2))

        outputs = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.pow(tf.reduce_sum(embed_x, axis=1, keepdims=True), 2),
                tf.reduce_sum(square_embed_x, axis=1, keepdims=True)
            ), axis=-1, keepdims=False
        )
        return outputs

    def get_config(self):
        if self._factors is None:
            config = {
                "factors":
                    self._factors,
            }
        else:
            config = {
                "factors":
                    self._factors,
                "kernel_initializer":
                    tf.keras.initializers.serialize(self._kernel_init),
                "kernel_regularizer":
                    tf.keras.regularizers.serialize(self._kernel_regu),
            }
        base_config = super(FM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FMTest(tf.test.TestCase):

    def test_invalid_factors(self):
        """ 测试 factors <= 0 """
        with self.assertRaisesRegexp(ValueError,
                                 r"should be bigger than 0"):
            x = np.random.random((10, 12, 5)).astype(np.float32) # pylint: disable=no-member
            layer = FM(factors=-1)
            layer(x)

    def test_int_factors(self):
        """ 测试 factors = int """
        x = np.asarray([[
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0],
        ]]).astype(np.float32)
        layer = FM(factors=2, kernel_init="ones")
        output = layer(x)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(np.asarray([[2.]]).astype(np.float32), output)

    def test_none_factors(self):
        """ 测试 factors = None """
        x = np.random.random((10, 12, 5)).astype(np.float32) # pylint: disable=no-member
        layer = FM(factors=None)
        layer(x)

    def test_x_invalid_dim(self):
        """ 测试 x dim invalid """
        with self.assertRaisesRegexp(ValueError,
                                r"`x` dim should be 3."):
            x = np.random.random((10, 60)).astype(np.float32) # pylint: disable=no-member
            layer = FM()
            layer(x)

    def test_outputs_with_diff_factors(self):
        """ 测试 factors = None 和 factors = 10 输出是否相等 """
        x = np.random.random((10, 12, 5)).astype(np.float32) # pylint: disable=no-member
        factors = 5

        def identity(shape, dtype=None):
            return np.eye(shape[-1])
            
        layer_factors_none = FM(factors=None)
        layer_factors_5 = FM(factors=factors, kernel_init=identity)
        layer_factors_none_output = layer_factors_none(x)
        layer_factors_5_output = layer_factors_5(x)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(layer_factors_none_output, layer_factors_5_output)

    def test_train_model(self):
        """ 测试训练模型 """

        def get_model():
            inputs = tf.keras.layers.Input(shape=(12, 5,))
            x = FM(factors=None)(inputs)
            logits = tf.keras.layers.Dense(units=1)(x)
            model = tf.keras.Model(inputs, logits)
            return model

        model = get_model()
        random_input = np.random.uniform(size=(10, 12, 5))
        random_output = np.random.uniform(size=(10,))
        model.compile(loss="mse")
        model.fit(random_input, random_output, verbose=0)

    def test_save_model(self):
        """ 测试保存模型 """

        def get_model():
            inputs = tf.keras.layers.Input(shape=(12, 5,))
            x = FM(factors=None)(inputs)
            logits = tf.keras.layers.Dense(units=1)(x)
            model = tf.keras.Model(inputs, logits)
            return model

        model = get_model()
        random_input = np.random.uniform(size=(10, 12, 5))
        model_pred = model.predict(random_input)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fm_model")
            model.save(
                path,
                options=tf.saved_model.SaveOptions(namespace_whitelist=["Addons"]))
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict(random_input)
        for model_layer, loaded_layer in zip(model.layers, loaded_model.layers):
            assert model_layer.get_config() == loaded_layer.get_config()
        self.assertAllEqual(model_pred, loaded_pred)


if __name__ == "__main__":
    tf.test.main()