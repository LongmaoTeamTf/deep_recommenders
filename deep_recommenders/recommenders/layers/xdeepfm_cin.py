"""
@Description: eXtreme Deep Factorization Machine (xDeepFM)
@version: https://arxiv.org/abs/1803.05170
@License: MIT
@Author: Wang Yao
@Date: 2020-11-28 11:16:54
@LastEditors: Wang Yao
@LastEditTime: 2021-02-18 11:56:08
"""
from typing import Optional, Union, Text, Tuple, Callable

import os
import tempfile
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class CIN(tf.keras.layers.Layer):
    """ Compressed Interaction Network in xDeepFM """

    def __init__(self, 
                 feature_map: Optional[int] = 3,
                 use_bias: bool = False,
                 activation: Union[Text, None, Callable] = "sigmoid",
                 kernel_init: Union[Text, tf.keras.initializers.Initializer] = "truncated_normal",
                 kernel_regu: Union[Text, None, tf.keras.regularizers.Regularizer] = None,
                 bias_init: Union[Text, tf.keras.initializers.Initializer] = "zeros",
                 bias_regu: Union[Text, None, tf.keras.regularizers.Regularizer] = None,
                 **kwargs):
    
        super(CIN, self).__init__(**kwargs)
        
        self._feature_map = feature_map
        self._use_bias = use_bias
        self._activation = tf.keras.activations.get(activation)
        self._kernel_init = tf.keras.initializers.get(kernel_init)
        self._kernel_regu = tf.keras.regularizers.get(kernel_regu)
        self._bias_init = tf.keras.initializers.get(bias_init)
        self._bias_regu = tf.keras.regularizers.get(bias_regu)
        
    def build(self, input_shape):

        if not isinstance(input_shape, tuple):
            raise ValueError("`CIN` layer's inputs type should be `tuple`."
                             "Got `CIN` layer's inputs type = `{}`".format(
                                 type(input_shape)))

        if len(input_shape) != 2:
            raise ValueError("`CIN` Layer inputs tuple length should be 2."
                             "Got `length` = {}".format(len(input_shape)))
        
        x0_shape, x_shape = input_shape
        self._x0_fields, self._x_fields = x0_shape[1], x_shape[1]

        self._kernel = self.add_weight(
            shape=(1, self._x0_fields * self._x_fields, self._feature_map), 
            initializer=self._kernel_init,
            regularizer=self._kernel_regu,
            trainable=True,
            name="kernel"
        )
        if self._use_bias is True:
            self._bias = self.add_weight(
                shape=(self._feature_map,),
                initializer=self._bias_init,
                regularizer=self._bias_regu,
                trainable=True,
                name="bias"
            )
        self.built = True
        
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):

        x0, x = inputs
        
        if tf.keras.backend.ndim(x0) != 3 or \
            tf.keras.backend.ndim(x) != 3:
            raise ValueError("`x0` and `x` dim should be 3."
                             "Got `x0` dim = {}, `x` dim = {}".format(
                                 tf.keras.backend.ndim(x0),
                                 tf.keras.backend.ndim(x)))

        field_dim = x0.shape[-1]
        x0 = tf.split(x0, field_dim, axis=-1)
        x = tf.split(x, field_dim, axis=-1)

        outer = tf.matmul(x0, x, transpose_b=True)
        outer = tf.reshape(outer, shape=[field_dim, -1, self._x0_fields * self._x_fields])
        outer = tf.transpose(outer, perm=[1, 0, 2])
    
        conv_out = tf.nn.conv1d(outer, self._kernel, stride=1, padding="VALID")

        if self._use_bias is True:
            conv_out = tf.nn.bias_add(conv_out, self._bias)

        outputs = self._activation(conv_out)
        return tf.transpose(outputs, perm=[0, 2, 1])

    def get_config(self):
        config = {
            "feature_map":
                self._feature_map,
            "use_bias":
                self._use_bias,
            "activation":
                tf.keras.activations.serialize(self._activation),
            "kernel_initializer":
                tf.keras.initializers.serialize(self._kernel_init),
            "kernel_regularizer":
                tf.keras.regularizers.serialize(self._kernel_regu),
            "bias_initializer":
                tf.keras.initializers.serialize(self._bias_init),
            "bias_regularizer":
                tf.keras.regularizers.serialize(self._bias_regu),
        }
        base_config = super(CIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

class CINTest(tf.test.TestCase):
    
    def test_invalid_inputs_type(self):
        """ 测试输入类型 """
        with self.assertRaisesRegexp(ValueError,
                r"`CIN` layer's inputs type should be `tuple`."):
            inputs = np.random.random((2, 3, 5)).astype(np.float32)
            CIN(feature_map=3)(inputs)

    def test_invalid_inputs_ndim(self):
        """ 测试输入维度 """
        with self.assertRaisesRegexp(ValueError,
                r"`x0` and `x` dim should be 3."):
            inputs = np.random.random((2, 15)).astype(np.float32)
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
            path = os.path.join(tmp, "cin_model")
            model.save(
                path,
                options=tf.saved_model.SaveOptions(namespace_whitelist=["Addons"]))
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict(x0)
        for model_layer, loaded_layer in zip(model.layers, loaded_model.layers):
            assert model_layer.get_config() == loaded_layer.get_config()
        self.assertAllEqual(model_pred, loaded_pred)


if __name__ == "__main__":
    tf.test.main()
