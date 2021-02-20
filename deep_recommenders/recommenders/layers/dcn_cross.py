"""
@Description: Cross in Deep & Cross Network (DCN)
@version: https://arxiv.org/abs/1708.05123
@License: MIT
@Author: Wang Yao
@Date: 2020-08-06 18:44:25
@LastEditors: Wang Yao
@LastEditTime: 2021-02-08 15:54:02
"""
from typing import Optional, Union, Text

import os
import tempfile
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class Cross(tf.keras.layers.Layer):
    """ Cross net in Deep & Cross Network (DCN) """

    def __init__(self, 
                 projection_dim: Optional[int] = None, 
                 diag_scale: Optional[float] = 0.0, 
                 use_bias: bool = True, 
                 kernel_init: Union[Text, tf.keras.initializers.Initializer] = "truncated_normal",
                 kernel_regu: Union[Text, None, tf.keras.regularizers.Regularizer] = None,
                 bias_init: Union[Text, tf.keras.initializers.Initializer] = "zeros",
                 bias_regu: Union[Text, None, tf.keras.regularizers.Regularizer] = None,
                 **kwargs):

        super(Cross, self).__init__(**kwargs)

        self._projection_dim = projection_dim
        self._diag_scale = diag_scale
        self._use_bias = use_bias
        self._kernel_init = tf.keras.initializers.get(kernel_init)
        self._kernel_regu = tf.keras.regularizers.get(kernel_regu)
        self._bias_init = tf.keras.initializers.get(bias_init)
        self._bias_regu = tf.keras.regularizers.get(bias_regu)

        assert self._diag_scale >= 0, \
            ValueError("diag scale must be non-negative, got {}".format(self._diag_scale))

        
    def build(self, input_shape):
        last_dim = input_shape[-1]

        if self._projection_dim is None:
            self._dense = tf.keras.layers.Dense(
                last_dim,
                kernel_initializer=self._kernel_init,
                kernel_regularizer=self._kernel_regu,
                bias_initializer=self._bias_init,
                bias_regularizer=self._bias_regu,
                use_bias=self._use_bias
            )
        else:
            if self._projection_dim < 0 or self._projection_dim > last_dim / 2:
                raise ValueError(
                    "`projection_dim` should be smaller than last_dim / 2 to improve "
                    "the model efficiency, and should be positive. Got "
                    "`projection_dim` {}, and last dimension of input {}".format(
                        self._projection_dim, last_dim))
            self._dense_u = tf.keras.layers.Dense(
                self._projection_dim,
                kernel_initializer=self._kernel_init,
                kernel_regularizer=self._kernel_regu,
                use_bias=False,
            )
            self._dense_v = tf.keras.layers.Dense(
                last_dim,
                kernel_initializer=self._kernel_init,
                bias_initializer=self._bias_init,
                kernel_regularizer=self._kernel_regu,
                bias_regularizer=self._bias_regu,
                use_bias=self._use_bias,
            )
        super(Cross, self).build(input_shape)

    def call(self, x0: tf.Tensor, x: Optional[tf.Tensor] = None):

        if x is None:
            x = x0
        
        if x0.shape[-1] != x.shape[-1]:
            raise ValueError("`x0` and `x` dim mismatch. " 
                             "Got `x0` dim = {} and `x` dim = {}".format(
                                x0.shape[-1], x.shape[-1]))
        
        if self._projection_dim is None:
            prod_output = self._dense(x)
        else:
            prod_output = self._dense_v(self._dense_u(x))

        if self._diag_scale:
            prod_output = prod_output + self._diag_scale * x

        return x0 * prod_output + x

    def get_config(self):
        config = {
            "projection_dim":
                self._projection_dim,
            "diag_scale":
                self._diag_scale,
            "use_bias":
                self._use_bias,
            "kernel_initializer":
                tf.keras.initializers.serialize(self._kernel_init),
            "kernel_regularizer":
                tf.keras.regularizers.serialize(self._kernel_regu),
            "bias_initializer":
                tf.keras.initializers.serialize(self._bias_init),
            "bias_regularizer":
                tf.keras.regularizers.serialize(self._bias_regu),
        }
        base_config = super(Cross, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CrossTest(tf.test.TestCase):

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