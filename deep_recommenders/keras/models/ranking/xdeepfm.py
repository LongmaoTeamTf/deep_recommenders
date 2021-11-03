#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Optional, Union, Text, Tuple

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class CIN(tf.keras.layers.Layer):
    """ Compressed Interaction Network in xDeepFM """

    def __init__(self, 
                 feature_map: Optional[int] = 3,
                 use_bias: bool = False,
                 activation: Union[Text, None, tf.keras.layers.Layer] = "sigmoid",
                 kernel_init: Union[Text, tf.keras.initializers.Initializer] = "truncated_normal",
                 kernel_regu: Union[Text, None, tf.keras.regularizers.Regularizer] = None,
                 bias_init: Union[Text, tf.keras.initializers.Initializer] = "zeros",
                 bias_regu: Union[Text, None, tf.keras.regularizers.Regularizer] = None,
                 **kwargs):
    
        super(CIN, self).__init__(**kwargs)
        
        self._feature_map = feature_map
        self._use_bias = use_bias

        if isinstance(activation, tf.keras.layers.Layer):
            self._activation = activation      
        elif isinstance(activation, str):
            self._activation = tf.keras.activations.get(activation)
        else:
            self._activation = None

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
        self._x0_fields = x0_shape[1]
        self._x_fields = x_shape[1]

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
        
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], **kwargs):

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
            "kernel_init":
                tf.keras.initializers.serialize(self._kernel_init),
            "kernel_regu":
                tf.keras.regularizers.serialize(self._kernel_regu),
            "bias_init":
                tf.keras.initializers.serialize(self._bias_init),
            "bias_regu":
                tf.keras.regularizers.serialize(self._bias_regu),
        }
        base_config = super(CIN, self).get_config()
        return {**base_config, **config}
