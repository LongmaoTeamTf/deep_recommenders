#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class ActivationUnit(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 interacter=None,
                 use_bias=True,
                 activation="relu",
                 kernel_init="truncated_normal",
                 kernel_regu=None,
                 bias_init="zeros",
                 bias_regu=None,
                 **kwargs):
        super(ActivationUnit, self).__init__(**kwargs)
        
        self._kernel_units = units
        self._interacter = interacter
        self._use_bias = use_bias

        if isinstance(activation, tf.keras.layers.Layer):
            self._kernel_activation = activation
        elif isinstance(activation, str):
            self._kernel_activation = tf.keras.activations.get(activation)
        else:
            self._kernel_activation = None
            
        self._kernel_init = tf.keras.initializers.get(kernel_init)
        self._kernel_regu = tf.keras.regularizers.get(kernel_regu)
        self._bias_init = tf.keras.initializers.get(bias_init)
        self._bias_regu = tf.keras.regularizers.get(bias_regu)
        
    def build(self, input_shape):

        self.dense_kernel = tf.keras.layers.Dense(
            self._kernel_units,
            activation=self._kernel_activation,
            use_bias=self._use_bias,
            kernel_initializer=self._kernel_init,
            kernel_regularizer=self._kernel_regu,
            bias_initializer=self._bias_init,
            bias_regularizer=self._bias_regu
        )
        self.dense_output = tf.keras.layers.Dense(
            1,
            activation=None,
            use_bias=self._use_bias,
            kernel_initializer=self._kernel_init,
            kernel_regularizer=self._kernel_regu,
            bias_initializer=self._bias_init,
            bias_regularizer=self._bias_regu
        )
        self.built = True
    
    def call(self, x_embeddings, y_embeddings=None, **kwargs):

        if y_embeddings is None:
            y_embeddings = x_embeddings

        x = tf.concat([x_embeddings, y_embeddings], axis=1)

        if self._interacter is not None:
            x = tf.concat([
                x, self._interacter([x_embeddings, y_embeddings])], axis=1)
        
        x = self.dense_kernel(x)
        return self.dense_output(x)
    
    def get_config(self):
        config = {
            "units": self._kernel_units,
            "interacter": self._interacter,
            "use_bias": self._use_bias,
            "activation": tf.keras.activations.serialize(self._kernel_activation),
            "kernel_init": tf.keras.initializers.serialize(self._kernel_init),
            "kernel_regu": tf.keras.regularizers.serialize(self._kernel_regu),
            "bias_init": tf.keras.initializers.serialize(self._bias_init),
            "bias_regu": tf.keras.regularizers.serialize(self._bias_regu),
        }
        base_config = super(ActivationUnit, self).get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable()
class Dice(tf.keras.layers.Layer):

    def __init__(self,
                 epsilon: float = 1e-8,
                 alpha_initializer="zeros",
                 alpha_regularizer=None,
                 **kwargs):
        super(Dice, self).__init__(**kwargs)
        
        self._epsilon = epsilon
        self._alpha_initializer = alpha_initializer
        self._alpha_regularizer = alpha_regularizer

    def build(self, input_shape):

        self.prelu = tf.keras.layers.PReLU(
            alpha_initializer=self._alpha_initializer,
            alpha_regularizer=self._alpha_regularizer
        )
        self.built = True

    def call(self, inputs, **kwargs):
        
        inputs_mean = tf.math.reduce_mean(inputs, axis=1, keepdims=True)
        inputs_var = tf.math.reduce_std(inputs, axis=1, keepdims=True)

        p = tf.nn.sigmoid((inputs - inputs_mean) / (tf.sqrt(inputs_var + self._epsilon)))

        x = self.prelu(inputs)

        outputs = tf.where(x > 0, x=p * x, y=(1 - p) * x)

        return outputs

    def get_config(self):
        config = {
            "epsilon": self._epsilon,
            "alpha_initializer": tf.keras.initializers.serialize(self._alpha_initializer),
            "alpha_regularizer": tf.keras.regularizers.serialize(self._alpha_regularizer)
        }
        base_config = super(Dice, self).get_config()
        return {**base_config, **config}
