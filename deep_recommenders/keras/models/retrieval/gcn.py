#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class GCN(tf.keras.layers.Layer):

    def __init__(self,
                 units: int,
                 residual=False,
                 use_bias=False,
                 activation="relu",
                 kernel_initializer="truncated_normal",
                 kernel_regularizer=None,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self._units = units
        self._residual = residual
        self._use_bias = use_bias
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._kernel_activation = tf.keras.activations.get(activation)

    def build(self, input_shape):

        self._kernel = tf.keras.layers.Dense(
            self._units,
            activation=self._kernel_activation,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_initializer=self._bias_initializer,
            bias_regularizer=self._bias_regularizer,
            use_bias=self._use_bias
        )
        self.built = True

    def call(self, features, adj, **kwargs):

        if isinstance(adj, tf.SparseTensor):
            agg_embeddings = tf.sparse.sparse_dense_matmul(adj, features)
        else:
            agg_embeddings = tf.linalg.matmul(adj, features)

        outputs = self._kernel(agg_embeddings)

        if self._residual is True:
            outputs += features

        return outputs

    def get_config(self):
        config = {
            "units": self._units,
            "use_bias": self._use_bias,
            "activation": tf.keras.activations.serialize(self._kernel_activation),
            "kernel_initializer": tf.keras.initializers.serialize(self._kernel_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(self._kernel_regularizer),
            "bias_initializer": tf.keras.initializers.serialize(self._bias_initializer),
            "bias_regularizer": tf.keras.regularizers.serialize(self._bias_regularizer),
        }
        base_config = super(GCN, self).get_config()
        return {**base_config, **config}
