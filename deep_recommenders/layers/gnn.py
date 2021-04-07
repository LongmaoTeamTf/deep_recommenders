"""
@Description: Graph Convolution Network in Semi-Supervised Classification with Graph Convolutional Networks
@version: https://arxiv.org/abs/1609.02907
@License: MIT
@Author: Wang Yao
@Date: 2021-04-01 14:57:43
@LastEditors: Wang Yao
@LastEditTime: 2021-04-06 19:05:36
"""
from typing import Optional, Union, Text

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class GCN(tf.keras.layers.Layer):
    """Graph Convolution Network"""
    
    def __init__(self,
                 units: int, 
                 use_bias: bool = False,
                 activation: Optional[Union[Text, None, tf.keras.layers.Layer]] = "relu", 
                 kernel_init: Union[Text, tf.keras.initializers.Initializer] = "truncated_normal",
                 kernel_regu: Union[Text, None, tf.keras.regularizers.Regularizer] = None,
                 bias_init: Union[Text, tf.keras.initializers.Initializer] = "zeros",
                 bias_regu: Union[Text, None, tf.keras.regularizers.Regularizer] = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self._units = units
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
        last_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(last_dim, self._units),
            initializer=self._kernel_init,
            regularizer=self._kernel_regu,
            name="kernel"
        )

        if self._use_bias:
            self.bias = self.add_weight(
                shape=(self._units,),
                initializer=self._bias_init,
                regularizer=self._bias_regu,
                name="bias"
            )
        self.built = True

    def call(self, embeddings, adj):

        if isinstance(adj, tf.SparseTensor):
            agg_embeddings = tf.sparse.sparse_dense_matmul(adj, embeddings)
        else:
            agg_embeddings = tf.linalg.matmul(adj, embeddings)
        
        outputs = tf.linalg.matmul(agg_embeddings, self.kernel)
        if self._use_bias is True:
            outputs = outputs + self.bias
            
        if self._kernel_activation is not None:
            outputs = self._kernel_activation(outputs)

        return outputs

    def get_config(self):
        config = {
            "units": self._units,
            "use_bias": self._use_bias,
            "activation":
                tf.keras.activations.serialize(self._kernel_activation),
            "kernel_init":
                tf.keras.initializers.serialize(self._kernel_init),
            "kernel_regu":
                tf.keras.regularizers.serialize(self._kernel_regu),
            "bias_init":
                tf.keras.initializers.serialize(self._bias_init),
            "bias_regu":
                tf.keras.regularizers.serialize(self._bias_regu),
        }
        base_config = super(GCN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
