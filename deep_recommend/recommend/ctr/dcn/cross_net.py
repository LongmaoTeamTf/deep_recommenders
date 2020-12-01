"""
@Description: Deep & Cross Network (DCN)
@version: https://arxiv.org/abs/1708.05123
@License: MIT
@Author: Wang Yao
@Date: 2020-08-06 18:44:25
@LastEditors: Wang Yao
@LastEditTime: 2020-12-01 21:44:22
"""
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer


class CrossLayer(Layer):

    def __init__(self, **kwargs):
        super(CrossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[0][-1], 1),
            initializer=initializers.glorot_uniform,
            trainable=True, 
            name="cross_layer_weight"
        )
        self.b = self.add_weight(
            shape=(input_shape[0][-1], ),
            initializer=initializers.zeros,
            trainable=True, 
            name="cross_layer_weight"
        )
        super(CrossLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        stack_embeddings, cross_inputs = inputs
        linear_project = tf.matmul(cross_inputs, self.W)
        feature_crossing = tf.math.multiply(stack_embeddings, linear_project)
        outputs = feature_crossing + self.b + cross_inputs # residual connect
        return outputs


class CrossNet(object):
    """ Cross Network """
    __layer_name__ = "cross_layer_"

    def __init__(self, cross_layers_num, **kwargs):
        super(CrossNet, self).__init__(**kwargs)
        self._cross_layers_num = cross_layers_num

    def __call__(self, concat_embeddings):
        """ build model """
        x = concat_embeddings
        for i in range(self._cross_layers_num):
            x = CrossLayer(name=self.__layer_name__+str(i))([concat_embeddings, x])
        return x
 