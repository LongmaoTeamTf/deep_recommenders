"""
@Description: Deep Cross Network
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-08-06 18:44:25
@LastEditors: Wang Yao
@LastEditTime: 2020-08-06 19:18:25
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer


class CrossNetwork(Layer):

    def __init__(self, **kwargs):
        super(CrossNetwork, self).__init__(**kwargs)
        

    def build(self, input_shape):

        super(CrossNetwork, self).build(input_shape)


    def call(self, inputs):
        return 

