'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-30 19:37:24
@LastEditors: Wang Yao
@LastEditTime: 2020-04-02 14:22:27
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class GRU(Layer):
    
    def __init__(self, **kwargs):

        super(GRU, self).__init__(**kwargs)


    def build(self, input_shape):
        
        super(GRU, self).build(input_shape)


    def call(self, inputs):

        return 


    def compute_output_shape(self, input_shape):
        return 