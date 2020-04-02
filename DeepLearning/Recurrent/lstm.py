'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-30 19:37:16
@LastEditors: Wang Yao
@LastEditTime: 2020-04-02 14:48:48
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class LSTM(Layer):
    
    def __init__(self, **kwargs):

        super(LSTM, self).__init__(**kwargs)


    def build(self, input_shape):
        
        super(LSTM, self).build(input_shape)


    def call(self, inputs):

        return 


    def compute_output_shape(self, input_shape):
        return 