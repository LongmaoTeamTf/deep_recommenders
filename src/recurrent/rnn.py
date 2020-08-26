'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-30 15:47:00
@LastEditors: Wang Yao
@LastEditTime: 2020-04-02 16:27:06
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer


class RNN(Layer):
    
    def __init__(self, 
            units,
            activation='tanh',
            return_outputs=False, 
            use_bias=True,
            **kwargs):
        super(RNN, self).__init__(**kwargs)
        self._units = units
        self._activation = activations.get(activation)
        self._return_outputs = return_outputs
        self._use_bias = use_bias
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(
            shape=(input_dim, self._units),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_input')
        self.U = self.add_weight(
            shape=(self._units, self._units),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_hidden')
        if self._use_bias:
            self.b = self.add_weight(
                shape=(self._units),
                initializer='zeros',
                trainable=True,
                name='bias_b')
        super(RNN, self).build(input_shape)

    def call(self, inputs):
        h_t = K.zeros((1, self._units))
        states = []
        for t in range(inputs.shape[1]):
            x_t = K.expand_dims(inputs[:, t, :], 1)
            a_t = K.dot(x_t, self.W) + K.dot(h_t, self.U)
            if self._use_bias: 
                a_t += self.b
            if self._activation is not None:
                h_t = self._activation(a_t)
                states.append(h_t)
        outputs = h_t
        if self._return_outputs:
            states = K.concatenate(states, axis=-2)
            outputs = states, h_t
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0] + (1, self._units,)
        if self._return_outputs:
            output_shape = [
                input_shape[:-1] + (self._units,),
                input_shape[0] + (1, self._units,)]
        return output_shape
