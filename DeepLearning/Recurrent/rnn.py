'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-30 15:47:00
@LastEditors: Wang Yao
@LastEditTime: 2020-04-02 14:35:20
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer


class RNN(Layer):
    
    def __init__(self, 
            kernel_dim,
            activation='tanh',
            return_outputs=False, 
            return_states=False,
            use_bias=True,
            **kwargs):
        super(RNN, self).__init__(**kwargs)
        self._kernel_dim = kernel_dim
        self._activation = activations.get(activation)
        self._return_outputs = return_outputs
        self._return_states = return_states
        self._use_bias = use_bias
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.U = self.add_weight(
            shape=(input_dim, self._kernel_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_input')
        self.W = self.add_weight(
            shape=(self._kernel_dim, self._kernel_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_hidden')
        if self._use_bias:
            self.b = self.add_weight(
                shape=(self._kernel_dim),
                initializer='zeros',
                trainable=True,
                name='bias_b')
        if self._return_outputs:
            self.V = self.add_weight(
                shape=(self._kernel_dim, self._kernel_dim),
                initializer='glorot_uniform',
                trainable=True,
                name='weights_output')
            if self._use_bias:
                self.c = self.add_weight(
                shape=(self._kernel_dim),
                initializer='zeros',
                trainable=True,
                name='bias_c')
        
        super(RNN, self).build(input_shape)

    def call(self, inputs):
        h_t = K.zeros((1, self._kernel_dim))
        ots, hts = [], []
        for t in range(inputs.shape[1]):
            x_t = K.expand_dims(inputs[:, t, :], 1)
            a_t = K.dot(x_t, self.U) + K.dot(h_t, self.W)
            if self._use_bias: a_t += self.b

            if self._activation is not None:
                h_t = self._activation(a_t)
                hts.append(h_t)

            if self._return_outputs:
                o_t = K.dot(h_t, self.V)
                if self._use_bias: o_t += self.c
                o_t = K.softmax(o_t)
                ots.append(o_t)
        outputs = h_t
        if self._return_outputs:
            outputs = ots
        if self._return_states:
            outputs = hts
        if self._return_outputs and self._return_states:
            outputs = ots, hts 
        return outputs

    def compute_output_shape(self, input_shape):
        return [input_shape[:-1] + (self._kernel_dim,), 
                    input_shape[:-1] + (self._kernel_dim,)]
