'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-30 19:37:24
@LastEditors: Wang Yao
@LastEditTime: 2020-04-02 16:16:43
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer


class GRU(Layer):
    
    def __init__(self, 
            units, 
            activation='tanh', 
            return_outputs=False,
            return_states=False,
            use_bias=True, 
            **kwargs):
        self._units = units
        self.activation = activations.get(activation)
        self._return_outputs = return_outputs
        self._use_bias = use_bias
        super(GRU, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self._units),
            initializer='glorot_uniform', 
            trainable=True, 
            name='W')
        self.U = self.add_weight(
            shape=(self._units, self._units),
            initializer='glorot_uniform', 
            trainable=True, 
            name='U')
        self.W_z = self.add_weight(
            shape=(input_shape[-1], self._units),
            initializer='glorot_uniform', 
            trainable=True, 
            name='W_z')
        self.U_z = self.add_weight(
            shape=(self._units, self._units),
            initializer='glorot_uniform', 
            trainable=True, 
            name='U_z')
        self.W_r = self.add_weight(
            shape=(input_shape[-1], self._units),
            initializer='glorot_uniform', 
            trainable=True, 
            name='W_r')
        self.U_r = self.add_weight(
            shape=(self._units, self._units),
            initializer='glorot_uniform', 
            trainable=True, 
            name='U_r')
        if self._use_bias:
            self.b = self.add_weight(
                shape=(self._units),
                initializer='zeros', 
                trainable=True, 
                name='bais')
            self.b_z = self.add_weight(
                shape=(self._units),
                initializer='zeros', 
                trainable=True, 
                name='bais_z')
            self.b_r = self.add_weight(
                shape=(self._units),
                initializer='zeros', 
                trainable=True, 
                name='bais_r')
        super(GRU, self).build(input_shape)

    def call(self, inputs):
        h_t = K.zeros(shape=(1, self._units))
        states = []
        max_seq_len = K.shape(inputs)[-2] 
        for t in range(max_seq_len):
            x_t = K.expand_dims(inputs[:, t, :], 1)
            z_t = K.dot(self.W_z, x_t) + K.dot(self.U_z, h_t)
            r_t = K.dot(self.W_r, x_t) + K.dot(self.U_r, h_t)
            if self._use_bias:
                z_t += self.b_z
                r_t += self.b_r
            z_t = K.sigmoid(z_t)
            r_t = K.sigmoid(r_t)
            h_t_ = K.dot(self.W, x_t) + K.dot(self.U, h_t)
            if self._use_bias:
                h_t_ += self.b
            if self._activation is not None:
                h_t_ = self.activation(h_t_)
            h_t = (1 - z_t) * h_t + z_t * h_t_
            states.append(h_t)
        outputs = h_t
        if self._return_outputs:
            outputs = states, h_t
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0] + (1, self._units,)
        if self._return_outputs:
            output_shape = [
                input_shape[:-1] + (self._units,),
                input_shape[0] + (1, self._units,)]
        return output_shape