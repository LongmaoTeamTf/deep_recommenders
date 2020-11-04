'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-31 17:55:21
@LastEditors: Wang Yao
@LastEditTime: 2020-04-07 10:55:29
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer



class Embedding(Layer):

    def __init__(self, vocab_size, model_dim, scale=True, trainable=True, **kwargs):
        self._vocab_size = vocab_size
        self._model_dim = model_dim
        self._scale = scale
        self._trainable = trainable
        super(Embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self._vocab_size, self._model_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="embeddings")
        super(Embedding, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        embeddings = K.gather(self.embeddings, inputs)
        if self._scale:
            embeddings *= self._model_dim ** 0.5 # Scale
        return embeddings

    def compute_output_shape(self, input_shape):

        return input_shape + (self._model_dim,)