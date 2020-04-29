'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-04-07 10:51:06
@LastEditors: Wang Yao
@LastEditTime: 2020-04-07 11:11:36
'''
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow.keras import regularizers


class OneOrder(Layer):

    def __init__(self, sparse_values_size, **kwargs):
        self._sparse_values_size = sparse_values_size
        super(OneOrder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.n_sparse = len(input_shape[0])
        self.n_denses = len(input_shape[1])
        self.sparse_weights = []
        for i in range(self.n_sparse):
            self.sparse_weights.append(self.add_weight(
                shape=(self._sparse_values_size[i], 1),
                initializer="glorot_uniform",
                trainable=True,
                name=f'sparse_weights_{i}'))
        self.dense_weights = self.add_weight(
            shape=(self.n_denses, 1),
            initializer="glorot_uniform",
            trainable=True,
            name='dense_weights')
        super(OneOrder, self).build(input_shape)

    def call(self, inputs):
        sparse_inputs, dense_inputs = inputs
        outputs = 0
        for i in range(self.n_sparse):
            sparse_w = K.gather(self.sparse_weights[i], sparse_inputs[i])
            outputs += sparse_w[:, 0, :]
        dense_inputs = K.concatenate(dense_inputs)
        outputs += K.dot(dense_inputs, self.dense_weights)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0][0], 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sparse_values_size': self._sparse_values_size})
        return config


class EmbeddingLayer(Layer):

    def __init__(self, sparse_values_size, embedding_dim, **kwargs):
        self._sparse_values_size = sparse_values_size
        self._embedding_dim = embedding_dim
        super(EmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.n_sparse = len(input_shape[0])
        self.n_denses = len(input_shape[1])
        self.sparse_weights = []
        for i in range(self.n_sparse):
            self.sparse_weights.append(self.add_weight(
                shape=(self._sparse_values_size[i], self._embedding_dim),
                initializer="glorot_uniform",
                regularizer=regularizers.l2(0.5),
                trainable=True,
                name=f'sparse_weights_{i}'))
        self.dense_weights = []
        for i in range(self.n_denses):
            self.dense_weights.append(self.add_weight(
                shape=(1, self._embedding_dim),
                initializer="glorot_uniform",
                regularizer=regularizers.l2(0.5),
                trainable=True,
                name=f'dense_weights_{i}'))
        super(EmbeddingLayer, self).build(input_shape)

    def call(self, inputs):
        sparse_inputs, dense_inputs = inputs
        embeddings = []
        for i in range(self.n_sparse):
            sparse_w = K.gather(self.sparse_weights[i], sparse_inputs[i])
            embeddings.append(sparse_w[:, 0, :])
        for i in range(self.n_denses):
            embeddings.append(K.dot(dense_inputs[i], self.dense_weights[i]))
        return embeddings

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sparse_values_size': self._sparse_values_size,
            'embedding_dim': self._embedding_dim})
        return config


class TwoOrder(Layer):

    def __init__(self, **kwargs):
        super(TwoOrder, self).__init__(**kwargs)

    # def build(self, input_shape):
    #     self.output_weight = self.add_weight(
    #         shape=(input_shape[0][-1], 1),
    #         initializer="glorot_uniform",
    #         trainable=True,
    #         name='output_weight')
    #     super(TwoOrder, self).build(input_shape)

    def call(self, inputs):
        exp_inputs = [K.expand_dims(x, axis=1) for x in inputs]
        cocat_inputs = K.concatenate(exp_inputs, axis=1)
        square_inputs = K.square(K.sum(cocat_inputs, axis=1))        
        sum_inputs = K.sum(K.square(cocat_inputs), axis=1)
        cross_term = square_inputs - sum_inputs
        outputs = 0.5 * K.sum(cross_term, axis=1, keepdims=True)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


class HighOrder(Layer):

    def __init__(self, n_layers, dropout_rate=0.5, activation='relu', **kwargs):
        self._n_layers = n_layers
        self._dropout_rate = dropout_rate
        self._activation = activations.get(activation)
        super(HighOrder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.n_fields = len(input_shape)
        self.kernel_size = input_shape[0][-1] * self.n_fields
        self.hidden_weights = []
        for i in range(self._n_layers):
            self.weights.append(self.add_weight(
                shape=(self.kernel_size, self.kernel_size),
                initializer="glorot_uniform",
                trainable=True,
                name=f'hidden_weight_{i}'))
        self.output_weight = self.add_weight(
            shape=(self.kernel_size, 1),
            initializer="glorot_uniform",
            trainable=True,
            name='output_weight')
        super(HighOrder, self).build(input_shape)

    def call(self, inputs):
        outputs = K.concatenate(inputs, axis=1)
        for i in range(self._n_layers):
            outputs = K.dot(outputs, self.weights[i])
            outputs = self._activation(outputs)
            outputs = K.dropout(outputs, self._dropout_rate)
        outputs = K.dot(outputs, self.output_weight)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_layers': self._n_layers,
            'dropout_rate': self._dropout_rate,
            'activation': self._activation})
        return config


class LR(Layer):

    def __init__(self, **kwargs):
        super(LR, self).__init__(**kwargs)

    # def build(self, input_shape):
    #     self.w = self.add_weight(
    #         shape=(3, 1),
    #         initializer="glorot_uniform",
    #         trainable=True,
    #         name='w')
    #     super(LR, self).build(input_shape)

    def call(self, inputs):
        outputs = K.concatenate(inputs, axis=1)
        outputs = K.sum(outputs, axis=1, keepdims=True)
        outputs = K.sigmoid(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)