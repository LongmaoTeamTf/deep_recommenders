#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

from deep_recommenders.estimator.models.feature_interaction import FM
from deep_recommenders.estimator.models.feature_interaction import dnn


class DeepFM(object):

    def __init__(self,
                 indicator_columns,
                 embedding_columns,
                 dnn_units,
                 dnn_activation=tf.nn.relu,
                 dnn_batch_normalization=False,
                 dnn_dropout=None,
                 **dnn_kwargs):
        self._indicator_columns = indicator_columns
        self._embedding_columns = embedding_columns
        self._dnn_hidden_units = dnn_units
        self._dnn_activation = dnn_activation
        self._dnn_batch_norm = dnn_batch_normalization
        self._dnn_dropout = dnn_dropout
        self._dnn_kwargs = dnn_kwargs

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, features):
        fm = FM(self._indicator_columns, self._embedding_columns)

        fm_outputs = fm(features)
        concat_embeddings = tf.concat(fm.embeddings, axis=1)

        dnn_outputs = dnn(concat_embeddings,
                          self._dnn_hidden_units + [1],
                          activation=self._dnn_activation,
                          batch_normalization=self._dnn_batch_norm,
                          dropout=self._dnn_dropout,
                          **self._dnn_kwargs)

        return tf.nn.sigmoid(fm_outputs + dnn_outputs)




