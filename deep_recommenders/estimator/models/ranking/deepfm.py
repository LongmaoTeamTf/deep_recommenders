#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

from deep_recommenders.estimator.models.feature_interaction import FM
from deep_recommenders.estimator.models.feature_interaction import dnn


class DeepFM(object):

    def __init__(self,
                 indicator_columns,
                 embedding_columns,
                 dnn_hidden_units):
        self._indicator_columns = indicator_columns
        self._embedding_columns = embedding_columns
        self._dnn_hidden_units = dnn_hidden_units

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, features):
        fm = FM(self._indicator_columns, self._embedding_columns)

        fm_outputs = fm(features)
        concat_embeddings = tf.concat(fm.embeddings, axis=1)

        dnn_outputs = dnn(concat_embeddings,
                          self._dnn_hidden_units,
                          output_activation=None)

        return tf.nn.sigmoid(fm_outputs + dnn_outputs)




