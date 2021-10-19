#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

if tf.__version__ >= "2.0.0":
    import tensorflow.compat.v1 as tf

from deep_recommenders.estimator.models.feature_interaction import dnn


class ESMM(object):

    def __init__(self,
                 feature_columns,
                 hidden_units,
                 activation=tf.nn.relu,
                 batch_normalization=False,
                 dropout=None,
                 **kwargs):
        self._columns = feature_columns
        self._hidden_units = hidden_units
        self._activation = activation
        self._batch_norm = batch_normalization
        self._dropout = dropout
        self._configs = kwargs

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, features):

        dnn_inputs = tf.feature_column.input_layer(features, self._columns)

        with tf.variable_scope("pCVR"):
            cvr = dnn(dnn_inputs,
                      self._hidden_units + [1],
                      activation=self._activation,
                      batch_normalization=self._batch_norm,
                      dropout=self._dropout,
                      **self._configs)
            p_cvr = tf.nn.sigmoid(cvr)

        with tf.variable_scope("pCTR"):
            ctr = dnn(dnn_inputs,
                      self._hidden_units + [1],
                      activation=self._activation,
                      batch_normalization=self._batch_norm,
                      dropout=self._dropout,
                      **self._configs)
            p_ctr = tf.nn.sigmoid(ctr)

        p_ctcvr = tf.math.multiply(p_ctr, p_cvr)

        return p_cvr, p_ctr, p_ctcvr
