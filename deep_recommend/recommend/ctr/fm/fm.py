"""
@Description: Factorization Machines
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-12-03 18:01:05
@LastEditors: Wang Yao
@LastEditTime: 2020-12-03 20:33:26
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC
from deep_recommend.recommend.ctr.embedding_layer import EmbeddingLayer


class FactorizationMachine(Layer):
    """ Factorization Machine """

    def __init__(self, fm_factors, **kwargs):
        super(FactorizationMachine, self).__init__(**kwargs)
        self._fm_factors = fm_factors

    def build(self, input_shape):
        self.V = self.add_weight(
            shape=(input_shape[-1], self._fm_factors),
            initializer="glorot_uniform",
            trainable=True,
            name="fm_v"
        )
        self.built = True

    def call(self, inputs, **kwargs):
        outputs = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.pow(tf.matmul(inputs, self.V), 2),
                tf.matmul(tf.pow(inputs, 2), tf.pow(self.V, 2))
            ), axis=1, keepdims=True
        )
        return outputs


class FM(object):
    """ FM explicit 2-order Model """
    __name__ = "fm"

    def __init__(self, dataset_config: dict, model_config: dict, **kwargs):
        super(FM, self).__init__(**kwargs)
        self._dataset_config = dataset_config
        self._model_config = model_config

    def __call__(self):
        """ build model """
        embedding_layer = EmbeddingLayer(
            self._dataset_config.get("features").get("sparse_features"),
            self._dataset_config.get("features").get("dense_features"),
            return_raw_features=True
        )
        _, model_inputs, raw_features = embedding_layer()
        fm_outputs = FactorizationMachine(
            self._model_config.get("fm").get("factors"), name=self.__name__)(raw_features)
        model_outputs = tf.math.sigmoid(fm_outputs, name="sigmoid")
        model = Model(model_inputs, model_outputs, name=self.__name__)
        model.compile(
            loss=self._model_config.get("model").get("loss"), 
            optimizer=self._model_config.get("model").get("optimizer"), 
            metrics=[AUC()])
        return model