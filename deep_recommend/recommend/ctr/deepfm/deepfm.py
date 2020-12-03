"""
@Description: Deep & FM Network
@version: https://arxiv.org/abs/1703.04247
@License: MIT
@Author: Wang Yao
@Date: 2020-12-01 18:57:30
@LastEditors: Wang Yao
@LastEditTime: 2020-12-03 11:40:52
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from deep_recommend.recommend.ctr.embedding_mlp import EmbeddingMLP


class FM(Layer):
    """ Factorization Machines """

    def __init__(self, fm_factors, **kwargs):
        super(FM, self).__init__(**kwargs)
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


class FmPart(object):
    """ FM explicit part """
    __layer__ = "fm"

    def __init__(self, fm_factors, **kwargs):
        super(FmPart, self).__init__(**kwargs)
        self._fm_factors = fm_factors

    def __call__(self, concat_embeddings):
        """ build model """
        return FM(self._fm_factors, name=self.__layer__)(concat_embeddings)


class DeepFM(object):
    """ Deep & Factorization Machine """
    
    def __init__(self, dataset_config: dict, model_config: dict, **kwargs):
        super(DeepFM, self).__init__(**kwargs)
        self._dataset_config = dataset_config
        self._model_config = model_config

    def __call__(self):
        embedding_mlp = EmbeddingMLP(
            self._dataset_config.get("features").get("dense_features"),
            self._dataset_config.get("features").get("sparse_features"),
            self._model_config.get("ff").get("hidden_sizes").split(","),
            self._model_config.get("ff").get("hidden_activation"),
            self._model_config.get("ff").get("hidden_dropout_rates").split(","),
            self._model_config.get("logits").get("size"),
            self._model_config.get("logits").get("activation"),
            self._model_config.get("model").get("name"),
            self._model_config.get("model").get("loss"),
            self._model_config.get("model").get("optimizer")
        )
        return embedding_mlp(FmPart(self._model_config.get("fm").get("factors")))