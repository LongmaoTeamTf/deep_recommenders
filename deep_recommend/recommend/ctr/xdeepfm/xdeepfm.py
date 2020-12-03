"""
@Description: eXtreme Deep Factorization Machine (xDeepFM)
@version: https://arxiv.org/abs/1803.05170
@License: MIT
@Author: Wang Yao
@Date: 2020-11-28 11:16:54
@LastEditors: Wang Yao
@LastEditTime: 2020-12-03 20:11:33
"""
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, Concatenate
from deep_recommend.recommend.ctr.embedding_mlp import EmbeddingMLP
from deep_recommend.recommend.ctr.embedding_layer import EmbeddingLayer


class CinBlock(Layer):
    """ Compressed Interaction Layer """
    def __init__(self, num_feature_map, **kwargs):
        super(CinBlock, self).__init__(**kwargs)
        self._num_feature_map = num_feature_map
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(1, input_shape[0][1] * input_shape[-1][1], self._num_feature_map,), 
            initializer=initializers.glorot_uniform,
            regularizer=None,
            trainable=True,
            name="filter")
        self.built = True
        
    def call(self, inputs, **kwargs):
        xk, x0 = inputs
        h, m, d = xk.shape[1], x0.shape[1], x0.shape[-1]
        split_x0 = tf.split(x0, d, axis=-1)
        split_xk = tf.split(xk, d, axis=-1)
        outer = tf.matmul(split_x0, split_xk, transpose_b=True)
        outer = tf.reshape(outer, shape=[d, -1, m * h])
        outer = tf.transpose(outer, perm=[1, 0, 2])
        filter_out = tf.nn.conv1d(outer, self.W, stride=1, padding="VALID")
        outputs = tf.nn.sigmoid(filter_out)
        outputs = tf.transpose(outputs, perm=[0, 2, 1])
        return outputs


class CIN(object):
    """ build Compressed Intercation Network """
    __layer_name__ = "cin_"

    def __init__(self, feature_maps: list, feature_embedding_dim: int, **kwargs):
        super(CIN, self).__init__(**kwargs)
        self._feature_maps = feature_maps
        self._feature_embedding_dim = feature_embedding_dim

    def __call__(self, concat_embeddings):
        """ build model """
        features_num = concat_embeddings.shape[-1]//self._feature_embedding_dim
        x0 = tf.reshape(concat_embeddings, (-1, features_num, self._feature_embedding_dim))
        x = x0
        cin_sumpoolings = []
        for i, h in enumerate(self._feature_maps):
            x = CinBlock(int(h), name=self.__layer_name__ + str(i))([x, x0])
            x_sumpooling = self.sumpooling(x)
            cin_sumpoolings.append(x_sumpooling)
        return Concatenate(axis=-1)(cin_sumpoolings)

    def sumpooling(self, inputs):
        """ Sum pooling through feature maps """
        return tf.reduce_sum(inputs, axis=-1)


class xDeepFM(object):
    """ eXtreme Deep Factorization Machine """
    
    def __init__(self, dataset_config: dict, model_config: dict, **kwargs):
        super(xDeepFM, self).__init__(**kwargs)
        self._dataset_config = dataset_config
        self._model_config = model_config

    def __call__(self):
        embedding_layer = EmbeddingLayer(
            self._dataset_config.get("features").get("sparse_features"),
            self._dataset_config.get("features").get("dense_features"),
            return_raw_features=True
        )
        embedding_mlp = EmbeddingMLP(
            self._model_config.get("ff").get("hidden_sizes").split(","),
            self._model_config.get("ff").get("hidden_activation"),
            self._model_config.get("ff").get("hidden_dropout_rates").split(","),
            self._model_config.get("logits").get("size"),
            self._model_config.get("logits").get("activation"),
            self._model_config.get("model").get("name"),
            self._model_config.get("model").get("loss"),
            self._model_config.get("model").get("optimizer"),
            need_raw_features=True
        )
        return embedding_mlp(CIN(
                self._model_config.get("cin").get("feature_maps").split(","),
                self._model_config.get("cin").get("feature_embedding_dim")),
                embedding_layer)
    
