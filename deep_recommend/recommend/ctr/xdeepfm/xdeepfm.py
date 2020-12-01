"""
@Description: eXtreme Deep Factorization Machine (xDeepFM)
@version: https://arxiv.org/abs/1803.05170
@License: MIT
@Author: Wang Yao
@Date: 2020-11-28 11:16:54
@LastEditors: Wang Yao
@LastEditTime: 2020-12-01 17:39:35
"""
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, Concatenate


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
        
    def call(self, inputs):
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
        for h in self._feature_maps:
            x = CinBlock(int(h))([x, x0])
            x_sumpooling = self.sumpooling(x)
            cin_sumpoolings.append(x_sumpooling)
        return Concatenate(axis=-1)(cin_sumpoolings)

    def sumpooling(self, inputs):
        """ Sum pooling through feature maps """
        return tf.reduce_sum(inputs, axis=-1)

    