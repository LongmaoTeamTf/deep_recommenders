"""
@Description: eXtreme Deep Factorization Machine (xDeepFM)
@version: https://arxiv.org/abs/1803.05170
@License: MIT
@Author: Wang Yao
@Date: 2020-11-28 11:16:54
@LastEditors: Wang Yao
@LastEditTime: 2020-11-30 19:52:13
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from dataset.criteo import create_feature_layers


class CompressedInteraction(Layer):
    """ Compressed Interaction Layer """
    def __init__(self, num_feature_map, **kwargs):
        self._num_feature_map = num_feature_map
        super(CompressedInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(
                1,
                input_shape[0][1] * input_shape[-1][1],
                self._num_feature_map, 
            ), 
            initializer=initializers.glorot_uniform,
            regularizer=None,
            trainable=True,
            name="filter"
        )
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
        

class SumPooling(Layer):

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=-1)


def build_cin(x0, feature_maps=[3, 3, 3]):
    """ 构建CIN """
    x = x0
    cin_sumpoolings = []
    for h in feature_maps:
        x = CompressedInteraction(h)([x, x0])
        x_sumpooling = SumPooling()(x)
        cin_sumpoolings.append(x_sumpooling)
    return tf.keras.layers.Concatenate(axis=-1)(cin_sumpoolings)


def build_xdeepfm():
    """ 构建xdeepfm模型 """
    (sparse_inputs, sparse_cols), (dense_inputs, dense_cols) = create_feature_layers()
    
    features_layer = tf.keras.layers.DenseFeatures(dense_cols + sparse_cols)

    concat_embeddings = features_layer({
        input_layer.name.split(":")[0]: input_layer
        for input_layer in dense_inputs + sparse_inputs
    })

    stack_embeddings = tf.reshape(concat_embeddings, (-1, concat_embeddings.shape[-1]//10, 10))

    cin_output = build_cin(stack_embeddings, feature_maps=[3, 4, 5])

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(cin_output)

    model = tf.keras.Model(sparse_inputs+dense_inputs, outputs)
    return model
    
    