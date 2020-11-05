'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-04-02 14:19:14
@LastEditors: Wang Yao
@LastEditTime: 2020-04-02 14:20:01
'''
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class Attention(Layer):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attetion = self.add_weight(
            shape=(input_shape[1],),
            initializer='glorot_uniform',
            trainable=True,
            name='attetion')
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        attetion = K.softmax(self.attetion)
        outputs = K.dot(attetion, inputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
