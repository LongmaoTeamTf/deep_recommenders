"""
@Description: Deep Cross Network
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-08-06 18:44:25
@LastEditors: Wang Yao
@LastEditTime: 2020-11-30 16:04:29
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer

from dataset.criteo import create_feature_layers


class CrossLayer(Layer):

    def __init__(self, **kwargs):
        super(CrossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[0][-1], 1),
            initializer=initializers.glorot_uniform,
            trainable=True, 
            name="cross_layer_weight"
        )
        self.b = self.add_weight(
            shape=(input_shape[0][-1], ),
            initializer=initializers.zeros,
            trainable=True, 
            name="cross_layer_weight"
        )
        super(CrossLayer, self).build(input_shape)

    def call(self, inputs):
        stack_embeddings, cross_inputs = inputs
        linear_project = tf.matmul(cross_inputs, self.W)
        feature_crossing = tf.math.multiply(stack_embeddings, linear_project)
        outputs = feature_crossing + self.b + cross_inputs # residual connect
        return outputs


class CombineLayer(Layer):

    def __init__(self, **kwargs):
        super(CombineLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[0][-1]+input_shape[1][-1], 1),
            initializer=initializers.glorot_uniform,
            trainable=True,
            name="logits_weight"
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer=initializers.zeros,
            trainable=True,
            name="logits_bais"
        )
        super(CombineLayer, self).build(input_shape)

    def call(self, inputs):
        cross_net_output, dnn_output = inputs
        concat_output = tf.concat([cross_net_output, dnn_output], axis=-1)
        logits_output = tf.matmul(concat_output, self.W) + self.b
        return tf.math.sigmoid(logits_output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)



def build_dcn(n_cross_layers, n_ff_layers, ff_size=32):
    """创建DCN"""
    (sparse_inputs, sparse_cols), (dense_inputs, dense_cols) = create_feature_layers()
    features_layer = tf.keras.layers.DenseFeatures(dense_cols + sparse_cols)

    stack_embeddings = features_layer({
        input_layer.name.split(":")[0]: input_layer
        for input_layer in dense_inputs + sparse_inputs
    })
    # 创建交叉网络
    cross_output = CrossLayer()([stack_embeddings, stack_embeddings])
    for i in range(n_cross_layers - 1):
        cross_output = CrossLayer(name=f"cross_layer_{i}")([stack_embeddings, cross_output])

    # 创建dnn
    ff_output = tf.keras.layers.Dense(ff_size, activation="relu")(stack_embeddings)
    for i in range(n_ff_layers - 1):
        ff_output = tf.keras.layers.Dense(ff_size, activation="relu")(ff_output)

    outputs = CombineLayer()([cross_output, ff_output])
    model = tf.keras.Model(inputs=dense_inputs+sparse_inputs, outputs=outputs, name='DCN')
    return model
    

if __name__ == "__main__":   
    model = build_dcn(3, 3)
    tf.keras.utils.plot_model(model, show_shapes=True, to_file="dcn_criteo.png")
   