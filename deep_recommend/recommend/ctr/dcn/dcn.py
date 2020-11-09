"""
@Description: Deep Cross Network
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-08-06 18:44:25
@LastEditors: Wang Yao
@LastEditTime: 2020-11-09 14:00:07
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer


class CrossLayer(Layer):

    def __init__(self, **kwargs):
        super(CrossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[0][-1],),
            initializer=initializers.glorot_uniform,
            trainable=True, 
            name="cross_layer_weight"
        )
        self.b = self.add_weight(
            shape=(input_shape[0][-1],),
            initializer=initializers.zeros,
            trainable=True, 
            name="cross_layer_weight"
        )
        super(CrossLayer, self).build(input_shape)

    def call(self, inputs):
        stack_embeddings, cross_inputs = inputs
        linear_project = tf.math.multiply(cross_inputs, self.W) + self.b
        feature_crossing = stack_embeddings * linear_project # inner product
        outputs = feature_crossing + cross_inputs # residual connect
        return outputs

    def get_config(self):
        config = super().get_config().copy()
        return config


class CombineLayer(Layer):

    def __init__(self, **kwargs):
        super(CombineLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[0][-1]+input_shape[1][-1],),
            initializer=initializers.glorot_uniform,
            trainable=True,
            name="logits_weight"
        )
        super(CombineLayer, self).build(input_shape)

    def call(self, inputs):
        cross_net_output, dnn_output = inputs
        concat_output = tf.concat([cross_net_output, dnn_output], axis=-1)
        logits_output = tf.math.multiply(concat_output, self.W)
        return tf.math.sigmoid(logits_output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def build_dcn(n_cross_layers, n_ff_layers, ff_size=32):
    """创建DCN"""
    sparse_0 = tf.feature_column.categorical_column_with_identity(key="sparse_0", num_buckets=100)
    sparse_0 = tf.feature_column.embedding_column(sparse_0, 10)
    sparse_1 = tf.feature_column.categorical_column_with_identity(key="sparse_1", num_buckets=100)
    sparse_1 = tf.feature_column.embedding_column(sparse_1, 10)

    dense_0 = tf.feature_column.numeric_column(key="dense_0")
    dense_1 = tf.feature_column.numeric_column(key="dense_1")

    features = [sparse_0, sparse_1, dense_0, dense_1]
    features_layer = tf.keras.layers.DenseFeatures(features)

    sparse_0_input = tf.keras.Input(shape=(1,), dtype=tf.int32)
    sparse_1_input = tf.keras.Input(shape=(1,), dtype=tf.int32)
    dense_0_input = tf.keras.Input(shape=(1,))
    dense_1_input = tf.keras.Input(shape=(1,))
    inputs = [sparse_0_input, sparse_1_input, dense_0_input, dense_1_input]
    stack_embeddings = features_layer({
        "sparse_0": sparse_0_input, 
        "sparse_1": sparse_1_input, 
        "dense_0": dense_0_input, 
        "dense_1": dense_1_input
    })
    # 创建交叉网络
    cross_output = CrossLayer()([stack_embeddings, stack_embeddings])
    for i in range(n_cross_layers - 1):
        cross_output = CrossLayer(name=f"cross_layer_{i}")([stack_embeddings, cross_output])

    # 创建dnn
    ff_output = tf.keras.layers.Dense(ff_size, activation="relu")(stack_embeddings)
    for i in range(n_ff_layers):
        ff_output = tf.keras.layers.Dense(ff_size, activation="relu")(ff_output)

    outputs = CombineLayer()([cross_output, ff_output])
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='DCN')
    return model
    

if __name__ == "__main__":   
    model = build_dcn(3, 2)
    tf.keras.utils.plot_model(model, show_shapes=True, to_file="dcn.png")
   