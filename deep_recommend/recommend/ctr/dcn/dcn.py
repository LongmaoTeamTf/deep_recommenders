"""
@Description: Deep Cross Network
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-08-06 18:44:25
@LastEditors: Wang Yao
@LastEditTime: 2020-11-10 15:28:49
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


def numeric_normalizer(value):
    """数值型特征归一化"""
    def true_fn(): return tf.math.log(value+1)
    def false_fn(): return value
    return tf.where(value > -1., true_fn(), false_fn())


def build_dcn(n_cross_layers, n_ff_layers, ff_size=32):
    """创建DCN"""
    integer_cols_names = [f"I{i}" for i in range(13)]
    categorical_cols_names = [f"C{i}" for i in range(26)]

    dense_inputs, dense_cols = [], []
    for integer_col_name in integer_cols_names:
        dense_input = tf.keras.Input(shape=(1,), name=integer_col_name)
        dense_col = tf.feature_column.numeric_column(key=integer_col_name, default_value=-1, normalizer_fn=numeric_normalizer)
        dense_inputs.append(dense_input)
        dense_cols.append(dense_col)
        
    sparse_inputs, sparse_cols = [], []
    for categorical_col_name in categorical_cols_names:
        sparse_input = tf.keras.Input(shape=(1,), name=categorical_col_name, dtype=tf.string)
        sparse_col = tf.feature_column.categorical_column_with_hash_bucket(key=categorical_col_name, hash_bucket_size=10000)
        sparse_embed_col = tf.feature_column.embedding_column(sparse_col, 10, trainable=True)
        sparse_inputs.append(sparse_input)
        sparse_cols.append(sparse_embed_col)

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
   