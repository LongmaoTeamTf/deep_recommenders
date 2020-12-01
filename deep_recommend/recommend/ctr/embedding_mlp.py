"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-12-01 11:48:59
@LastEditors: Wang Yao
@LastEditTime: 2020-12-01 16:59:22
"""
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras import activations
from tensorflow.keras.layers import ReLU, Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.metrics import AUC


class EmbeddingMLP(object):
    """ Embedding&MLP框架 """
    __dnn_layer__ = "dnn_layer_"
    __logits_layer__ = "logits_layer"
    __embeding_layer__ = "embedding_layer"

    def __init__(self,
            dense_features_configs: dict,
            sparse_features_configs: dict,
            ff_hidden_sizes: list,
            ff_hidden_activation: str,
            ff_hidden_dropout_rates: list,
            logits_size: int,
            logits_activation: str,
            model_name: str,
            model_loss: str,
            model_optimizer: str,
            **kwargs):
        super(EmbeddingMLP, self).__init__(**kwargs)
        self._sparse_features_configs = sparse_features_configs
        self._dense_features_configs = dense_features_configs
        self._ff_hidden_sizes = ff_hidden_sizes
        self._ff_hidden_activation = ff_hidden_activation
        self._ff_hidden_dropout_rates = ff_hidden_dropout_rates
        self._logits_size = logits_size
        self._logits_activation = logits_activation
        self._model_name = model_name
        self._model_loss = model_loss
        self._model_optimizer = model_optimizer

    def __call__(self, explicit_part) -> Model:
        """ Joint explicit part & dnn """
        concat_embeddings, model_inputs = self.build_embedding_layers(
            self._dense_features_configs, self._sparse_features_configs)
        explicit_part_output = explicit_part(concat_embeddings)
        dnn_output = self.build_dnn(concat_embeddings)
        model_outputs = self.build_logits_output([explicit_part_output, dnn_output])
        model = Model(model_inputs, model_outputs, name=self._model_name)
        model.compile(
            loss=self._model_loss, 
            optimizer=self._model_optimizer, 
            metrics=[AUC()])
        return model

    def build_dnn(self, inputs: Tensor) -> Tensor:
        """ dnn: implicit part """ 
        x = inputs
        for i, (ff_hidden_size, ff_hidden_dropout_rate) in \
            enumerate(zip(self._ff_hidden_sizes, self._ff_hidden_dropout_rates)):
            x = Dense(int(ff_hidden_size), name=self.__dnn_layer__ + str(i))(x)
            x = ReLU()(x)
            x = Dropout(float(ff_hidden_dropout_rate))(x)
        return x

    def build_logits_output(self, inputs: list) -> Tensor:
        """ logits output layer """
        concat_inputs = Concatenate(axis=-1)(inputs)
        x = Dense(
            self._logits_size, 
            activation=activations.get(self._logits_activation),
            name=self.__logits_layer__
        )(concat_inputs)
        return x
    
    def build_embedding_layers(self, dense_features_configs: dict, sparse_features_configs: dict) -> Tensor:
        """ embedding layers """
        dense_inputs, dense_cols = [], []
        for dense_feature in dense_features_configs:
            dense_input = tf.keras.Input(shape=(1,), name=dense_feature.get("name"))
            dense_col = tf.feature_column.numeric_column(
                key=dense_feature.get("name"),
                default_value=dense_feature.get("default"),
                normalizer_fn=dense_feature.get("norm"))
            dense_col = tf.feature_column.bucketized_column(
                dense_col, [int(b) for b in dense_feature.get("boundaries").split(",")])
            dense_col = tf.feature_column.embedding_column(
                dense_col, dense_feature.get("embedding_dim"), trainable=True)
            dense_inputs.append(dense_input)
            dense_cols.append(dense_col)

        sparse_inputs, sparse_cols = [], []
        for sparse_feature in sparse_features_configs:
            sparse_input = tf.keras.Input(
                shape=(sparse_feature.get("input_length"),), 
                name=sparse_feature.get("name"), 
                dtype=tf.string)
            sparse_col = tf.feature_column.categorical_column_with_hash_bucket(
                key=sparse_feature.get("name"), hash_bucket_size=sparse_feature.get("hash_bucket_size"))
            sparse_embed_col = tf.feature_column.embedding_column(
                sparse_col, sparse_feature.get("embedding_dim"), trainable=True)
            sparse_inputs.append(sparse_input)
            sparse_cols.append(sparse_embed_col)
    
        features_layer = tf.keras.layers.DenseFeatures(dense_cols + sparse_cols, name=self.__embeding_layer__)
        concat_embeddings = features_layer({
            input_layer.name.split(":")[0]: input_layer
            for input_layer in dense_inputs + sparse_inputs})
        model_inputs = dense_inputs + sparse_inputs
        return concat_embeddings, model_inputs
        