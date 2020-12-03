"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-12-03 18:49:03
@LastEditors: Wang Yao
@LastEditTime: 2020-12-03 20:07:39
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer



class EmbeddingLayer(object):
    """ Embedding Layer """

    __raw_layer__ = "raw_layer"
    __embeding_layer__ = "embedding_layer"

    def __init__(self, 
                 sparse_features_configs: dict, 
                 dense_features_configs: dict,
                 return_raw_features: bool,
                 **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self._sparse_features_configs = sparse_features_configs
        self._dense_features_configs = dense_features_configs
        self._return_raw_features = return_raw_features
        
    def __call__(self) -> Layer:
        """ build embedding layers """
        sparse_inputs, sparse_cols, sparse_raw_features = self.build_sparse_embedding_layers()
        dense_inputs, dense_cols, dense_raw_features = self.build_dense_embedding_layers()
        features_layer = tf.keras.layers.DenseFeatures(dense_cols + sparse_cols, name=self.__embeding_layer__)
        concat_embeddings = features_layer({
            input_layer.name.split(":")[0]: input_layer
            for input_layer in dense_inputs + sparse_inputs})
        model_inputs = dense_inputs + sparse_inputs

        if self._return_raw_features is True:
            raw_features_layer = tf.keras.layers.DenseFeatures(
                sparse_raw_features + dense_raw_features, name=self.__raw_layer__)
            raw_features = raw_features_layer({
                input_layer.name.split(":")[0]: input_layer
                for input_layer in dense_inputs + sparse_inputs})
            return concat_embeddings, model_inputs, raw_features
        return concat_embeddings, model_inputs

    def build_sparse_embedding_layers(self):
        """ build sparse features embedding layers """
        sparse_raw_features = []
        sparse_inputs, sparse_cols = [], []
        for sparse_feature in self._sparse_features_configs:
            sparse_input = tf.keras.Input(
                shape=(sparse_feature.get("input_length"),), 
                name=sparse_feature.get("name"), 
                dtype=tf.string)
            sparse_col = tf.feature_column.categorical_column_with_hash_bucket(
                key=sparse_feature.get("name"), hash_bucket_size=sparse_feature.get("hash_bucket_size"))
            if self._return_raw_features is True:
                sparse_raw_features.append(tf.feature_column.indicator_column(sparse_col))
            sparse_embed_col = tf.feature_column.embedding_column(
                sparse_col, sparse_feature.get("embedding_dim"), trainable=True,)
            sparse_inputs.append(sparse_input)
            sparse_cols.append(sparse_embed_col)
        return sparse_inputs, sparse_cols, sparse_raw_features

    def build_dense_embedding_layers(self):
        """ build dense features embedding layers """
        dense_raw_features = []
        dense_inputs, dense_cols = [], []
        for dense_feature in self._dense_features_configs:
            dense_input = tf.keras.Input(shape=(1,), name=dense_feature.get("name"))
            dense_col = tf.feature_column.numeric_column(
                key=dense_feature.get("name"),
                default_value=dense_feature.get("default"),
                normalizer_fn=dense_feature.get("norm"))
            dense_col = tf.feature_column.bucketized_column(
                dense_col, [int(b) for b in dense_feature.get("boundaries").split(",")])
            if self._return_raw_features is True:
                dense_raw_features.append(tf.feature_column.indicator_column(dense_col))
            dense_embed_col = tf.feature_column.embedding_column(
                dense_col, dense_feature.get("embedding_dim"), trainable=True)
            dense_inputs.append(dense_input)
            dense_cols.append(dense_embed_col)
        return dense_inputs, dense_cols, dense_raw_features
