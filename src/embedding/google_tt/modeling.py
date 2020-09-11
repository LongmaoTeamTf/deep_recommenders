"""
@Description: Sampling-Bias-Corrected Neural Retrieval Model
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-08-27 17:22:16
@LastEditors: Wang Yao
@LastEditTime: 2020-09-11 19:11:41
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.keras import regularizers


class HashEmbeddings(Layer):

    def __init__(self, 
                 hash_bucket_size, 
                 embedding_dim,
                 mean=False,
                 regularizer='l2',
                 initializer='he_uniform',
                 trainable=True,
                 **kwargs):
        super(HashEmbeddings, self).__init__(**kwargs)
        self._hash_bucket_size = hash_bucket_size
        self._embedding_dim = embedding_dim
        self._mean = mean
        self._regularizer = regularizers.get(regularizer)
        self._initializer = initializers.get(initializer)
        self._trainable = trainable

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self._hash_bucket_size, self._embedding_dim),
            regularizer=self._regularizer,
            initializer=self._initializer,
            trainable=self._trainable,
            name='embeddings'
        )
        super(HashEmbeddings, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        outputs = K.dot(inputs, self.embeddings)
        if self._mean is True:
            condition = tf.greater(tf.reduce_sum(inputs, axis=-1, keepdims=True), 0)
            outputs = tf.where(condition,
                outputs / tf.tile(tf.reduce_sum(inputs, axis=-1, keepdims=True), (1, self._embedding_dim)),
                K.zeros_like(outputs, dtype=tf.float32))
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self._embedding_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hash_bucket_size': self._hash_bucket_size,
            'embedding_dim': self._embedding_dim,
            'regularizer': self._regularizer,
            'initializer': self._initializer,
            'trainable': self._trainable
        })
        return config


class L2Normalization(Layer):

    def call(self, inputs, **kwargs):
        return K.l2_normalize(inputs)


def _log_norm(value):
    """对数标准化"""
    def _true_fn(val):
        return tf.math.log(tf.cast(val, tf.float32) + 1)
    
    def _false_fn(val):
        return tf.cast(val, 'float32')

    value = tf.where(value >= 0, _true_fn(value), _false_fn(value))
    
    return tf.reshape(value, [-1, 1])


def _time_exp_norm(value, time_dancy=3.0):
    """时间间隔类型特征指数标准化"""
    
    def _true_fn(val):
        val = val / (60 * 60 * 24 * 365)
        return 1 / tf.math.pow(time_dancy, tf.cast(val, tf.float32))
    
    def _false_fn(val):
        return tf.cast(val, 'float32')
    
    value = tf.where(value >= 0, _true_fn(value), _false_fn(value))

    return tf.reshape(value, [-1, 1])


def build_model():
    """构建双塔模型"""
    _video_ids_hash_bucket_size = 100000
    _video_categories_hash_bucket_size = 20
    _video_tags_hash_bucket_size = 1000
    _video_ids_embedding_dim = 128
    _video_categories_embedding_dim = 32
    _video_tags_embedding_dim = 64
    _max_tags_num = 5
    _past_watches_num = 30

    video_ids_hash = tf.feature_column.categorical_column_with_hash_bucket(
            key='video_ids', hash_bucket_size=_video_ids_hash_bucket_size, dtype=tf.string)
    video_ids_indicator = tf.feature_column.indicator_column(video_ids_hash)
    video_ids_dense = tf.keras.layers.DenseFeatures(video_ids_indicator, trainable=True, name="video_ids")
    seed_video_id_input = tf.keras.layers.Input(shape=(1,), name='seed_id', dtype=tf.string)
    cand_video_id_input = tf.keras.layers.Input(shape=(1,), name='cand_id', dtype=tf.string)
    seed_video_id = video_ids_dense({'video_ids': seed_video_id_input})
    cand_video_id = video_ids_dense({'video_ids': cand_video_id_input})

    video_categories_hash = tf.feature_column.categorical_column_with_hash_bucket(
            key='video_categories', hash_bucket_size=_video_categories_hash_bucket_size, dtype=tf.string)
    video_categories_indicator = tf.feature_column.indicator_column(video_categories_hash)
    video_categories_dense = tf.keras.layers.DenseFeatures(video_categories_indicator, trainable=True, name='video_categories')
    seed_video_category_input = tf.keras.layers.Input(shape=(1,), name='seed_category', dtype=tf.string)
    cand_video_category_input = tf.keras.layers.Input(shape=(1,), name='cand_category', dtype=tf.string)
    seed_video_category = video_categories_dense({'video_categories': seed_video_category_input})
    cand_video_category = video_categories_dense({'video_categories': cand_video_category_input})

    video_tags_hash = tf.feature_column.categorical_column_with_hash_bucket(
            key='video_tags', hash_bucket_size=_video_tags_hash_bucket_size, dtype=tf.string)
    video_tags_indicator = tf.feature_column.indicator_column(video_tags_hash)
    video_tags_dense = tf.keras.layers.DenseFeatures(video_tags_indicator, trainable=True, name='video_tags')
    seed_video_tags_input = tf.keras.layers.Input(shape=(_max_tags_num,), name='seed_tags', dtype=tf.string)
    cand_video_tags_input = tf.keras.layers.Input(shape=(_max_tags_num,), name='cand_tags', dtype=tf.string)
    seed_video_tags = video_tags_dense({'video_tags': seed_video_tags_input})
    cand_video_tags = video_tags_dense({'video_tags': cand_video_tags_input})

    video_gap_time_num = tf.feature_column.numeric_column(
        key='video_gap_time', default_value=-1, dtype=tf.int32, normalizer_fn=_time_exp_norm)
    video_gap_time_dense = tf.keras.layers.DenseFeatures(video_gap_time_num, trainable=True, name='video_gap_time')
    seed_video_gap_time_input = tf.keras.layers.Input(shape=(1,), name='seed_gap_time')
    cand_video_gap_time_input = tf.keras.layers.Input(shape=(1,), name='cand_gap_time')
    seed_video_gap_time = video_gap_time_dense({'video_gap_time': seed_video_gap_time_input})
    cand_video_gap_time = video_gap_time_dense({'video_gap_time': cand_video_gap_time_input})

    video_duration_time = tf.feature_column.numeric_column(
        key='video_duration_time', default_value=-1, dtype=tf.int32, normalizer_fn=_log_norm)
    video_duration_time_dense = tf.keras.layers.DenseFeatures(video_duration_time, trainable=True, name='video_duration_time')
    seed_video_duration_time_input = tf.keras.layers.Input(shape=(1,), name='seed_duration_time')
    cand_video_duration_time_input = tf.keras.layers.Input(shape=(1,), name='cand_duration_time')
    seed_video_duration_time = video_duration_time_dense({'video_duration_time': seed_video_duration_time_input})
    cand_video_duration_time = video_duration_time_dense({'video_duration_time': cand_video_duration_time_input})

    video_seed_numerical_inputs = {}
    video_cand_numerical_inputs = {}
    video_seed_numerical_features = {}
    video_cand_numerical_features = {}
    for feat in ['play_count', 'like_count', 'collect_count', 'share_count']:
        feat_num = tf.feature_column.numeric_column(
            key='video_'+feat, default_value=-1, dtype=tf.int32, normalizer_fn=_log_norm)
        feat_dense = tf.keras.layers.DenseFeatures(feat_num, trainable=True, name='video_'+feat)
        seed_feat_input = tf.keras.layers.Input(shape=(1,), name='seed_'+feat)
        cand_feat_input = tf.keras.layers.Input(shape=(1,), name='cand_'+feat)
        video_seed_numerical_inputs['seed_'+feat] = seed_feat_input
        video_cand_numerical_inputs['cand_'+feat] = cand_feat_input
        video_seed_numerical_features['seed_'+feat] = feat_dense({'video_'+feat: seed_feat_input})
        video_cand_numerical_features['cand_'+feat] = feat_dense({'video_'+feat: cand_feat_input})

    past_watches_input = tf.keras.layers.Input(shape=(_past_watches_num,), dtype=tf.string, name='past_watches')
    user_past_watches = video_ids_dense({'video_ids': past_watches_input})

    video_ids_hash_embeddings = HashEmbeddings(
        _video_ids_hash_bucket_size, _video_ids_embedding_dim, name='video_ids_hash_embeddings')
    video_categories_hash_embeddings = HashEmbeddings(
        _video_categories_hash_bucket_size, _video_categories_embedding_dim, name='video_categories_hash_embeddings')
    video_tags_hash_embeddings = HashEmbeddings(
        _video_tags_hash_bucket_size, _video_tags_embedding_dim, mean=True, name='video_tags_hash_embeddings')

    seed_video_id_embeddings = video_ids_hash_embeddings(seed_video_id)
    cand_video_id_embeddings = video_ids_hash_embeddings(cand_video_id)
    user_past_watches_embeddings = video_ids_hash_embeddings(user_past_watches)

    seed_video_category_embeddings = video_categories_hash_embeddings(seed_video_category)
    cand_video_category_embeddings = video_categories_hash_embeddings(cand_video_category)

    seed_video_tags_embeddings = video_tags_hash_embeddings(seed_video_tags)
    cand_video_tags_embeddings = video_tags_hash_embeddings(cand_video_tags)

    seed_features = tf.keras.layers.Concatenate(axis=-1, name='seed_features_concat')([
        seed_video_id_embeddings, 
        seed_video_category_embeddings,
        seed_video_tags_embeddings,
        seed_video_gap_time,
        seed_video_duration_time,
    ] + list(video_seed_numerical_features.values()))

    left_tower_inputs = tf.keras.layers.Concatenate(axis=-1, name='seed_concat_user')([
        seed_features, user_past_watches_embeddings
    ])
    left_x = tf.keras.layers.Dense(512, activation='relu', name='left_dense_0')(left_tower_inputs)
    left_x = tf.keras.layers.Dense(128, activation='relu', name='left_dense_1')(left_x)
    left_x = L2Normalization(name='left_l2_norm')(left_x)

    right_tower_inputs = tf.keras.layers.Concatenate(axis=-1, name='cand_features_concat')([
        cand_video_id_embeddings, 
        cand_video_category_embeddings,
        cand_video_tags_embeddings,
        cand_video_gap_time,
        cand_video_duration_time,
    ] + list(video_cand_numerical_features.values()))
    right_x = tf.keras.layers.Dense(512, activation='relu', name='right_dense_0')(right_tower_inputs)
    right_x = tf.keras.layers.Dense(128, activation='relu', name='right_dense_1')(right_x)
    right_x = L2Normalization(name='right_l2_norm')(right_x)

    left_tower = tf.keras.Model(
        inputs=[
            past_watches_input,
            seed_video_id_input,
            seed_video_category_input,
            seed_video_tags_input,
            seed_video_gap_time_input,
            seed_video_duration_time_input,
        ] + list(video_seed_numerical_inputs.values()), 
        outputs=left_x, 
        name='left_tower'
    )

    right_tower = tf.keras.Model(
        inputs=[
            cand_video_id_input,
            cand_video_category_input,
            cand_video_tags_input,
            cand_video_gap_time_input,
            cand_video_duration_time_input,
        ] + list(video_cand_numerical_inputs.values()), 
        outputs=right_x,
        name='right_tower'
    )
    return left_tower, right_tower


if __name__ == "__main__":
    left_tower, right_tower  = build_model()
    left_tower.summary()
    right_tower.summary()
    tf.keras.utils.plot_model(left_tower, to_file='pngs/left_tower.png', show_shapes=True)
    tf.keras.utils.plot_model(right_tower, to_file='pngs/right_tower.png', show_shapes=True)
    

