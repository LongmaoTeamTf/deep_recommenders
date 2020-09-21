"""
@Description: Sampling-Bias-Corrected Neural Retrieval Model
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-08-27 17:22:16
@LastEditors: Wang Yao
@LastEditTime: 2020-09-21 14:10:48
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
                 regularizer='l2',
                 initializer='he_uniform',
                 trainable=True,
                 **kwargs):
        super(HashEmbeddings, self).__init__(**kwargs)
        self._hash_bucket_size = hash_bucket_size
        self._embedding_dim = embedding_dim
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

    def call(self, inputs, mean=False, **kwargs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        outputs = K.dot(inputs, self.embeddings)
        if mean is True:
            outputs = tf.math.divide_no_nan(
                outputs, 
                tf.tile(tf.reduce_sum(inputs, axis=-1, keepdims=True), (1, self._embedding_dim))
            )
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

    def __init__(self, kernel_size, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
        self._kernel_size = kernel_size

    def build(self, input_shape):
        self._weights = self.add_weight(
            shape=(self._kernel_size,),
            initializer=initializers.glorot_uniform,
            trainable=True,
            name='l2norm_weights'
        )
        self._bais = self.add_weight(
            shape=(self._kernel_size,),
            initializer=initializers.zeros,
            trainable=True,
            name='l2norm_bais'
        )

    def call(self, inputs, **kwargs):
        outputs = self._weights * inputs + self._bais
        return K.l2_normalize(outputs)


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

    _boundaries = {
        'gap_time': [0, 1/365, 3/365, 7/365, 15/365, 30/365, 0.5, 1., 2., 3., 4.],
        'duration_time': [x*60 for x in range(30)],
        'play_count': [x*50 for x in range(101)],
        'like_count': [x*20 for x in range(201)],
        'collect_count': [x*10 for x in range(201)],
        'share_count': [x*10 for x in range(101)]
    }

    video_ids_hash = tf.feature_column.categorical_column_with_hash_bucket(
            key='video_ids', hash_bucket_size=_video_ids_hash_bucket_size, dtype=tf.string)
    video_ids_indicator = tf.feature_column.indicator_column(video_ids_hash)
    video_ids_dense = tf.keras.layers.DenseFeatures(video_ids_indicator, trainable=False, name="video_ids")
    seed_video_id_input = tf.keras.layers.Input(shape=(1,), name='seed_id', dtype=tf.string)
    cand_video_id_input = tf.keras.layers.Input(shape=(1,), name='cand_id', dtype=tf.string)
    seed_video_id = video_ids_dense({'video_ids': seed_video_id_input})
    cand_video_id = video_ids_dense({'video_ids': cand_video_id_input})

    video_categories_hash = tf.feature_column.categorical_column_with_hash_bucket(
            key='video_categories', hash_bucket_size=_video_categories_hash_bucket_size, dtype=tf.string)
    video_categories_indicator = tf.feature_column.indicator_column(video_categories_hash)
    video_categories_dense = tf.keras.layers.DenseFeatures(video_categories_indicator, trainable=False, name='video_categories')
    seed_video_category_input = tf.keras.layers.Input(shape=(1,), name='seed_category', dtype=tf.string)
    cand_video_category_input = tf.keras.layers.Input(shape=(1,), name='cand_category', dtype=tf.string)
    seed_video_category = video_categories_dense({'video_categories': seed_video_category_input})
    cand_video_category = video_categories_dense({'video_categories': cand_video_category_input})

    video_tags_hash = tf.feature_column.categorical_column_with_hash_bucket(
            key='video_tags', hash_bucket_size=_video_tags_hash_bucket_size, dtype=tf.string)
    video_tags_indicator = tf.feature_column.indicator_column(video_tags_hash)
    video_tags_dense = tf.keras.layers.DenseFeatures(video_tags_indicator, trainable=False, name='video_tags')
    seed_video_tags_input = tf.keras.layers.Input(shape=(_max_tags_num,), name='seed_tags', dtype=tf.string)
    cand_video_tags_input = tf.keras.layers.Input(shape=(_max_tags_num,), name='cand_tags', dtype=tf.string)
    seed_video_tags = video_tags_dense({'video_tags': seed_video_tags_input})
    cand_video_tags = video_tags_dense({'video_tags': cand_video_tags_input})

    video_gap_time_num = tf.feature_column.numeric_column(
        key='video_gap_time', default_value=-1, dtype=tf.int32, normalizer_fn=_time_exp_norm)
    video_gap_time_num = tf.feature_column.bucketized_column(video_gap_time_num, boundaries=_boundaries.get('gap_time'))
    video_gap_time_dense = tf.keras.layers.DenseFeatures(video_gap_time_num, trainable=False, name='video_gap_time')
    seed_video_gap_time_input = tf.keras.layers.Input(shape=(1,), name='seed_gap_time')
    cand_video_gap_time_input = tf.keras.layers.Input(shape=(1,), name='cand_gap_time')
    seed_video_gap_time = video_gap_time_dense({'video_gap_time': seed_video_gap_time_input})
    cand_video_gap_time = video_gap_time_dense({'video_gap_time': cand_video_gap_time_input})

    video_duration_time = tf.feature_column.numeric_column(
        key='video_duration_time', default_value=-1, dtype=tf.int32, normalizer_fn=None)
    video_duration_time = tf.feature_column.bucketized_column(video_duration_time, boundaries=_boundaries.get('duration_time'))
    video_duration_time_dense = tf.keras.layers.DenseFeatures(video_duration_time, trainable=False, name='video_duration_time')
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
            key='video_'+feat, default_value=-1, dtype=tf.int32, normalizer_fn=None)
        feat_num = tf.feature_column.bucketized_column(feat_num, boundaries=_boundaries.get(feat))
        feat_dense = tf.keras.layers.DenseFeatures(feat_num, trainable=False, name='video_'+feat)
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
        _video_tags_hash_bucket_size, _video_tags_embedding_dim, name='video_tags_hash_embeddings')

    seed_video_id_embeddings = video_ids_hash_embeddings(seed_video_id)
    cand_video_id_embeddings = video_ids_hash_embeddings(cand_video_id)
    user_past_watches_embeddings = video_ids_hash_embeddings(user_past_watches, mean=True)

    seed_video_category_embeddings = video_categories_hash_embeddings(seed_video_category)
    cand_video_category_embeddings = video_categories_hash_embeddings(cand_video_category)

    seed_video_tags_embeddings = video_tags_hash_embeddings(seed_video_tags, mean=True)
    cand_video_tags_embeddings = video_tags_hash_embeddings(cand_video_tags, mean=True)

    seed_features = tf.keras.layers.Concatenate(axis=-1, name='seed_features_concat')([
        seed_video_id_embeddings, 
        seed_video_category_embeddings,
        seed_video_tags_embeddings,
        seed_video_gap_time,
        seed_video_duration_time,
    ] + list(video_seed_numerical_features.values()))

    query_tower_inputs = tf.keras.layers.Concatenate(axis=-1, name='seed_concat_user')([
        seed_features, user_past_watches_embeddings
    ])
    query_x = tf.keras.layers.Dense(512, name='query_dense_0')(query_tower_inputs)
    query_x = tf.keras.layers.PReLU(name='query_prelu_0')(query_x)
    query_x = tf.keras.layers.Dense(128, name='query_dense_1')(query_x)
    query_x = tf.keras.layers.PReLU(name='query_prelu_1')(query_x)
    query_x = L2Normalization(128, name='query_l2_norm')(query_x)

    candidate_tower_inputs = tf.keras.layers.Concatenate(axis=-1, name='cand_features_concat')([
        cand_video_id_embeddings, 
        cand_video_category_embeddings,
        cand_video_tags_embeddings,
        cand_video_gap_time,
        cand_video_duration_time,
    ] + list(video_cand_numerical_features.values()))
    candidate_x = tf.keras.layers.Dense(512, name='candidate_dense_0')(candidate_tower_inputs)
    candidate_x = tf.keras.layers.PReLU(name='candidate_prelu_0')(candidate_x)
    candidate_x = tf.keras.layers.Dense(128, name='candidate_dense_1')(candidate_x)
    candidate_x = tf.keras.layers.PReLU(name='candidate_prelu_1')(candidate_x)
    candidate_x = L2Normalization(128, name='candidate_l2_norm')(candidate_x)

    query_tower = tf.keras.Model(
        inputs=[
            past_watches_input,
            seed_video_id_input,
            seed_video_category_input,
            seed_video_tags_input,
            seed_video_gap_time_input,
            seed_video_duration_time_input,
        ] + list(video_seed_numerical_inputs.values()), 
        outputs=query_x, 
        name='query_tower'
    )

    candidate_tower = tf.keras.Model(
        inputs=[
            cand_video_id_input,
            cand_video_category_input,
            cand_video_tags_input,
            cand_video_gap_time_input,
            cand_video_duration_time_input,
        ] + list(video_cand_numerical_inputs.values()), 
        outputs=candidate_x,
        name='candidate_tower'
    )
    return query_tower, candidate_tower


if __name__ == "__main__":
    query_tower, candidate_tower  = build_model()
    query_tower.summary()
    candidate_tower.summary()
    tf.keras.utils.plot_model(query_tower, to_file='query_tower.png', show_shapes=True)
    tf.keras.utils.plot_model(candidate_tower, to_file='candidate_tower.png', show_shapes=True)
    

