"""
@Description: Criteo CTR dataset
@version: https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz
@License: MIT
@Author: Wang Yao
@Date: 2020-11-09 14:19:38
@LastEditors: Wang Yao
@LastEditTime: 2020-11-30 16:17:40
"""
import numpy as np
import tensorflow as tf
from dataset.utils import numeric_log_normalizer


class criteoDataFLow():
    """ Criteo数据集输入数据流 """

    def __init__(self, filepath, batch_size=128, epochs=10):
        self._filepath = filepath
        self._batch_size = batch_size
        self._epochs = epochs
        self._integer_cols_num = 13
        self._categorical_cols_num = 26
        self.integer_cols_names = [f"I{i}" for i in range(self._integer_cols_num)]
        self.categorical_cols_names = [f"C{i}" for i in range(self._categorical_cols_num)]

    def create_criteo_generator(self):
        """创建Criteo数据迭代器"""
        for fn in self._filepath:
            with open(fn, "r", encoding="utf8") as f:
                for line in f:
                    instance = line.strip("\n").split("\t")
                    label = int(instance[0])
                    integer_cols = [-1 if x == "" else int(x) for x in instance[1:self._integer_cols_num+1]]
                    categorical_cols = instance[self._integer_cols_num+1:]
                    yield label, integer_cols, categorical_cols

    def criteo_data_format(self, label, integer_cols, categorical_cols):
        """数据格式转换"""
        integer_cols_tensor = tf.split(integer_cols, self._integer_cols_num, axis=-1)
        categorical_cols_tensor = tf.split(categorical_cols, self._categorical_cols_num, axis=-1)
        features = dict(zip(
            self.integer_cols_names + self.categorical_cols_names,
            integer_cols_tensor + categorical_cols_tensor
        ))
        return features, label

    def create_criteo_dataset_from_generator(self):
        """创建输入"""
        output_types = (tf.float32, tf.float32, tf.string)
        criteo_dataset = tf.data.Dataset.from_generator(self.create_criteo_generator, output_types)

        criteo_dataset = criteo_dataset.map(self.criteo_data_format)
        criteo_dataset = criteo_dataset.repeat(self._epochs)
        criteo_dataset = criteo_dataset.batch(self._batch_size)
        criteo_dataset = criteo_dataset.prefetch(self._batch_size)

        return criteo_dataset


def create_feature_layers():
    """ 构造Criteo数据的特征处理层 """ 
    integer_cols_names = [f"I{i}" for i in range(13)]
    categorical_cols_names = [f"C{i}" for i in range(26)]

    dense_inputs, dense_cols = [], []
    for integer_col_name in integer_cols_names:
        dense_input = tf.keras.Input(shape=(1,), name=integer_col_name)
        dense_col = tf.feature_column.numeric_column(
            key=integer_col_name, default_value=-1, normalizer_fn=None)
        dense_col = tf.feature_column.bucketized_column(dense_col, list(range(0, 5000, 10)))
        dense_col = tf.feature_column.embedding_column(dense_col, 10, trainable=True)
        dense_inputs.append(dense_input)
        dense_cols.append(dense_col)
        
    sparse_inputs, sparse_cols = [], []
    for categorical_col_name in categorical_cols_names:
        sparse_input = tf.keras.Input(shape=(1,), name=categorical_col_name, dtype=tf.string)
        sparse_col = tf.feature_column.categorical_column_with_hash_bucket(key=categorical_col_name, hash_bucket_size=10000)
        sparse_embed_col = tf.feature_column.embedding_column(sparse_col, 10, trainable=True)
        sparse_inputs.append(sparse_input)
        sparse_cols.append(sparse_embed_col)

    return (sparse_inputs, sparse_cols), (dense_inputs, dense_cols)
