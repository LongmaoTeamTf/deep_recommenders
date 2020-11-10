"""
@Description: Criteo CTR dataset
@version: https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz
@License: MIT
@Author: Wang Yao
@Date: 2020-11-09 14:19:38
@LastEditors: Wang Yao
@LastEditTime: 2020-11-10 15:28:14
"""
import numpy as np
import tensorflow as tf


class criteoDataFLow():
    """Criteo数据集输入数据流"""

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
        with open(self._filepath, "r", encoding="utf8") as f:
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

