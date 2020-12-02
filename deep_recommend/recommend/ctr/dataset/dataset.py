"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-12-02 15:13:59
@LastEditors: Wang Yao
@LastEditTime: 2020-12-02 19:02:47
"""
import os
import tensorflow as tf


class TfDatasetCSV(object):
    """ Tensorflow Dataset """

    __float32__ = tf.constant(0, dtype=tf.float32)
    __int32__ = tf.constant(0, dtype=tf.int32)
    __string__ = tf.constant("", dtype=tf.string)

    def __init__(self, 
                 csv_header: str, 
                 select_cols_indexs: list,
                 select_cols_defs: list,
                 label_column_index: int,
                 batch_size: int, 
                 skip_head_lines: int=1,
                 field_delim=",",
                 na_value="",
                 **kwargs):
        super(TfDatasetCSV, self).__init__(**kwargs)
        self._csv_header = csv_header
        self._select_cols_indexs = self.parse_select_indexs(select_cols_indexs)
        self._select_cols_defs = self.parse_select_defs(select_cols_defs)
        self._label_column_index = label_column_index
        self._batch_size = batch_size
        self._skip_head_lines = skip_head_lines
        self._field_delim = field_delim
        self._na_value = na_value

    def __call__(self, filenames):
        """ 生成数据集 """
        dataset = self.get_dataset_from_txts(filenames, self.parse_line)
        steps = self.calc_steps(filenames)
        return dataset, steps

    def calc_steps(self, filenames):
        """ 计算迭代步数 """
        total_num = 0
        for fn in filenames:
            cmd = "wc -l < {}".format(fn)
            cmd_res = os.popen(cmd)
            num_lines = int(cmd_res.read().strip())
            num_lines -= self._skip_head_lines
            total_num += num_lines
        steps = total_num // self._batch_size
        return steps
    
    def parse_select_indexs(self, indexs):
        return [int(i) for i in indexs]

    def parse_select_defs(self, defs):
        select_defs = []
        for default in defs:
            if default == "F":
                select_defs.append(self.__float32__)
            elif default == "I":
                select_defs.append(self.__int32__)
            elif default == "S":
                select_defs.append(self.__string__)
            else:
                select_defs.append(self.__string__)
        return select_defs

    def parse_line(self, csv_line):
        """ 解析数据 """
        parsed_columns = tf.io.decode_csv(
            csv_line,
            self._select_cols_defs,
            select_cols=self._select_cols_indexs,
            field_delim=self._field_delim,
            na_value=self._na_value
        )
        features = {
            self._csv_header[i]: col
            for i, col in zip(self._select_cols_indexs, parsed_columns)
            if i != self._label_column_index
        }
        labels = tf.stack(parsed_columns[
            self._select_cols_indexs.index(self._label_column_index)])
        return features, labels

    def get_dataset_from_txts(self, filenames: list, parser_func):
        """ 构建文本文件数据集 """
        list_ds = tf.data.Dataset.list_files(filenames)
        dataset = list_ds.interleave(
            lambda fp: tf.data.TextLineDataset(fp).skip(self._skip_head_lines),
        )
        dataset = dataset.map(map_func=parser_func)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(self._batch_size)
        return dataset
