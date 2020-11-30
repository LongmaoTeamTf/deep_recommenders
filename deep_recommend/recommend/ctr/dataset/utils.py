"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-11-30 15:40:28
@LastEditors: Wang Yao
@LastEditTime: 2020-11-30 15:44:18
"""
import tensorflow as tf


def numeric_log_normalizer(value):
    """数值型特征对数归一化"""
    def true_fn(): return tf.math.log(value+1)
    def false_fn(): return value
    return tf.where(value > -1., true_fn(), false_fn())

