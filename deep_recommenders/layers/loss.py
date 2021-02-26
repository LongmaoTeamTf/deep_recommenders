"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2021-02-20 16:23:44
@LastEditors: Wang Yao
@LastEditTime: 2021-02-26 15:23:37
"""
from typing import Tuple

import numpy as np
import tensorflow as tf


MAX_FLOAT = np.finfo(np.float32).max / 100.0
MIN_FLOAT = np.finfo(np.float32).min / 100.0


def _gather_elements_along_row(data: tf.Tensor,
                               column_indices: tf.Tensor) -> tf.Tensor:
  """与factorized_top_k中_take_long_axis相同"""
  with tf.control_dependencies(
      [tf.assert_equal(tf.shape(data)[0], tf.shape(column_indices)[0])]):
    num_row = tf.shape(data)[0]
    num_column = tf.shape(data)[1]
    num_gathered = tf.shape(column_indices)[1]
    row_indices = tf.tile(
        tf.expand_dims(tf.range(num_row), -1),
        [1, num_gathered])
    flat_data = tf.reshape(data, [-1])
    flat_indices = tf.reshape(
        row_indices * num_column + column_indices, [-1])
    return tf.reshape(
        tf.gather(flat_data, flat_indices), [num_row, num_gathered])


class HardNegativeMining(tf.keras.layers.Layer):
    """Hard Negative"""

    def __init__(self, num_hard_negatives: int, **kwargs):
        super(HardNegativeMining, self).__init__(**kwargs)
        
        self._num_hard_negatives = num_hard_negatives


    def call(self, logits: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        num_sampled = tf.minimum(self._num_hard_negatives + 1, tf.shape(logits)[1])

        _, indices = tf.nn.top_k(logits + labels * MAX_FLOAT, k=num_sampled, sorted=False)

        logits = _gather_elements_along_row(logits, indices)
        labels = _gather_elements_along_row(labels, indices)

        return logits, labels


class RemoveAccidentalNegative(tf.keras.layers.Layer):
    """删除batch内与正样本id一致的负样本"""

    def call(self, 
             logits: tf.Tensor, 
             labels: tf.Tensor, 
             identifiers: tf.Tensor
        ) -> tf.Tensor:
        """Zeros logits of accidental negatives
        Args:
            logits: [batch_size, num_candidates] 2D tensor
            labels: [batch_size, num_candidates] one-hot 2D tensor
            identifiers: [num_candidates] candidates identifiers tensor
        Returns:
            logits: Modified logits.
        """
        identifiers = tf.expand_dims(identifiers, 1)
        positive_indices = tf.math.argmax(labels, axis=1)
        positive_identifier = tf.gather(identifiers, positive_indices)

        duplicate = tf.equal(positive_identifier, tf.transpose(identifiers))
        duplicate = tf.cast(duplicate, labels.dtype)

        duplicate = duplicate - labels

        return logits + duplicate * MIN_FLOAT


class SamplingProbablityCorrection(tf.keras.layers.Layer):
  """Sampling probability correction."""

  def call(self, 
           logits: tf.Tensor,
           candidate_sampling_probability: tf.Tensor) -> tf.Tensor:
    """Corrects the input logits to account for candidate sampling probability."""

    return logits - tf.math.log(candidate_sampling_probability)
