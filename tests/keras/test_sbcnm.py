#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf

from absl.testing import parameterized

from deep_recommenders.keras.models.retrieval import sbcnm


class TestSBCNM(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters(3, 5, 10, 15)
    def test_hard_negative_mining(self, num_hard_negatives):

        logits_shape = (2, 20)
        rng = np.random.RandomState(42) # pylint: disable=no-member

        logits = rng.uniform(size=logits_shape).astype(np.float32)
        labels = rng.permutation(np.eye(*logits_shape).T).T.astype(np.float32)

        out_logits, out_labels = sbcnm.HardNegativeMining(num_hard_negatives)(logits, labels)

        self.assertEqual(out_logits.shape[-1], num_hard_negatives + 1)

        self.assertAllClose(
            tf.reduce_sum(out_logits * out_labels, axis=-1),
            tf.reduce_sum(logits * labels, axis=-1))

        logits = logits + labels * 1000.0

        out_logits, out_labels = sbcnm.HardNegativeMining(num_hard_negatives)(logits, labels)
        out_logits, out_labels = out_logits.numpy(), out_labels.numpy()

        # Highest K logits are always returned.
        self.assertAllClose(
            np.sort(logits, axis=1)[:, -num_hard_negatives - 1:],
            np.sort(out_logits))

    def test_remove_accidental_negative(self):

        logits_shape = (2, 4)
        rng = np.random.RandomState(42) # pylint: disable=no-member

        logits = rng.uniform(size=logits_shape).astype(np.float32)
        labels = rng.permutation(np.eye(*logits_shape).T).T.astype(np.float32)
        identifiers = rng.randint(0, 3, size=logits_shape[-1])

        out_logits = sbcnm.RemoveAccidentalNegative()(logits, labels, identifiers)

        self.assertAllClose(tf.reduce_sum(out_logits * labels, axis=1),
                            tf.reduce_sum(logits * labels, axis=1))


if __name__ == "__main__":
    tf.test.main()
