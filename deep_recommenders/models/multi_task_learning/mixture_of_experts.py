#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def _synthetic_data(num_examples, example_dim=100, c=0.3, p=0.8, m=5):

    mu1 = np.random.normal(size=example_dim)
    mu1 = (mu1 - np.mean(mu1)) / (np.std(mu1) * np.sqrt(example_dim))

    mu2 = np.random.normal(size=example_dim)
    mu2 -= mu2.dot(mu1) * mu1
    mu2 /= np.linalg.norm(mu2)

    w1 = c * mu1
    w2 = c * (p * mu1 + np.sqrt(1. - p ** 2) * mu2)

    alpha = np.random.normal(size=m)
    beta = np.random.normal(size=m)

    examples = np.random.normal(size=(num_examples, example_dim))

    w1x = np.matmul(examples, w1)
    w2x = np.matmul(examples, w2)

    sin1, sin2 = 0., 0.
    for i in range(m):
        sin1 += np.sin(alpha[i] * w1x + beta[i])
        sin2 += np.sin(alpha[i] * w2x + beta[i])

    y1 = w1x + sin1 + np.random.normal(size=num_examples, scale=0.01)
    y2 = w2x + sin2 + np.random.normal(size=num_examples, scale=0.01)

    return examples.astype(np.float32), (y1.astype(np.float32), y2.astype(np.float32))


def synthetic_data_input_fn(num_examples, epochs=1, batch_size=256, buffer_size=256, **kwargs):

    synthetic_data = _synthetic_data(num_examples, **kwargs)

    dataset = tf.data.Dataset.from_tensor_slices(synthetic_data)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(buffer_size)

    return dataset
