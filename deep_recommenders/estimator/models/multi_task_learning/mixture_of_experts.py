#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

if tf.__version__ >= "2.3.0":
    import tensorflow.compat.v1 as tf

from deep_recommenders.estimator.models.multi_task_learning import multi_task


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


def gating_network(inputs, num_experts, expert_index=None):
    """
    Gating network: y = SoftMax(W * inputs)
    :param inputs: tf.Tensor
    :param num_experts: Int > 0, number of expert networks.
    :param expert_index: Int, index of expert network.
    :return: tf.Tensor
    """

    x = tf.layers.dense(inputs,
                        units=num_experts,
                        use_bias=False,
                        name="expert{}_gate".format(expert_index))

    return tf.nn.softmax(x)


def one_gate(inputs,
             num_tasks,
             num_experts,
             task_hidden_units,
             task_output_activations,
             expert_hidden_units,
             expert_hidden_activation=tf.nn.relu,
             task_hidden_activation=tf.nn.relu,
             task_initializer=None,
             task_dropout=None):

    experts_gate = gating_network(inputs, num_experts)

    experts_outputs = []
    for i in range(num_experts):
        x = inputs
        for j, units in enumerate(expert_hidden_units):
            x = tf.layers.dense(x, units, activation=expert_hidden_activation, name="expert{}_dense{}".format(i, j))
        experts_outputs.append(x)

    experts_outputs = tf.stack(experts_outputs, axis=1)
    experts_selector = tf.expand_dims(experts_gate, axis=1)

    outputs = tf.linalg.matmul(experts_selector, experts_outputs)

    multi_task_inputs = tf.squeeze(outputs)

    return multi_task(multi_task_inputs,
                      num_tasks,
                      task_hidden_units,
                      task_output_activations,
                      hidden_activation=task_hidden_activation,
                      hidden_dropout=task_dropout,
                      initializer=task_initializer)


def multi_gate(inputs,
               num_tasks,
               num_experts,
               task_hidden_units,
               task_output_activations,
               expert_hidden_units,
               expert_hidden_activation=tf.nn.relu,
               task_hidden_activation=tf.nn.relu,
               task_initializer=None,
               task_dropout=None):

    experts_outputs = []
    for i in range(num_experts):
        x = inputs
        for j, units in enumerate(expert_hidden_units[:-1]):
            x = tf.layers.dense(x, units, activation=expert_hidden_activation, name="expert{}_dense{}".format(i, j))

        x = tf.layers.dense(x, expert_hidden_units[-1], name="expert{}_out".format(i))

        experts_outputs.append(x)

    experts_outputs = tf.stack(experts_outputs, axis=1)

    outputs = []
    for i in range(num_experts):
        expert_gate = gating_network(inputs, num_experts, expert_index=i)
        expert_selector = tf.expand_dims(expert_gate, axis=1)

        output = tf.linalg.matmul(expert_selector, experts_outputs)

        outputs.append(tf.squeeze(output))

    return multi_task(outputs,
                      num_tasks,
                      task_hidden_units,
                      task_output_activations,
                      hidden_activation=task_hidden_activation,
                      hidden_dropout=task_dropout,
                      initializer=task_initializer)
