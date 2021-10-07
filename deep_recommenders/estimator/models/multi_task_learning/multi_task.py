#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

if tf.__version__ >= "2.0.0":
    import tensorflow.compat.v1 as tf


from deep_recommenders.estimator.models.feature_interaction import dnn


def multi_task(inputs,
               num_tasks,
               task_activations,
               dnn_units,
               dnn_activation=tf.nn.relu,
               dnn_batch_normalization=False,
               dnn_dropout=None,
               **dnn_kwargs):

    outputs = []

    for i in range(num_tasks):

        task_inputs = inputs[i] if isinstance(inputs, list) else inputs

        logits = dnn(task_inputs,
                     dnn_units + [1],
                     activation=dnn_activation,
                     batch_normalization=dnn_batch_normalization,
                     dropout=dnn_dropout,
                     **dnn_kwargs)
        if task_activations[i] is not None:
            logits = task_activations[i](logits)
        outputs.append(logits)

    return outputs
