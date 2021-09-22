#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

if tf.__version__ >= "2.0.0":
    import tensorflow.compat.v1 as tf


from deep_recommenders.estimator.models.feature_interaction import DNN


def multi_task(inputs,
               num_tasks,
               task_hidden_units,
               task_output_activations,
               **kwargs):

    outputs = []

    for i in range(num_tasks):

        task_inputs = inputs[i] if isinstance(inputs, list) else inputs

        output = DNN(task_inputs,
                     task_hidden_units,
                     output_activation=task_output_activations[i],
                     **kwargs)
        outputs.append(output)

    return outputs
