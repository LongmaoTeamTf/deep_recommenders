#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

if tf.__version__ >= "2.3.0":
    import tensorflow.compat.v1 as tf

from deep_recommenders.estimator.models.multi_task_learning import multi_task


def _dense(x, units, activation=None, dropout=None, name=None):
    weights = tf.get_variable("w{}".format(name),
                              shape=(x.shape[-1], units),
                              dtype=tf.float32)
    bias = tf.get_variable("b{}".format(name),
                           shape=(units,),
                           dtype=tf.float32,
                           initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, bias)

    if dropout is not None:
        x = tf.nn.dropout(x, rate=dropout)

    if activation is not None:
        x = activation(x)

    return x


def shared_bottom(x: tf.Tensor,
                  num_tasks: int,
                  bottom_units: list,
                  task_units: list,
                  task_output_activation: list,
                  bottom_initializer: tf.Tensor = tf.truncated_normal_initializer(),
                  bottom_activation=tf.nn.relu,
                  bottom_dropout: float = None,
                  task_initializer: tf.Tensor = tf.truncated_normal_initializer(),
                  task_dropout: float = None,
                  task_activation=tf.nn.relu):

    with tf.variable_scope("bottom", initializer=bottom_initializer):
        for i, units in enumerate(bottom_units[:-1]):
            x = _dense(x, units, activation=bottom_activation, dropout=bottom_dropout, name=i)

        bottom_out = _dense(x, bottom_units[-1], name="out")

    outputs = []

    for task_idx in range(num_tasks):
        x = bottom_out
        with tf.variable_scope("task{}".format(task_idx), initializer=task_initializer):
            for i, units in enumerate(task_units):
                x = _dense(x, units, activation=task_activation, dropout=task_dropout, name=i)

            task_out = _dense(x, 1, name="out")

            output_activation = task_output_activation[task_idx]
            if output_activation == "sigmoid":
                task_out = tf.nn.sigmoid(task_out)

            outputs.append(task_out)

    return outputs


def shared_bottom_v2(x: tf.Tensor,
                     num_tasks: int,
                     bottom_units: list,
                     task_hidden_units: list,
                     task_output_activations: list,
                     bottom_initializer: tf.Tensor = None,
                     bottom_activation=tf.nn.relu,
                     bottom_dropout: float = None,
                     task_initializer: tf.Tensor = None,
                     task_dropout: float = None,
                     task_activation=tf.nn.relu):

    for i, units in enumerate(bottom_units[:-1]):
        x = tf.layers.dense(x,
                            units,
                            activation=bottom_activation,
                            kernel_initializer=bottom_initializer,
                            name="bottom_dense{}".format(i))

        if bottom_dropout is not None:
            x = tf.layers.dropout(x, rate=bottom_dropout, name="bottom_dropout{}".format(i))

    bottom_out = tf.layers.dense(x, bottom_units[-1], kernel_initializer=bottom_initializer, name="bottom_out")

    outputs = multi_task(bottom_out,
                         num_tasks,
                         task_hidden_units,
                         task_output_activations,
                         hidden_activation=task_activation,
                         hidden_dropout=task_dropout,
                         initializer=task_initializer)

    return outputs


def model_fn(features, labels, mode, params):

    outputs = shared_bottom_v2(features["inputs"],
                               num_tasks=params.get("num_tasks"),
                               bottom_units=params.get("bottom_units"),
                               task_hidden_units=params.get("task_units"),
                               task_output_activations=params.get("task_output_activations"))
    predictions = {
        "y{}".format(i): y
        for i, y in enumerate(outputs)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    task_losses = params.get("task_losses")

    total_loss = tf.Variable(0., name="total_loss")
    losses = []
    metrics = {}

    for i, (t, y) in enumerate(predictions.items()):

        y = tf.squeeze(y)

        if task_losses[i] == 'log_loss':
            loss = tf.losses.log_loss(labels=labels[i], predictions=y)
            auc_op = tf.metrics.auc(labels=labels, predictions=y, name='auc_op')
            tf.summary.scalar("auc", auc_op[-1])
            metrics["auc"] = auc_op

        elif task_losses[i] == 'mse':
            loss = tf.losses.mean_squared_error(labels=labels[i], predictions=y)

        else:
            loss = tf.losses.mean_squared_error(labels=labels[i], predictions=y)

        losses.append(loss)
        total_loss = total_loss + loss
        metrics["loss_{}".format(t)] = loss
        tf.summary.scalar("loss_{}".format(t), loss)

    tf.summary.scalar("total_loss", total_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)

    optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])

    train_op = tf.group(*[
        optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        for loss in losses
    ])

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def shared_bottom_estimator(model_dir, inter_op, intra_op, params):

    config_proto = tf.compat.v1.ConfigProto(device_count={'GPU': 0},
                                            inter_op_parallelism_threads=inter_op,
                                            intra_op_parallelism_threads=intra_op)

    run_config = tf.estimator.RunConfig().replace(
        tf_random_seed=42,
        keep_checkpoint_max=10,
        save_checkpoints_steps=200,
        log_step_count_steps=10,
        session_config=config_proto)

    return tf.estimator.Estimator(model_fn=model_fn,
                                  model_dir=model_dir,
                                  params=params,
                                  config=run_config)
