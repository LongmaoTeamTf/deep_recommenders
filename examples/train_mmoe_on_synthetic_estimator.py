#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from deep_recommenders.datasets import SyntheticForMultiTask
from deep_recommenders.estimator.models.multi_task_learning import MMoE


EXAMPLE_DIM = 256


def build_columns():
    return [
        tf.feature_column.numeric_column("C{}".format(i))
        for i in range(EXAMPLE_DIM)
    ]


def model_fn(features, labels, mode):
    columns = build_columns()
    outputs = MMoE(columns,
                   num_tasks=2,
                   num_experts=2,
                   task_hidden_units=[32, 10],
                   expert_hidden_units=[64, 32])(features)

    predictions = {
        "predictions0": outputs[0],
        "predictions1": outputs[1]
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    labels0 = tf.expand_dims(labels["labels0"], axis=1)
    labels1 = tf.expand_dims(labels["labels1"], axis=1)

    loss0 = tf.losses.mean_squared_error(labels=labels0, predictions=outputs[0])
    loss1 = tf.losses.mean_squared_error(labels=labels1, predictions=outputs[1])

    total_loss = loss0 + loss1

    tf.summary.scalar("task0_loss", loss0)
    tf.summary.scalar("task1_loss", loss1)
    tf.summary.scalar("total_loss", total_loss)

    metrics = {
        "task0_mse": tf.metrics.mean_squared_error(labels0, outputs[0]),
        "task1_mse": tf.metrics.mean_squared_error(labels1, outputs[1])
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = tf.group(
        optimizer.minimize(loss=loss0, global_step=tf.train.get_global_step()),
        optimizer.minimize(loss=loss1, global_step=tf.train.get_global_step()),
    )

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def build_estimator(model_dir=None, inter_op=8, intra_op=8):

    config_proto = tf.ConfigProto(device_count={'GPU': 0},
                                  inter_op_parallelism_threads=inter_op,
                                  intra_op_parallelism_threads=intra_op)

    run_config = tf.estimator.RunConfig().replace(
        tf_random_seed=42,
        keep_checkpoint_max=10,
        save_checkpoints_steps=1000,
        log_step_count_steps=100,
        session_config=config_proto)

    return tf.estimator.Estimator(model_fn=model_fn,
                                  model_dir=model_dir,
                                  config=run_config)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = build_estimator()
    early_stop_hook = tf.estimator.experimental.stop_if_no_decrease_hook(estimator, "loss", 1000)

    synthetic = SyntheticForMultiTask(512 * 1000, example_dim=EXAMPLE_DIM)
    train_spec = tf.estimator.TrainSpec(lambda: synthetic.input_fn().take(800),
                                        max_steps=None,
                                        hooks=[early_stop_hook])
    eval_spec = tf.estimator.EvalSpec(lambda: synthetic.input_fn().skip(800).take(200),
                                      steps=None,
                                      start_delay_secs=60,
                                      throttle_secs=60)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
