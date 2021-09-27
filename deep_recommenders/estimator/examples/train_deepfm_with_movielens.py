#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from build_input_pipeline_on_movielens import build_columns
from build_input_pipeline_on_movielens import MovielensInputFun
from deep_recommenders.estimator.models.ranking import DeepFM


def model_fn(features, labels, mode):
    indicator_columns, embedding_columns = build_columns()
    outputs = DeepFM(indicator_columns, embedding_columns, [64, 32])(features)

    predictions = {"predictions": outputs}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.log_loss(labels, outputs)
    metrics = {"auc": tf.metrics.auc(labels, outputs)}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


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

    movielens = MovielensInputFun()
    train_spec = tf.estimator.TrainSpec(lambda: movielens.training_input_fn,
                                        max_steps=None,
                                        hooks=[early_stop_hook])
    eval_spec = tf.estimator.EvalSpec(lambda: movielens.testing_input_fn,
                                      steps=None,
                                      start_delay_secs=60,
                                      throttle_secs=60)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
