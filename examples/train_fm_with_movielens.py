#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from build_input_pipeline_on_movielens import build_columns
from build_input_pipeline_on_movielens import MovielensInputFun
from deep_recommenders.estimator.models.feature_interaction import FM


def model_fn(features, labels, mode):
    indicator_columns, embedding_columns = build_columns()
    outputs = FM(indicator_columns, embedding_columns)(features)

    predictions = {"predictions": outputs}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sigmoid_cross_entropy(labels, outputs)
    metrics = {"auc": tf.metrics.auc(labels, tf.nn.sigmoid(outputs))}
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


def export_saved_model(estimator, export_path):
    indicator_columns, embedding_columns = build_columns()
    columns = indicator_columns + embedding_columns

    feature_spec = tf.feature_column.make_parse_example_spec(columns)
    example_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    estimator.export_saved_model(export_path, example_input_fn)


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

    export_saved_model(estimator, "FM")


if __name__ == '__main__':
    main()
