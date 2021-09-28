#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from build_input_pipeline_on_movielens import build_columns
from build_input_pipeline_on_movielens import MovielensInputFun
from deep_recommenders.estimator.models.ranking import WDL


def cross_product_transformation():
    crossed_columns = [
        tf.feature_column.crossed_column(['user_gender', 'user_age'], 14),
        tf.feature_column.crossed_column(['user_gender', 'user_occupation'], 40),
        tf.feature_column.crossed_column(['user_age', 'user_occupation'], 140),
    ]
    crossed_product_columns = [
        tf.feature_column.indicator_column(c)
        for c in crossed_columns
    ]
    return crossed_product_columns


def model_fn(features, labels, mode):
    indicator_columns, embedding_columns = build_columns()
    crossed_product_columns = cross_product_transformation()
    outputs = WDL(indicator_columns + crossed_product_columns, embedding_columns, [64, 16])(features)

    predictions = {"predictions": outputs}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.log_loss(labels, outputs)
    metrics = {"auc": tf.metrics.auc(labels, outputs)}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    wide_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, "wide")
    wide_optimizer = tf.train.FtrlOptimizer(0.01, l1_regularization_strength=0.5)
    wide_train_op = wide_optimizer.minimize(loss=loss,
                                            global_step=tf.train.get_global_step(),
                                            var_list=wide_variables)

    deep_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, "deep")
    deep_optimizer = tf.train.AdamOptimizer(0.01)
    deep_train_op = deep_optimizer.minimize(loss=loss,
                                            global_step=tf.train.get_global_step(),
                                            var_list=deep_variables)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(update_ops, wide_train_op, deep_train_op)

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
