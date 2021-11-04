#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf


from deep_recommenders.datasets import MovielensRanking
from deep_recommenders.estimator.models.ranking import DeepFM


def build_columns():
    movielens = MovielensRanking()
    user_id = tf.feature_column.categorical_column_with_hash_bucket(
        "user_id", movielens.num_users)
    user_gender = tf.feature_column.categorical_column_with_vocabulary_list(
        "user_gender", movielens.gender_vocab)
    user_age = tf.feature_column.categorical_column_with_vocabulary_list(
        "user_age", movielens.age_vocab)
    user_occupation = tf.feature_column.categorical_column_with_vocabulary_list(
        "user_occupation", movielens.occupation_vocab)
    movie_id = tf.feature_column.categorical_column_with_hash_bucket(
        "movie_id", movielens.num_movies)
    movie_genres = tf.feature_column.categorical_column_with_vocabulary_list(
        "movie_genres", movielens.gender_vocab)

    base_columns = [user_id, user_gender, user_age, user_occupation, movie_id, movie_genres]
    indicator_columns = [
        tf.feature_column.indicator_column(c)
        for c in base_columns
    ]
    embedding_columns = [
        tf.feature_column.embedding_column(c, dimension=16)
        for c in base_columns
    ]
    return indicator_columns, embedding_columns


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

    movielens = MovielensRanking()
    train_spec = tf.estimator.TrainSpec(lambda: movielens.training_input_fn,
                                        max_steps=None,
                                        hooks=[early_stop_hook])
    eval_spec = tf.estimator.EvalSpec(lambda: movielens.testing_input_fn,
                                      steps=None,
                                      start_delay_secs=0,
                                      throttle_secs=0)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
