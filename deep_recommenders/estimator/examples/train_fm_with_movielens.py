#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from deep_recommenders.datasets import MovieLens
from deep_recommenders.estimator.models.feature_interaction import FM


def build_columns():
    movielens = MovieLens()
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


def input_fn(split, epochs=10, batch_size=1024):
    movielens = MovieLens()
    dataset = movielens.dataset(epochs, batch_size)
    dataset = dataset.map(lambda x, y: (
        {
            "user_id": x["UserID"],
            "user_gender": x["Gender"],
            "user_age": x["Age"],
            "user_occupation": x["Occupation"],
            "movie_id": x["MovieID"],
            "movie_genres": x["Genres"]
        },
        tf.expand_dims(tf.where(y > 3,
                       tf.ones_like(y, dtype=tf.float32),
                       tf.zeros_like(y, dtype=tf.float32)), axis=1)
    ))
    dataset = dataset.prefetch(1024)
    train_steps = int(movielens.num_ratings * epochs * 0.8 // batch_size)
    test_steps = movielens.num_ratings // batch_size - train_steps
    if split == "train":
        return dataset.take(train_steps)
    elif split == "test":
        return dataset.skip(train_steps).take(test_steps)
    return dataset


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = build_estimator()

    early_stop_hook = tf.estimator.experimental.stop_if_no_decrease_hook(estimator, "loss", 1000)

    train_spec = tf.estimator.TrainSpec(lambda: input_fn("train"),
                                        max_steps=None,
                                        hooks=[early_stop_hook])
    eval_spec = tf.estimator.EvalSpec(lambda: input_fn("test"),
                                      steps=None,
                                      start_delay_secs=60,
                                      throttle_secs=60)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
