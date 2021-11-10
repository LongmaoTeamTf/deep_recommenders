#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf


from deep_recommenders.datasets import MovielensRanking
from deep_recommenders.keras.models.ranking import DeepFM


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


def main():
    movielens = MovielensRanking()
    indicator_columns, embedding_columns = build_columns()

    model = DeepFM(indicator_columns, embedding_columns, dnn_units_size=[256, 32])
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.AUC(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])

    model.fit(movielens.training_input_fn,
              epochs=10,
              steps_per_epoch=movielens.train_steps_per_epoch,
              validation_data=movielens.testing_input_fn,
              validation_steps=movielens.test_steps,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])


if __name__ == '__main__':
    main()
