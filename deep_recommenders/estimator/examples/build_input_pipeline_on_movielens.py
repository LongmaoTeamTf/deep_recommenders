#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from deep_recommenders.datasets import MovieLens


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


class MovielensInputFun(MovieLens):

    def __init__(self,
                 epochs: int = 10,
                 batch_size: int = 1024,
                 buffer_size: int = 1024,
                 train_size: float = 0.8,
                 *args, **kwargs):
        super(MovielensInputFun, self).__init__(*args, **kwargs)
        self._epochs = epochs
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._train_size = train_size

    @property
    def train_steps(self):
        num_train_ratings = self.num_ratings * self._epochs * self._train_size
        return int(num_train_ratings // self._batch_size)

    @property
    def test_steps(self):
        return self.num_ratings // self._batch_size - self.train_steps

    @property
    def training_input_fn(self):
        return self.input_fn().take(self.train_steps)

    @property
    def testing_input_fn(self):
        return self.input_fn().skip(self.train_steps).take(self.test_steps)

    def input_fn(self):
        dataset = self.dataset(self._epochs, self._batch_size)
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
        dataset = dataset.prefetch(self._buffer_size)
        return dataset
