#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True

import os
import tempfile
import numpy as np
import tensorflow as tf

from deep_recommenders.keras.models.ranking import FactorizationMachine
from deep_recommenders.datasets import MovielensRanking


class TestFM(tf.test.TestCase):

    def test_model(self):
        movielens = MovielensRanking()

        def build_columns():
            user_id = tf.feature_column.categorical_column_with_hash_bucket(
                "user_id", movielens.num_users)
            movie_id = tf.feature_column.categorical_column_with_hash_bucket(
                "movie_id", movielens.num_movies)
            base_columns = [user_id, movie_id]
            _indicator_columns = [
                tf.feature_column.indicator_column(c)
                for c in base_columns
            ]
            _embedding_columns = [
                tf.feature_column.embedding_column(c, dimension=16)
                for c in base_columns
            ]
            return _indicator_columns, _embedding_columns

        indicator_columns, embedding_columns = build_columns()
        model = FactorizationMachine(indicator_columns, embedding_columns)
        model.compile(loss=tf.keras.losses.binary_crossentropy,
                      optimizer=tf.keras.optimizers.Adam())
        dataset = movielens.testing_input_fn.map(
            lambda x, y: ({"user_id": x["user_id"], "movie_id": x["movie_id"]}, y))
        model.fit(dataset,
                  steps_per_epoch=100,
                  verbose=-1)
        test_data = {"user_id": np.asarray([["1"], ["2"]]),
                     "movie_id": np.asarray([["1"], ["2"]])}
        model_pred = model.predict(test_data)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "FM")
            model.save(path)
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict(test_data)
        for model_layer, loaded_layer in zip(model.layers, loaded_model.layers):
            assert model_layer.get_config() == loaded_layer.get_config()
        self.assertAllEqual(model_pred, loaded_pred)


if __name__ == '__main__':
    tf.test.main()
