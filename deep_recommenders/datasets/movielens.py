#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import random
import tensorflow as tf


def _download_and_unzip(filename="ml-1m.zip"):
    import requests
    import zipfile
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)
    f = zipfile.ZipFile(filename)
    f.extractall()


def _data_shard(filename, num_shards=4):
    cmd = "wc -l < {}".format(filename)
    cmd_res = os.popen(cmd)
    total_lines = int(cmd_res.read().strip())
    block_lines = total_lines // num_shards
    num_lines, num_shard = 0, 0
    with open(filename, "r", encoding="unicode_escape") as f:
        for line in f:
            if num_lines % block_lines == 0:
                if num_shard < num_shards:
                    _f = open(filename+str(num_shard), "w")
                num_shard += 1
            _f.write(line)
            num_lines += 1


def _shuffle_data(filename):
    shuffled_filename = f"{filename}.shuffled"
    with open(filename, "r") as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open(shuffled_filename, "w") as f:
        f.writelines(lines)
    return shuffled_filename


def _load_data(filename, columns):
    data = {}
    with open(filename, "r", encoding="unicode_escape") as f:
        for line in f:
            ls = line.strip("\n").split("::")
            data[ls[0]] = dict(zip(columns[1:], ls[1:]))
    return data


def _serialize_example(feature):
    serialize_feature = {}
    for c in ["Age", "Occupation", "Rating", "Timestamp"]:
        serialize_feature[c] = tf.train.Feature(int64_list=tf.train.Int64List(value=[feature[c]]))
    for c in ["UserID", "MovieID", "Gender", "Zip-code", "Title"]:
        serialize_feature[c] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature[c]]))
    serialize_feature["Genres"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=feature["Genres"]))
    example_proto = tf.train.Example(features=tf.train.Features(feature=serialize_feature))
    return example_proto.SerializeToString()


def serialize_tfrecords(tfrecords_fn, datadir="ml-1m", download=False):

    if download is True:
        print("Downloading MovieLens-1M dataset ...")
        _download_and_unzip(datadir + ".zip")

    users_data = _load_data(datadir + "/users.dat",
                            columns=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
    movies_data = _load_data(datadir + "/movies.dat",
                             columns=["MovieID", "Title", "Genres"])

    ratings_columns = ["UserID", "MovieID", "Rating", "Timestamp"]
    with tf.io.TFRecordWriter(tfrecords_fn) as writer:
        shuffled_filename = _shuffle_data(datadir + "/ratings.dat")
        with open(shuffled_filename, "r", encoding="unicode_escape") as f:
            for line in f:
                ls = line.strip().split("::")
                rating = dict(zip(ratings_columns, ls))
                rating.update(users_data.get(ls[0]))
                rating.update(movies_data.get(ls[1]))
                for c in ["Age", "Occupation", "Rating", "Timestamp"]:
                    rating[c] = int(rating[c])
                for c in ["UserID", "MovieID", "Gender", "Zip-code", "Title"]:
                    rating[c] = rating[c].encode("utf-8")
                rating["Genres"] = [x.encode("utf-8") for x in rating["Genres"].split("|")]
                serialized = _serialize_example(rating)
                writer.write(serialized)


class MovieLens(object):

    def __init__(self, filename="movielens.tfrecords"):
        self._filename = os.path.join(os.path.dirname(__file__), filename)
        self._columns = ["UserID", "MovieID", "Rating", "Timestamp",
                         "Gender", "Age", "Occupation", "Zip-code",
                         "Title", "Genres"]
        self.num_ratings = 1000209
        self.num_users = 6040
        self.num_movies = 3952
        self.gender_vocab = ["F", "M"]
        self.age_vocab = [1, 18, 25, 35, 45, 50, 56]
        self.occupation_vocab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.genres_vocab = ["Action", "Adventure", "Animation", "Children's", "Comedy",
                             "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                             "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

    def dataset(self, epochs=1, batch_size=256):

        def _parse_example(serialized_example):
            features = {}
            for c in ["Age", "Occupation", "Rating", "Timestamp"]:
                features[c] = tf.io.FixedLenFeature([], tf.int64)
            for c in ["UserID", "MovieID", "Gender", "Zip-code", "Title"]:
                features[c] = tf.io.FixedLenFeature([], tf.string)
            features["Genres"] = tf.io.VarLenFeature(tf.string)
            example = tf.io.parse_example(serialized_example, features)
            ratings = example.pop("Rating")
            return example, ratings

        ds = tf.data.TFRecordDataset(self._filename)
        ds = ds.repeat(epochs)
        ds = ds.batch(batch_size)
        ds = ds.map(_parse_example, num_parallel_calls=-1)
        return ds


class MovielensRanking(MovieLens):

    def __init__(self,
                 epochs: int = 10,
                 batch_size: int = 1024,
                 buffer_size: int = 1024,
                 train_size: float = 0.8,
                 *args, **kwargs):
        super(MovielensRanking, self).__init__(*args, **kwargs)
        self._epochs = epochs
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._train_size = train_size

    @property
    def train_steps(self):
        num_train_ratings = self.num_ratings * self._epochs * self._train_size
        return int(num_train_ratings // self._batch_size)

    @property
    def train_steps_per_epoch(self):
        num_train_ratings = self.num_ratings * self._train_size
        return int(num_train_ratings // self._batch_size)

    @property
    def test_steps(self):
        return self.num_ratings // self._batch_size - self.train_steps_per_epoch

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


if __name__ == '__main__':
    serialize_tfrecords("movielens.tfrecords", download=True)

