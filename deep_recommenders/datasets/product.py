#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf


class ProductDataset(object):

    def __init__(self, train_dir, valid_dir):
        self._train_dir = train_dir
        self._valid_dir = valid_dir
        self._columns = [
            "event", "action_date", "action_time", "account_id", "udid", "video_id",
            "carrier", "manufacturer", "model", "os", "network_type", "wifi",
            "country", "province", "city", "is_first_day", "play_time", "play_rate",
            "user_name", "user_nickname", "user_avatar", "user_type", "user_gender",
            "user_register_date", "user_description", "user_cover", "user_birthday",
            "user_country", "user_city", "user_university", "user_job",
            "video_category_id", "video_author_id", "video_duration",
            "video_history_play_count", "video_history_like_count", "video_history_collect_count",
            "video_history_share_count", "video_history_reply_count",
        ]

    @property
    def record_defaults(self):
        return [
            self.string, self.string, self.string, self.string, self.string, self.string,
            self.string, self.string, self.string, self.string, self.string, self.string,
            self.string, self.string, self.string, self.string, self.int32, self.float32,
            self.string, self.string, self.string, self.string, self.string,
            self.string, self.string, self.string, self.string,
            self.string, self.string, self.string, self.string,
            self.string, self.string, self.float32,
            self.float32, self.float32, self.float32, self.float32, self.float32
        ]

    @property
    def string(self):
        return tf.constant("", dtype=tf.string)

    @property
    def float32(self):
        return tf.constant(0, dtype=tf.float32)

    @property
    def int32(self):
        return tf.constant(0, dtype=tf.int32)

    def build(self, data_dir, epochs=1, batch_size=512, buffer_size=512):

        def _parse_example(example):
            parsed_example = tf.io.decode_csv(example,
                                              self.record_defaults,
                                              na_value="null")
            features = dict(zip(self._columns, parsed_example))
            event = features.pop('event')
            ctr = tf.cast(tf.equal(event, "click_play"), tf.float32)
            cpr = features.pop('play_rate')
            return features, {"ctr": ctr, "cpr": cpr}

        ls = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)]
        dataset = tf.data.TextLineDataset(ls, num_parallel_reads=-1)
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(_parse_example, num_parallel_calls=-1)
        dataset = dataset.prefetch(buffer_size)
        return dataset

    def train_input_fn(self, epochs=1, batch_size=512, buffer_size=512):
        return self.build(self._train_dir, epochs, batch_size, buffer_size)

    def valid_input_fn(self, epochs=1, batch_size=512, buffer_size=512):
        return self.build(self._valid_dir, epochs, batch_size, buffer_size)
