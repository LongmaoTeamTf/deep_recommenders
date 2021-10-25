#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from deep_recommenders.datasets.product import ProductDataset
from deep_recommenders.estimator.models.multi_task_learning import ESMM


def build_columns():
    carrier = tf.feature_column.categorical_column_with_hash_bucket(
        'carrier', hash_bucket_size=150)
    manufacturer = tf.feature_column.categorical_column_with_hash_bucket(
        'manufacturer', hash_bucket_size=150)
    model = tf.feature_column.categorical_column_with_hash_bucket(
        'model', hash_bucket_size=1500)
    os = tf.feature_column.categorical_column_with_vocabulary_list(
        'os', ['Android', 'iOS'])
    network_type = tf.feature_column.categorical_column_with_vocabulary_list(
        'network_type', ['2G', '3G', '4G', '5G', 'WIFI'])
    wifi = tf.feature_column.categorical_column_with_vocabulary_list(
        'wifi', ['false', 'true'])
    country = tf.feature_column.categorical_column_with_hash_bucket(
        'country', hash_bucket_size=100)
    province = tf.feature_column.categorical_column_with_hash_bucket(
        'province', hash_bucket_size=250)
    city = tf.feature_column.categorical_column_with_hash_bucket(
        'city', hash_bucket_size=1000)
    is_first_day = tf.feature_column.categorical_column_with_vocabulary_list(
        'is_first_day', ['false', 'true'])

    user_gender = tf.feature_column.categorical_column_with_vocabulary_list(
        "user_gender", ["female", "male"])
    user_country = tf.feature_column.categorical_column_with_hash_bucket(
        "user_country", hash_bucket_size=100)
    user_city = tf.feature_column.categorical_column_with_hash_bucket(
        "user_city", hash_bucket_size=1000)
    user_university = tf.feature_column.categorical_column_with_hash_bucket(
        "user_university", hash_bucket_size=500)
    user_job = tf.feature_column.categorical_column_with_hash_bucket(
        "user_job", hash_bucket_size=26)

    video_category_id = tf.feature_column.categorical_column_with_hash_bucket(
        'video_category_id', hash_bucket_size=18)
    video_author_id = tf.feature_column.categorical_column_with_hash_bucket(
        'video_author_id', hash_bucket_size=5000)
    video_duration = tf.feature_column.numeric_column('video_duration')
    video_duration_bucket = tf.feature_column.bucketized_column(
        video_duration, boundaries=[minute * 60 for minute in range(0, 11)])

    base_columns = [
        carrier,
        manufacturer,
        model,
        os,
        network_type,
        wifi,
        country,
        province,
        city,
        is_first_day,
        user_gender,
        user_country,
        user_city,
        user_university,
        user_job,
        video_category_id,
        video_author_id,
        video_duration_bucket
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            ['user_gender', 'video_category_id'], hash_bucket_size=30),
        tf.feature_column.crossed_column(
            ['user_job', 'video_category_id'], hash_bucket_size=300),
    ]

    embedding_columns = [
        tf.feature_column.embedding_column(c, dimension=32)
        for c in base_columns + crossed_columns
    ]
    return embedding_columns


def model_fn(features, labels, mode):
    columns = build_columns()
    p_cpr, p_ctr, p_ctcvr = ESMM(columns, [256, 64, 10])(features)

    predictions = {
        "p_ctr": p_ctr,
        "p_cpr": p_cpr
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    true_ctr = tf.expand_dims(labels["ctr"], axis=1)
    true_cpr = tf.expand_dims(labels["cpr"], axis=1)

    true_ctcpr = tf.multiply(true_ctr, true_cpr)

    loss_ctr = tf.losses.log_loss(labels=true_ctr, predictions=p_ctr)
    loss_ctcpr = tf.losses.log_loss(labels=true_ctcpr, predictions=p_ctcvr)

    total_loss = loss_ctr + loss_ctcpr

    tf.summary.scalar("loss_ctr", loss_ctr)
    tf.summary.scalar("loss_ctcpr", loss_ctcpr)
    tf.summary.scalar("total_loss", total_loss)

    metrics = {
        "auc_ctr": tf.metrics.auc(true_ctr, p_ctr),
        "auc_ctcpr": tf.metrics.auc(true_ctcpr, p_ctcvr)
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    train_op = tf.group(
        optimizer.minimize(loss=loss_ctr, global_step=tf.train.get_global_step()),
        optimizer.minimize(loss=loss_ctcpr, global_step=tf.train.get_global_step()),
    )

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def build_estimator(model_dir=None, inter_op=8, intra_op=8):

    config_proto = tf.ConfigProto(device_count={'GPU': 0},
                                  inter_op_parallelism_threads=inter_op,
                                  intra_op_parallelism_threads=intra_op)

    run_config = tf.estimator.RunConfig().replace(
        tf_random_seed=42,
        keep_checkpoint_max=10,
        save_checkpoints_steps=200,
        log_step_count_steps=100,
        session_config=config_proto)

    return tf.estimator.Estimator(model_fn=model_fn,
                                  model_dir=model_dir,
                                  config=run_config)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = build_estimator()
    early_stop_hook = tf.estimator.experimental.stop_if_no_decrease_hook(estimator, "loss", 1000)

    product = ProductDataset("multi_task_learning/train",
                             "multi_task_learning/valid")
    train_spec = tf.estimator.TrainSpec(lambda: product.train_input_fn(epochs=10, batch_size=1024),
                                        max_steps=None,
                                        hooks=[early_stop_hook])
    eval_spec = tf.estimator.EvalSpec(lambda: product.valid_input_fn(),
                                      steps=None,
                                      start_delay_secs=60,
                                      throttle_secs=60)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()


