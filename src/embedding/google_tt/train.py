"""
@Description: Sampling-Bias-Corrected Neural Model
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-08-26 20:47:47
@LastEditors: Wang Yao
@LastEditTime: 2020-09-07 10:43:36
"""
import functools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def parse_csv_line(left_columns,
                   right_columns,
                   csv_header,
                   csv_line):
    """解析csv数据"""
    
    def _parse_columns(columns):
        columns_defs = []
        columns_indexs = []
        for col in columns:
            columns_indexs.append(csv_header.index(col))
            if "time" in col or "count" in col:
                columns_defs.append(tf.constant(-1, dtype=tf.int32))
            else:
                columns_defs.append(tf.constant(""))
    
        parsed_columns = tf.io.decode_csv(
            csv_line,
            record_defaults=columns_defs,
            select_cols=columns_indexs,
            use_quote_delim=False
        )
        return parsed_columns

    def _parse_multi_hot(tensor, max_length=10, duplicate=True, delim='_'):
        """Multi-hot"""
        vals = tensor.numpy().decode('utf-8').strip(delim).split(delim)
        if duplicate is True:
            vals = list(set(vals))
        if len(vals) < max_length:
            vals = vals + ['']* (max_length-len(vals))
        if len(vals) > max_length:
            vals = vals[: max_length]
        return tf.constant(vals)

    left_parsed_columns = _parse_columns(left_columns)
    right_parsed_columns = _parse_columns(right_columns)
    labels = _parse_columns(['label'])
    
    left_features = dict(zip(left_columns, left_parsed_columns))
    right_features = dict(zip(right_columns, right_parsed_columns))
    left_features['past_watches'] = tf.py_function(
        func=functools.partial(_parse_multi_hot, max_length=10, duplicate=False), 
        inp=[left_features['past_watches']], 
        Tout=[tf.string])[0]
    left_features['seed_tags'] = tf.py_function(
        func=functools.partial(_parse_multi_hot, max_length=5, duplicate=True), 
        inp=[left_features['seed_tags']], 
        Tout=[tf.string])[0]
    right_features['cand_tags'] = tf.py_function(
        func=functools.partial(_parse_multi_hot, max_length=5, duplicate=True), 
        inp=[right_features['cand_tags']], 
        Tout=[tf.string])[0]
    labels = tf.py_function(
        func=functools.partial(_parse_multi_hot, max_length=5, duplicate=False), 
        inp=[labels[0]], 
        Tout=[tf.string])
    weight_labels = tf.math.reduce_sum(
        tf.math.multiply(tf.strings.to_number(labels), tf.constant([0.1, 0.2, 0.3, 0.2, 0.2])))

    return left_features, right_features, weight_labels


def get_dataset_from_csv_files(filenames,
                               left_columns,
                               right_columns,
                               csv_header,
                               batch_size=256,
                               epochs=None,
                               shuffle_size=500):
    """消费csv文件列表"""
    list_ds = tf.data.Dataset.list_files(filenames)
    dataset = list_ds.interleave(
        lambda fp: tf.data.TextLineDataset(fp).skip(1),
        cycle_length=2,
        block_length=batch_size,
        num_parallel_calls=2
    )
    dataset = dataset.map(
        map_func=functools.partial(
            parse_csv_line, 
            left_columns,
            right_columns,
            csv_header),
        num_parallel_calls=2
    )
    if epochs is not None:
        dataset = dataset.repeat(epochs)
    if shuffle_size is not None:
        print("Dataset shuffle size: {}".format(shuffle_size))
        dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset


def sampling_p_estimation_single_hash(array_a, array_b, hash_indexs, global_step, alpha=0.01):
    """单Hash函数采样概率估计"""
    array_b[hash_indexs] = (1 - alpha) * array_b[hash_indexs] + \
                                alpha * (global_step - array_a[hash_indexs])
    array_a[hash_indexs] = global_step
    sampling_p = 1 / array_b[hash_indexs]
    return array_a, array_b, sampling_p


def hash_simple(ids, hash_bucket_size):
    """直接取整Hash函数"""
    hash_indexs = []
    for id in ids:
        hash_indexs.append(int(id) % hash_bucket_size)
    return hash_indexs


@tf.function
def log_q(x, y, sampling_p, temperature=0.05):
    """logQ correction used in sampled softmax model."""
    inner_product = tf.reduce_sum(tf.math.multiply(x, y)) / temperature
    return inner_product - tf.math.log(sampling_p)


@tf.function
def corrected_batch_softmax(x, y, sampling_p):
    """logQ correction softmax"""
    correct_inner_product = log_q(x, y, sampling_p)
    return tf.math.exp(correct_inner_product) / tf.math.reduce_sum(tf.math.exp(correct_inner_product))


@tf.function
def reward_cross_entropy(reward, output):
    """Reward correction batch """
    return -tf.reduce_mean(reward * tf.math.log(output))


def custom_train_model(left_model, 
                       right_model, 
                       train_dataset, 
                       train_steps,
                       valid_dataset,
                       valid_steps,
                       num_epochs,
                       ids_column,
                       ids_hash_bucket_size,
                       beta=100,
                       lr=0.01):
    """自定义训练"""

    @tf.function
    def loss(left_x, right_x, sampling_p, reward):
        left_y_ = left_model(left_x)
        right_y_ = right_model(right_x)
        y_ = corrected_batch_softmax(left_y_, right_y_, sampling_p)
        return reward_cross_entropy(reward, y_)

    @tf.function
    def grad(left_x, right_x, sampling_p, reward):
        with tf.GradientTape(persistent=True) as tape:
            loss_value = loss(left_x, right_x, sampling_p, reward)
        left_grads = tape.gradient(loss_value, left_model.trainable_variables)
        right_grads = tape.gradient(loss_value, right_model.trainable_variables)
        return loss_value, left_grads, right_grads

    optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr)
    
    array_a = np.zeros(shape=(ids_hash_bucket_size,), dtype=np.float32)
    array_b = np.ones(shape=(ids_hash_bucket_size,), dtype=np.float32) * beta
    train_loss_results = []
    valid_loss_results = []
    
    # summary_writer = tf.summary.create_file_writer('./logs')

    for epoch in range(num_epochs):
        epoch_train_loss_avg = tf.keras.metrics.Mean()
        train_step = 0
        for left_x, right_x, reward in train_dataset:
            cand_ids = right_x.get(ids_column)
            cand_hash_indexs = hash_simple(cand_ids, ids_hash_bucket_size)
            array_a, array_b, sampling_p = sampling_p_estimation_single_hash(array_a, array_b, cand_hash_indexs, train_step)
            
            loss_value, left_grads, right_grads = grad(left_x, right_x, sampling_p, reward)
            optimizer.apply_gradients(zip(left_grads, left_model.trainable_variables))
            optimizer.apply_gradients(zip(right_grads, right_model.trainable_variables))

            epoch_train_loss_avg(loss_value)
            train_step += 1
            progress = '='*int(train_step/train_steps*50)+'>'+' '*(50-int(train_step/train_steps*50))
            print("\rEpoch {:03d}/{:03d}: Train - {}/{} [{}] correct-sfx: {:.3f}".format(
                epoch+1, num_epochs, train_step, train_steps, progress, loss_value), end='', flush=True)

        train_loss_results.append(epoch_train_loss_avg.result())
    
        print("\nEpoch {:03d}/{:03d}: train-epoch-avg-correct-sfx: {:.3f}".format(
            epoch+1, num_epochs, epoch_train_loss_avg.result()))
        
        epoch_valid_loss_avg = tf.keras.metrics.Mean()
        valid_step = 0
        for left_x, right_x, reward in valid_dataset:
            cand_ids = right_x.get(ids_column)
            cand_hash_indexs = hash_simple(cand_ids, ids_hash_bucket_size)
            sampling_p = 1 / array_b[cand_hash_indexs]
            loss_value, left_grads, right_grads = grad(left_x, right_x, sampling_p, reward)
            epoch_valid_loss_avg(loss_value)
            valid_step += 1
            progress = '='*int(valid_step/valid_steps*20)+'>'+' '*(20-int(valid_step/valid_steps*20))
            print("\rEpoch {:03d}/{:03d}: Valid - {}/{} [{}] correct-sfx: {:.3f}".format(
                epoch+1, num_epochs, valid_step, valid_steps, progress, loss_value), end='', flush=True)

        valid_loss_results.append(epoch_valid_loss_avg.result())
    
        print("\nEpoch {:03d}/{:03d}: valid-epoch-avg-correct-sfx: {:.3f}".format(
            epoch+1, num_epochs, epoch_valid_loss_avg.result()))

        # with summary_writer.as_default(): # pylint: disable=not-context-manager
        #     tf.summary.scalar('train_loss', epoch_loss_avg.result(), step=epoch)
        
    return left_model, right_model


def plot_metric(metric_results, metric_name):
    """metrics变化曲线"""
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.plot(metric_results)
    plt.show()

    