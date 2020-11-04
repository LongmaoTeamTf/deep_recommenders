"""
@Description: Sampling-Bias-Corrected Neural Model
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-08-26 20:47:47
@LastEditors: Wang Yao
@LastEditTime: 2020-11-04 18:50:55
"""
import os
import time
import math
import pathlib
import functools
import numpy as np
import tensorflow as tf

from modeling import build_model


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
        tf.math.multiply(tf.strings.to_number(labels), 
        tf.constant([0.5, 0.2, 0.1, 0.2, 0.]))) # 点击、点赞、分享、收藏、评论

    return left_features, right_features, weight_labels


def get_dataset_from_csv_files(filenames,
                               left_columns,
                               right_columns,
                               csv_header,
                               batch_size=256,
                               epochs=None,
                               shuffle_size=None):
    """消费csv文件列表"""
    list_ds = tf.data.Dataset.list_files(filenames)
    dataset = list_ds.interleave(
        lambda fp: tf.data.TextLineDataset(fp).skip(1),
        cycle_length=2,
        block_length=batch_size*2,
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
    dataset = dataset.cache()
    if epochs is not None:
        dataset = dataset.repeat(epochs)
    if shuffle_size is not None:
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
    """取余Hash函数"""
    if tf.keras.backend.dtype(ids) != 'int32':
        ids = tf.cast(ids, dtype=tf.int32)
    return tf.math.mod(ids, hash_bucket_size)


def log_q(x, y, sampling_p=None, temperature=0.05):
    """logQ correction used in sampled softmax model."""
    inner_product = tf.reduce_sum(tf.math.multiply(x, y), axis=-1) / temperature
    if sampling_p is not None:
        return inner_product - tf.math.log(sampling_p)
    return inner_product


def corrected_batch_softmax(x, y, sampling_p=None):
    """logQ correction softmax"""
    correct_inner_product = log_q(x, y, sampling_p=sampling_p)
    return tf.math.exp(correct_inner_product) / tf.math.reduce_sum(tf.math.exp(correct_inner_product))
    

def reward_cross_entropy(reward, output):
    """Reward correction batch """
    return -tf.reduce_mean(reward * tf.math.log(output))


def topk_recall(output, reward, k=10):
    """TopK Recall rate"""
    _, indices = tf.math.top_k(output, k=k)

    def _ture(reward, indices):
        return tf.math.count_nonzero(tf.gather(reward, indices)) / tf.math.count_nonzero(reward)
    
    def _false():
        return tf.constant(0., dtype=tf.float64)

    return tf.cond(tf.math.count_nonzero(reward) > 0, lambda: _ture(reward, indices), lambda: _false())


def topk_positive(output, reward, k=10):
    """Topk Positive rate"""
    _, indices = tf.math.top_k(output, k=k)

    def _ture(reward, indices):
        return tf.math.count_nonzero(tf.gather(reward, indices)) / k
    
    def _false():
        return tf.constant(0., dtype=tf.float64)

    return tf.cond(tf.math.count_nonzero(reward) > 0, lambda: _ture(reward, indices), lambda: _false())
    

def custom_train(strategy,
                dataset, 
                steps,
                epochs,
                ids_column,
                ids_hash_bucket_size,
                tensorboard_dir=None,
                checkpoints_dir=None,
                streaming=False,
                beta=100,
                lr=0.001):
    """自定义训练"""

    dataset = strategy.experimental_distribute_dataset(dataset)

    with strategy.scope():
        left_model, right_model = build_model()

        def pred(left_x, right_x, sampling_p):
            left_y_ = left_model(left_x, training=True)
            right_y_ = right_model(right_x, training=True)
            output = corrected_batch_softmax(left_y_, right_y_, sampling_p=sampling_p)
            return output

        def loss(left_x, right_x, sampling_p, reward):
            output = pred(left_x, right_x, sampling_p)
            return reward_cross_entropy(reward, output)

        def grad(left_x, right_x, sampling_p, reward):
            with tf.GradientTape(persistent=True) as tape:
                loss_value = loss(left_x, right_x, sampling_p, reward)
            left_grads = tape.gradient(loss_value, left_model.trainable_variables)
            right_grads = tape.gradient(loss_value, right_model.trainable_variables)
            return loss_value, left_grads, right_grads

        epoch_recall_avg = tf.keras.metrics.Mean()
        epoch_positive_avg = tf.keras.metrics.Mean()

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        left_checkpointer = tf.train.Checkpoint(optimizer=optimizer, model=left_model)
        right_checkpointer = tf.train.Checkpoint(optimizer=optimizer, model=right_model)

        left_checkpoint_prefix = os.path.join(checkpoints_dir, "left-ckpt")
        right_checkpoint_prefix = os.path.join(checkpoints_dir, "right-ckpt")

        def train_step(inputs, sampling_p):
            left_x, right_x, reward = inputs
            loss_value, left_grads, right_grads = grad(left_x, right_x, sampling_p, reward)
            optimizer.apply_gradients(zip(left_grads, left_model.trainable_variables))
            optimizer.apply_gradients(zip(right_grads, right_model.trainable_variables))

            epoch_recall_avg.update_state(topk_recall(pred(left_x, right_x, sampling_p), reward))
            epoch_positive_avg.update_state(topk_positive(pred(left_x, right_x, sampling_p), reward))

            return loss_value

        @tf.function
        def distributed_train_step(inputs, sampling_p=None):
            per_replica_losses = strategy.run(train_step, args=(inputs, sampling_p,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
            
        loss_results = []
        recall_results = []
        positive_results = []
        
        if tensorboard_dir is not None:
            summary_writer = tf.summary.create_file_writer(tensorboard_dir)

        print("Start Traning ... ")
        for epoch in range(epochs):
            if streaming is True:
                array_a = np.zeros(shape=(ids_hash_bucket_size,), dtype=np.float32)
                array_b = np.ones(shape=(ids_hash_bucket_size,), dtype=np.float32) * beta
            total_loss = 0.0
            batches_train_time = 0.0
            batches_load_data_time = 0.0
            epoch_trian_time = 0.0
            epoch_load_data_time = 0.0
            step = 1
            
            batch_load_data_start = time.time()
            for inputs in dataset:
                batch_load_data_stop = time.time()
                if streaming is True:
                    cand_ids = inputs[1].get(ids_column)
                    cand_hash_indexs = hash_simple(cand_ids, ids_hash_bucket_size)
                    array_a, array_b, sampling_p = sampling_p_estimation_single_hash(array_a, array_b, cand_hash_indexs, step)
                else:
                    sampling_p = None
                
                batch_train_start = time.time()
                total_loss += distributed_train_step(inputs, sampling_p=sampling_p)
                # total_loss += train_step(inputs, sampling_p=sampling_p)
                batch_train_stop = time.time()
                batch_train_time = batch_train_stop - batch_train_start
                
                batches_train_time += batch_train_time
                epoch_trian_time += batch_train_time

                batch_load_data_time = batch_load_data_stop - batch_load_data_start
                batches_load_data_time += batch_load_data_time
                epoch_load_data_time += batch_load_data_time
                
                if step % 50 == 0:
                    print("Epoch[{}/{}]: Batch({}/{}) "
                            "DataSpeed: {:.4f}sec/batch "
                            "TrainSpeed: {:.4f}sec/batch "
                            "correct_sfx_loss={:.4f} "
                            "topk_recall={:.4f} "
                            "topk_positive={:.4f}".format(
                            epoch+1, epochs, step, steps,
                            batches_load_data_time/50,
                            batches_train_time/50,
                            total_loss/step, 
                            epoch_recall_avg.result(), 
                            epoch_positive_avg.result()))
                    batches_train_time = 0.0
                    batches_load_data_time = 0.0
                step += 1
                batch_load_data_start = time.time()

            # optimizer.lr = 0.1 * optimizer.lr

            loss_results.append(total_loss/steps)
            recall_results.append(epoch_recall_avg.result())
            positive_results.append(epoch_positive_avg.result())
        
            print("Epoch[{}/{}]: correct_sfx_loss={:.4f} topk_recall={:.4f} topk_positive={:.4f}".format(
                    epoch+1, epochs, total_loss/step, epoch_recall_avg.result(), epoch_positive_avg.result()))
            print("Epoch[{}/{}]: Train time: {:.4f}".format(epoch+1, epochs, epoch_trian_time))
            print("Epoch[{}/{}]: Load data time: {:.4f}".format(epoch+1, epochs, epoch_load_data_time))
            
            if tensorboard_dir is not None:
                with summary_writer.as_default(): # pylint: disable=not-context-manager
                    tf.summary.scalar('correct_sfx_loss', total_loss/steps, step=epoch)
                    tf.summary.scalar('topk_recall', epoch_recall_avg.result(), step=epoch)
                    tf.summary.scalar('topk_positive', epoch_positive_avg.result(), step=epoch)

            if (epoch+1) % 2 == 0:
                left_checkpointer.save(left_checkpoint_prefix)
                print(f'Saved checkpoints to: {left_checkpoint_prefix}')
                right_checkpointer.save(right_checkpoint_prefix)
                print(f'Saved checkpoints to: {right_checkpoint_prefix}')

            epoch_recall_avg.reset_states()
            epoch_positive_avg.reset_states()
                
    return left_model, right_model


def distribute_train_model(dataset_config, train_config):
    """分布式训练模型"""

    data_dir = dataset_config.get('data_dir')
    data_dir = pathlib.Path(data_dir)
    filenames = sorted([str(fn) for fn in data_dir.glob("*.csv")])[-7:]
    print("Train filenames: ")
    for fn in filenames:
        print(fn)
    global_batch_size = dataset_config.get('batch_size') * train_config.get('workers_num')

    version = train_config.get('version')
    tensorboard_dir = os.path.join(train_config.get('tensorboard_dir'), version)
    checkpoints_dir = os.path.join(train_config.get('checkpoints_dir'), version)
    query_saved_path = os.path.join(train_config.get('query_saved_path'), version)
    candidate_saved_path = os.path.join(train_config.get('candidate_saved_path'), version)
    
    strategy = tf.distribute.MirroredStrategy()

    train_dataset = get_dataset_from_csv_files(
        filenames, 
        dataset_config.get('query_columns'), 
        dataset_config.get('candidate_columns'),
        dataset_config.get('csv_header'), 
        batch_size=global_batch_size
    )

    def _get_steps(fns, batch_size, skip_header=True):
        """获取数据集迭代步数"""
        _total_num = 0
        for fn in fns:
            cmd = "wc -l < {}".format(fn)
            cmd_res = os.popen(cmd)
            _num_lines = int(cmd_res.read().strip())
            if skip_header is True:
                _num_lines -= 1
            _total_num += _num_lines
        _steps = math.ceil(_total_num / batch_size)
        return _steps

    train_steps = _get_steps(filenames, global_batch_size)
    
    query_model, candidate_model = custom_train(
        strategy,
        train_dataset, 
        train_steps,
        epochs=train_config.get('epochs'),
        ids_column=train_config.get('ids_column'),
        ids_hash_bucket_size=train_config.get('ids_hash_bucket_size'),
        tensorboard_dir=tensorboard_dir,
        checkpoints_dir=checkpoints_dir
    )
    
    print("Saving query model ... ")
    query_model.save(query_saved_path)
    print("Query model saved.")
    print("Saving candidate model ... ")
    candidate_model.save(candidate_saved_path)
    print("Candidate model saved.")


if __name__ == "__main__":

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    query_columns = [
        'past_watches',
        'seed_id',
        'seed_category',
        'seed_tags',
        'seed_gap_time',
        'seed_duration_time',
        'seed_play_count',
        'seed_like_count',
        'seed_share_count',
        'seed_collect_count'
    ]
    candidate_columns = [
        'cand_id',
        'cand_category',
        'cand_tags',
        'cand_gap_time',
        'cand_duration_time',
        'cand_play_count',
        'cand_like_count',
        'cand_share_count',
        'cand_collect_count'
    ]
    csv_header = [
        'label',
        'udid',
        'past_watches',
        'seed_id',
        'seed_category',
        'seed_tags',
        'seed_gap_time',
        'seed_duration_time',
        'seed_play_count',
        'seed_like_count',
        'seed_share_count',
        'seed_collect_count',
        'cand_id',
        'cand_category',
        'cand_tags',
        'cand_gap_time',
        'cand_duration_time',
        'cand_play_count',
        'cand_like_count',
        'cand_share_count',
        'cand_collect_count'
    ]

    dataset_config = {
        'data_dir': '/home/xddz/data/two_tower_data',
        'batch_size': 512,
        'query_columns': query_columns,
        'candidate_columns': candidate_columns,
        'csv_header': csv_header
    }
    
    train_config = {
        'workers_num': 2,
        'epochs': 10,
        'ids_column': 'cand_id',
        'ids_hash_bucket_size': 200000,
        'version': '20200928',
        'tensorboard_dir': '/home/xddz/data/two_tower_data/model/training_tensorboard',
        'checkpoints_dir': '/home/xddz/data/two_tower_data/model/training_checkpoints',
        'query_saved_path': '/home/xddz/data/two_tower_data/model/models/google_tt_query',
        'candidate_saved_path': '/home/xddz/data/two_tower_data/model/models/google_tt_candidate'
    }

    distribute_train_model(
        dataset_config,
        train_config
    )
    