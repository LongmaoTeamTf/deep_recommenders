"""
@Description: Sampling-Bias-Corrected Neural Model Training Demo
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-09-03 16:26:18
@LastEditors: Wang Yao
@LastEditTime: 2020-09-23 18:54:22
"""
import os
import sys
sys.path.append("../..")
import math
import pathlib
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from src.embedding.google_tt.modeling import build_model
from src.embedding.google_tt.train import get_dataset_from_csv_files
from src.embedding.google_tt.train import train_model


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


def distribute_train_model(dataset_config, train_config):
    """分布式训练模型"""

    data_dir = dataset_config.get('data_dir')
    data_dir = pathlib.Path(data_dir)
    filenames = [str(fn) for fn in data_dir.glob("*.csv")]
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
    train_steps = _get_steps(filenames, global_batch_size)
    
    query_model, candidate_model = train_model(
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
        'batch_size': 192,
        'query_columns': query_columns,
        'candidate_columns': candidate_columns,
        'csv_header': csv_header
    }
    
    train_config = {
        'workers_num': 2,
        'epochs': 10,
        'ids_column': 'cand_id',
        'ids_hash_bucket_size': 100000,
        'version': '20200922',
        'tensorboard_dir': './training_tensorboard',
        'checkpoints_dir': './training_checkpoints',
        'query_saved_path': './models/google_tt_query',
        'candidate_saved_path': './models/google_tt_candidate'
    }

    distribute_train_model(
        dataset_config,
        train_config
    )