"""
@Description: Sampling-Bias-Corrected Neural Model Training Demo
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-09-03 16:26:18
@LastEditors: Wang Yao
@LastEditTime: 2020-09-21 14:13:14
"""
import os
import sys
sys.path.append("../..")
import math
import tensorflow as tf

from src.embedding.google_tt.modeling import build_model
from src.embedding.google_tt.train import get_dataset_from_csv_files
from src.embedding.google_tt.train import train_model


left_columns = [
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
right_columns = [
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


def distribute_train_model():
    def _get_steps(fns, batch_size, skip_header=True):
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

    filenames = [
        '/home/xddz/data/two_tower_data/2020-08-31.csv',
        '/home/xddz/data/two_tower_data/2020-09-01.csv',
        '/home/xddz/data/two_tower_data/2020-09-02.csv',
        '/home/xddz/data/two_tower_data/2020-09-03.csv',
        '/home/xddz/data/two_tower_data/2020-09-04.csv',
        '/home/xddz/data/two_tower_data/2020-09-05.csv',
        '/home/xddz/data/two_tower_data/2020-09-06.csv' 
    ]
    batch_size = 256 * 2
    epochs = 10
    steps = _get_steps(filenames, batch_size)
    ids_column = 'cand_id'
    ids_hash_bucket_size=100000
    version = '20200921'
    tensorboard_dir = f'./training_tensorboard/{version}'
    checkpoint_dir = f'./training_checkpoints/{version}'
    strategy = tf.distribute.MirroredStrategy()

    train_dataset = get_dataset_from_csv_files(
        filenames, 
        left_columns, 
        right_columns,
        csv_header, 
        batch_size=batch_size
    )
    
    left_model, right_model = train_model(
        strategy,
        train_dataset, 
        steps,
        epochs=epochs,
        ids_column=ids_column,
        ids_hash_bucket_size=ids_hash_bucket_size,
        tensorboard_dir=tensorboard_dir,
        checkpoint_dir=checkpoint_dir
    )
    
    left_model.save(f'./models/google_tt_query/{version}')
    right_model.save(f'./models/google_tt_candidate/{version}')


if __name__ == "__main__":
    distribute_train_model()