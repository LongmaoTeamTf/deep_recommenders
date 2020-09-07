"""
@Description: Sampling-Bias-Corrected Neural Model Training Demo
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-09-03 16:26:18
@LastEditors: Wang Yao
@LastEditTime: 2020-09-07 10:51:42
"""
import sys
sys.path.append('../..')
import math
from src.embedding.google_tt.modeling import build_model
from src.embedding.google_tt.train import custom_train_model
from src.embedding.google_tt.train import get_dataset_from_csv_files



if __name__ == "__main__":
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
        'seed_collect_count',
        # 'seed_reply_count',   
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
        'cand_collect_count',
        # 'cand_reply_count',
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
        # 'seed_reply_count',
        'cand_id',
        'cand_category',
        'cand_tags',
        'cand_gap_time',
        'cand_duration_time',
        'cand_play_count',
        'cand_like_count',
        'cand_share_count',
        'cand_collect_count',
        # 'cand_reply_count',
    ]
    
    train_dataset = get_dataset_from_csv_files(
        [
            '/home/xddz/data/two_tower_data/2020-09-01.csv',
            '/home/xddz/data/two_tower_data/2020-09-02.csv'
        ], 
        left_columns, 
        right_columns, 
        csv_header, 
        batch_size=256
    )

    valid_dataset = get_dataset_from_csv_files(
        [
            '/home/xddz/data/two_tower_data/2020-09-03.csv'
        ], 
        left_columns, 
        right_columns, 
        csv_header, 
        batch_size=256
    )

    left_model, right_model = build_model()
    
    print(left_model.get_weights()[-2])
    print(right_model.get_weights()[-2])

    left_model, right_model = custom_train_model(
        left_model, 
        right_model, 
        train_dataset=train_dataset, 
        train_steps=math.ceil(712877/256),
        valid_dataset=valid_dataset, 
        valid_steps=math.ceil(350730/256),
        ids_column='cand_id',
        ids_hash_bucket_size=100000,
        num_epochs=10
    )

    print(left_model.get_weights()[-2])
    print(right_model.get_weights()[-2])


