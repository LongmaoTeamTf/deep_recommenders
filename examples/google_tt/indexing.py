"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-04-30 15:18:32
@LastEditors: Wang Yao
@LastEditTime: 2020-09-27 14:21:53
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
import mkl
import json
import faiss
import pathlib
import requests
import numpy as np
import tensorflow as tf

sys.path.append("../..")
from src.embedding.google_tt.modeling import _log_norm, _time_exp_norm, build_model
from src.embedding.google_tt.train import get_dataset_from_csv_files


mkl.get_max_threads()

data_dir = pathlib.Path("/home/xddz/data/two_tower_data")
# model_dir = "/home/xddz/data/two_tower_data/model/models/google_tt_candidate/20200924"
checkpoints_path = "/home/xddz/data/two_tower_data/model/training_checkpoints/20200927/right-ckpt-3"
faiss_index_path = "/home/xddz/data/two_tower_data/index/google_tt_20200927.faiss"

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

filenames = sorted([str(fn) for fn in data_dir.glob("*.csv")])[-7:]
print("Index filenames: ")
for fn in filenames:
    print(fn)

dataset = get_dataset_from_csv_files(
    filenames, 
    left_columns, 
    right_columns,
    csv_header, 
    batch_size=1024
)

# model = tf.keras.models.load_model(
#     model_dir, 
#     custom_objects={
#         '_log_norm': _log_norm, 
#         '_time_exp_norm': _time_exp_norm
#     },
#     compile=False
# )

_, model = build_model()

ckpt = tf.train.Checkpoint(model=model)
ckpt.restore(checkpoints_path)


d = 128
nlist = 10

quantizer = faiss.IndexFlatIP(d)
faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
faiss_index_id_map = faiss.IndexIDMap(faiss_index)

batches = 0

global_datas = {}
for _, candidates, _ in dataset:

    cand_ids = candidates.get('cand_id').numpy()
    predictions = model.predict(candidates)

    for cand_id, pred in zip(cand_ids, predictions):
        global_datas[int(cand_id)] = pred
    
    if batches % 50 == 0:
        print('Faiss index: Batches[{}] ntotal={}'.format(batches, len(global_datas.keys())))

    batches += 1

vectors = np.array(list(global_datas.values()), dtype=np.float32)
ids = np.array(list(global_datas.keys()), dtype=np.int64)

faiss_index_id_map.train(vectors)               # pylint: disable=no-value-for-parameter
faiss_index_id_map.add_with_ids(vectors, ids)   # pylint: disable=no-value-for-parameter

print('Faiss index: ntotal={}'.format(faiss_index_id_map.ntotal))
faiss.write_index(faiss_index_id_map, faiss_index_path)
print('Faiss index saved.')

