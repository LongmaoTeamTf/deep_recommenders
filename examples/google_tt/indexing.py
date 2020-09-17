"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-04-30 15:18:32
@LastEditors: Wang Yao
@LastEditTime: 2020-09-17 10:40:58
"""
import os
import sys
import mkl
import json
import faiss
import pathlib
import numpy as np
import tensorflow as tf

sys.path.append("../..")
from src.embedding.google_tt.modeling import _log_norm, _time_exp_norm
from src.embedding.google_tt.train import get_dataset_from_csv_files


mkl.get_max_threads()

data_dir = pathlib.Path("/home/xddz/data/two_tower_data")
model_dir = "/home/xddz/code/DeepRecommend/examples/google_tt/models/google_tt_candidate/20200915"
faiss_index_path = "./google_tt.faiss"

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

filenames = [
    str(fn)
    for fn in data_dir.glob("*.csv")
]

dataset = get_dataset_from_csv_files(
    filenames, 
    left_columns, 
    right_columns,
    csv_header, 
    batch_size=512
)

model = tf.keras.models.load_model(
    model_dir, 
    custom_objects={
        '_log_norm': _log_norm, 
        '_time_exp_norm': _time_exp_norm
    },
    compile=False
)

d = 128
nlist = 10

quantizer = faiss.IndexFlatIP(d)
faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
faiss_index_id_map = faiss.IndexIDMap(faiss_index)

for _, candidates, _ in dataset:
    candidates_ids = [int(cand_id) for cand_id in candidates.get('cand_id')]
    candidates_ids = np.array(candidates_ids, dtype=np.int64)
    predictions = model.predict(candidates)
    
    faiss_index_id_map.train(predictions)                           # pylint: disable=no-value-for-parameter
    faiss_index_id_map.add_with_ids(predictions, candidates_ids)    # pylint: disable=no-value-for-parameter

    if faiss_index_id_map.ntotal % 500 == 0:
        print('Faiss index: ntotal={}'.format(faiss_index_id_map.ntotal))

print('Faiss index: ntotal={}'.format(faiss_index_id_map.ntotal))
faiss.write_index(faiss_index_id_map, faiss_index_path)
print('Faiss index saved.')
    

faiss_index_id_map = faiss.read_index(faiss_index_path)
faiss_index_id_map.nprobe = 10

query = np.array([[0.2] *  128], dtype=np.float32)

distances, cand_ids = faiss_index_id_map.search(query, 10)
print(distances)
print(cand_ids)


