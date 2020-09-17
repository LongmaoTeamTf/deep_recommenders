"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-04-30 15:18:32
@LastEditors: Wang Yao
@LastEditTime: 2020-09-17 16:58:39
"""
import os
import sys
import mkl
import json
import faiss
import pathlib
import requests
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

def get_invlists(index):
    index = faiss.extract_index_ivf(index)
    invlists = index.invlists
    all_ids = []
    for listno in range(index.nlist):
        ls = invlists.list_size(listno)
        if ls == 0:
            continue
        all_ids.append(
            faiss.rev_swig_ptr(invlists.get_ids(listno), ls).copy()
        )
    all_ids = np.hstack(all_ids)
    return all_ids


d = 128
nlist = 10

quantizer = faiss.IndexFlatIP(d)
faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
faiss_index_id_map = faiss.IndexIDMap(faiss_index)

global_ids = set()

headers = {"content-type": "application/json"}

batches = 0

global_datas = {}
for _, candidates, _ in dataset:

    cand_ids = candidates.get('cand_id').numpy()
    predictions = model.predict(candidates)
    # candidates_data = {}
    # for k, v in candidates.items():
    #     if v.dtype == 'string':
    #         if len(v.shape) > 1:
    #             candidates_data[k] = [[d.decode('utf-8') for d in x] for x in v.numpy()]
    #         else:
    #             candidates_data[k] = [[x.decode('utf-8')] for x in v.numpy()]
    #     else:
    #         candidates_data[k] = v.numpy().astype('float').reshape((-1, 1)).tolist()

    # data = json.dumps({"signature": "serving_default", "inputs": candidates_data})

    # json_response = requests.post('http://localhost:8501/v1/models/google_tt_candidate:predict', data=data, headers=headers)
    # predictions = json.loads(json_response.text)['outputs']

    for cand_id, pred in zip(cand_ids, predictions):
        global_datas[int(cand_id)] = pred
    
    # candidates_ids = []
    # candidates_add_indexs = []
    # candidates_update_ids = []
    # candidates_update_indexs = []
    # for i, cand_id in enumerate(candidates.get('cand_id').numpy()):
    #     candidates_ids.append(int(cand_id))
    #     if cand_id not in global_ids:
    #         global_ids.add(cand_id)
    #         candidates_add_indexs.append(i)
    #     else:
    #         if cand_id not in candidates_update_ids:
    #             candidates_update_ids.append(cand_id)
    #             candidates_update_indexs.append(i)
    #         else:
    #             candidates_update_indexs[candidates_update_ids.index(cand_id)] = i

    # candidates_ids = np.array(candidates_ids, dtype=np.int64)

    # predictions = model.predict(candidates)
    
    # faiss_index_id_map.train(predictions[candidates_add_indexs])                        # pylint: disable=no-value-for-parameter
    # faiss_index_id_map.add_with_ids(                                                    # pylint: disable=no-value-for-parameter
    #     predictions[candidates_add_indexs], candidates_ids[candidates_add_indexs])

    # if candidates_ids[candidates_update_indexs].size != 0:
    #     faiss_index_id_map.remove_ids(candidates_ids[candidates_update_indexs])
    #     faiss_index_id_map.train(predictions[candidates_update_indexs])                 # pylint: disable=no-value-for-parameter
    #     faiss_index_id_map.add_with_ids(                                                # pylint: disable=no-value-for-parameter
    #         predictions[candidates_update_indexs], candidates_ids[candidates_update_indexs]) 

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
    

faiss_index_id_map = faiss.read_index(faiss_index_path)
faiss_index_id_map.nprobe = 10

query = np.array([[0.2] *  128], dtype=np.float32)

distances, cand_ids = faiss_index_id_map.search(query, 10)
print(distances)
print(cand_ids)


