import sys
import pathlib
sys.path.append("../..")
import json
import requests
import tensorflow as tf

from src.embedding.google_tt.modeling import build_model, HashEmbeddings, L2Normalization, _log_norm, _time_exp_norm
from src.embedding.google_tt.train import get_dataset_from_csv_files, hash_simple


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

data_dir = "/home/xddz/data/two_tower_data"

data_dir = pathlib.Path(data_dir)
filenames = data_dir.glob("*.csv")

train_dataset = get_dataset_from_csv_files(
    filenames, 
    left_columns, 
    right_columns,
    csv_header, 
    batch_size=10
)

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
dataset = strategy.experimental_distribute_dataset(train_dataset)

for x, y, l in dataset:
    # print(x)
    # print(type(x))
    cands = y.get('cand_id')
    print(cands)
    indexs = hash_simple(cands, 100000)
    print(indexs)
    break

# headers = {"content-type": "application/json"}


# for queries, candidates, reward in train_dataset.take(1):
#     data = {}
#     for feat_name, feat_value in candidates.items():
#         if feat_value.dtype == 'string':
#             if len(feat_value.shape) == 2:
#                 feat_value = [[v.decode('utf-8') for v in val] for val in feat_value.numpy()]
#             else:
#                 feat_value = [[val.decode('utf-8')] for val in feat_value.numpy()]
#         else:
#             feat_value = [[float(val)] for val in feat_value.numpy()]
#         data[feat_name] = feat_value
    
#     print(data)
#     data = json.dumps({"signature": "serving_default", "inputs": data})
#     json_response = requests.post('http://localhost:8501/v1/models/indexing:predict', data=data, headers=headers)
#     predictions = json.loads(json_response.text)
#     print(predictions)


# l_model, r_model = build_model()

# # model = tf.saved_model.load('/Users/wangyao/Desktop/Recommend/eyepetizer/google_tt/google_tt_left_0.01')

# path = "/home/xddz/code/DeepRecommend/examples/google_tt/models/google_tt_left_0.01"
# model = tf.keras.models.load_model(path, custom_objects={'_log_norm': _log_norm, '_time_exp_norm': _time_exp_norm})

# for l in model.layers:
#     # if 'tags' in l.name:
#     #     if l.weights:
#     #         print(l.name)
#     #         print(l.weights[0].numpy().tolist())
#     print(l.name)
#     print(l.weights)