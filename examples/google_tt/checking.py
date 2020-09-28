import sys
sys.path.append('../..')
import json
import time
import faiss
import requests
import numpy as np
import tensorflow as tf

from src.embedding.google_tt.modeling import build_model
from src.embedding.google_tt.modeling import _log_norm, _time_exp_norm



def check_serving(url, data):
    """检查Serving"""
    headers = {"content-type": "application/json"}
    data = json.dumps({"signature": "serving_default", "inputs": data})

    start = time.time()
    json_response = requests.post(url, data=data, headers=headers)
    stop = time.time()

    print("Predictions: \n{}".format(json_response.text))
    print("Time: {:.4f}".format(stop-start))


def check_model(model_dir, data):
    """检查模型"""
    model = tf.keras.models.load_model(
        model_dir, 
        custom_objects={
            '_log_norm': _log_norm, 
            '_time_exp_norm': _time_exp_norm
        },
        compile=False
    )

    for l in model.layers:
        print(l.name)
        print(l.weights)

    predictions = model.predict(data)
    print("Predictions: \n{}".format(predictions))
    return predictions
    

def check_checkpoints(checkpoints_path, model, data):
    """检查checkpoints"""
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(checkpoints_path)

    for l in model.layers:
        print(l.name)
        print(l.weights)

    predictions = model.predict(data)
    print("Predictions: \n{}".format(predictions))
    return predictions


def check_faiss(faiss_path, query):
    """检查faiss索引"""
    faiss_index_id_map = faiss.read_index(faiss_path)
    faiss_index_id_map.nprobe = 10

    distances, cand_ids = faiss_index_id_map.search(query, 10)
    print(distances)
    print(cand_ids)


if __name__ == "__main__":
   
    query_data = {
        'past_watches': np.array([
            ['214318', '216388', '181692', '208917', '216444'], 
            ['213680', '213740', '216747', '214813', '213372'],
            ['212192', '213680', '213740', '216747', '214813'],
            ['215915', '73574', '213190', '216450', '200319'],
            ['214439', '207264', '213830', '216450', '123042']
        ]), 
        'seed_id': np.array([
            ['212199'], 
            ['212192'], 
            ['213372'], 
            ['44236'], 
            ['172221']
        ]), 
        'seed_category': np.array([['记录'], ['旅行'], ['动画'], ['生活'], ['记录']]), 
        'seed_tags': np.array([
            ['催泪', '温情', '浪漫', '记录精选', ''], 
            ['混剪', '跟着开眼看世界', '集锦', '风光大片', '星空'], 
            ['搞笑', '动画梦工厂', '魔性', '童趣', '喜剧'],
            ['生活', '', '', '', ''],
            ['', '', '', '', '']
        ]), 
        'seed_gap_time': np.array([
            [288781.0], 
            [2253378.0], 
            [1908595.0],
            [98364354.0],
            [33854679.0]
        ]), 
        'seed_duration_time': np.array([[767], [325], [420], [69], [283]]), 
        'seed_play_count': np.array([[1836], [7311], [6950], [529], [75]]), 
        'seed_like_count': np.array([[87],[654], [539], [31], [3]]), 
        'seed_collect_count': np.array([[35], [250], [152], [2], [0]]), 
        'seed_share_count': np.array([[63], [426], [156], [32], [3]])
    }

    # model_dir = '/home/xddz/data/two_tower_data/model/models/google_tt_query/20200924'
    left_checkpoints_dir = "/home/xddz/data/two_tower_data/model/training_checkpoints/20200927_debug/left-ckpt-3"
    # right_checkpoints_dir = "/home/xddz/data/two_tower_data/model/training_checkpoints/20200927_debug/right-ckpt-3"
    faiss_path = "/home/xddz/data/two_tower_data/index/google_tt_20200927_debug.faiss"
    # left_saved_model_path = "/home/xddz/data/two_tower_data/model/models/google_tt_query/20200927"
    # right_saved_model_path = "/home/xddz/data/two_tower_data/model/models/google_tt_candidate/20200927"

    left_model, right_model = build_model()

    left_ckpt = tf.train.Checkpoint(model=left_model)
    left_ckpt.restore(left_checkpoints_dir)

    # left_model.save(left_saved_model_path)

    # right_ckpt = tf.train.Checkpoint(model=right_model)
    # right_ckpt.restore(right_checkpoints_dir)

    # right_model.save(right_saved_model_path)

    # predictions = check_model(model_dir, query_data)
    
    predictions = check_checkpoints(left_checkpoints_dir, left_model, query_data)

    check_faiss(faiss_path, predictions)

    