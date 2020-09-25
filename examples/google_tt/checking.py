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
    optimizer = tf.keras.optimizers.Adagrad()

    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.restore(checkpoints_path)

    for l in model.layers:
        print(l.name)
        print(l.weights)

    predictions = model.predict(data)
    print("Predictions: \n{}".format(predictions))


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
            ['214318,216388,181692,208917,216444,192761,215891,215969,213241,215971,215849,215930,215902,215772,215612,209749,215901,215903,215366,215776,128415,215757,215490,214615,215190,214990,215777,2104,170828,155349,154290,155149,5614,179233,176368,170321,209965,209966,154751,209684,13885,215888,155885,167643,209963,166176,177235,153110,7746,168968,209686'.split(',')],
            ['213680,213740,216747,214813,213372,212192,215915,216141,209191,132922,215766,216828,216221,216783,616,175433,3284,215431,200927,211212,213824,209411,216004,216459,214784,216193,216415,216486,216567,216568,214471,127541,215933,210790,145700,211952,211931,215970,213176,211058,215366,214345,214656,213184,210025,214660,214990,207742,215190,215968,214955'.split(',')],
            ['212192,213680,213740,216747,214813,213372,215915,216141,209191,132922,215766,216828,216221,216783,616,175433,3284,215431,200927,211212,213824,209411,216004,216459,214784,216193,216415,216486,216567,216568,214471,127541,215933,210790,145700,211952,211931,215970,213176,211058,215366,214345,214656,213184,210025,214660,214990,207742,215190,215968,214955'.split(',')]
        ]), 
        'seed_id': np.array([['212199'], ['212192'], ['213372']]), 
        'seed_category': np.array([['记录'], ['旅行'], ['动画']]), 
        'seed_tags': np.array([['催泪, 温情, 浪漫, 记录精选'.split(',')], ['混剪, 跟着开眼看世界, 集锦, 风光大片, 星空, 极光, 摄影艺术, 旅游'.split(',')], ['搞笑, 动画梦工厂, 魔性, 童趣, 喜剧, 无厘头, 讽刺, 幽默, 动画'.split(',')]]), 
        'seed_gap_time': np.array([[288781.0], [2253378.0], [1908595.0]]), 
        'seed_duration_time': np.array([[767], [325], [420]]), 
        'seed_play_count': np.array([[1836], [7311], [6950]]), 
        'seed_like_count': np.array([[87],[654], [539]]), 
        'seed_collect_count': np.array([[35], [250], [152]]), 
        'seed_share_count': np.array([[63], [426], [156]])
    }

    model_dir = '/home/xddz/data/two_tower_data/model/models/google_tt_query/20200921'
    faiss_path = '/home/xddz/data/two_tower_data/index/google_tt_20200921.faiss'

    predictions = check_model(model_dir, query_data)

    check_faiss(faiss_path, predictions)

    