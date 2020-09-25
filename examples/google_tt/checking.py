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
            ['214318_216388_181692_208917_216444_192761_215891_215969_213241_215971_215849_215930_215902_215772_215612_209749_215901_215903_215366_215776_128415_215757_215490_214615_215190_214990_215777_2104_170828_155349_154290_155149_5614_179233_176368_170321_209965_209966_154751_209684_13885_215888_155885_167643_209963_166176_177235_153110_7746_168968_209686'],
            ['213680_213740_216747_214813_213372_212192_215915_216141_209191_132922_215766_216828_216221_216783_616_175433_3284_215431_200927_211212_213824_209411_216004_216459_214784_216193_216415_216486_216567_216568_214471_127541_215933_210790_145700_211952_211931_215970_213176_211058_215366_214345_214656_213184_210025_214660_214990_207742_215190_215968_214955'],
            ['212192_213680_213740_216747_214813_213372_215915_216141_209191_132922_215766_216828_216221_216783_616_175433_3284_215431_200927_211212_213824_209411_216004_216459_214784_216193_216415_216486_216567_216568_214471_127541_215933_210790_145700_211952_211931_215970_213176_211058_215366_214345_214656_213184_210025_214660_214990_207742_215190_215968_214955']
        ]), 
        'seed_id': np.array([['212199'], ['212192'], ['213372']]), 
        'seed_category': np.array([['记录'], ['旅行'], ['动画']]), 
        'seed_tags': np.array([['催泪_温情_浪漫_记录精选'], ['混剪_跟着开眼看世界_集锦_风光大片_星空_极光_摄影艺术_旅游'], ['搞笑_动画梦工厂_魔性_童趣_喜剧_无厘头_讽刺_幽默_动画']]), 
        'seed_gap_time': np.array([[288781.0], [2253378.0], [1908595.0]]), 
        'seed_duration_time': np.array([[767], [325], [420]]), 
        'seed_play_count': np.array([[1836], [7311], [6950]]), 
        'seed_like_count': np.array([[87],[654], [539]]), 
        'seed_collect_count': np.array([[35], [250], [152]]), 
        'seed_share_count': np.array([[63], [426], [156]])
    }

    model_dir = '/home/xddz/code/DeepRecommend/examples/google_tt/models/google_tt_query/20200924'
    faiss_path = '/home/xddz/data/two_tower_data/index/google_tt_20200924.faiss'

    predictions = check_model(model_dir, query_data)

    check_faiss(faiss_path, predictions)

    