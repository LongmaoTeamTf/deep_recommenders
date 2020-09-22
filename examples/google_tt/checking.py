import sys
sys.path.append('../..')
import json
import time
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


if __name__ == "__main__":
   
    query_data = {
        'past_watches': [['214318_216388_181692_208917_216444_192761_215891_215969_213241_215971_215849_215930_215902_215772_215612_209749_215901_215903_215366_215776_128415_215757_215490_214615_215190_214990_215777_2104_170828_155349_154290_155149_5614_179233_176368_170321_209965_209966_154751_209684_13885_215888_155885_167643_209963_166176_177235_153110_7746_168968_209686']], 
        'seed_id': [['212199']], 
        'seed_category': [['记录']], 
        'seed_tags': [['催泪_温情_浪漫_记录精选']], 
        'seed_gap_time': [[288781.0]], 
        'seed_duration_time': [[767]], 
        'seed_play_count': [[1836]], 
        'seed_like_count': [[87]], 
        'seed_collect_count': [[35]], 
        'seed_share_count': [[63]]
    }

