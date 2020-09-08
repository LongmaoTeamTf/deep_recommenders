"""
@Description: 分布式训练
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-08-17 15:51:03
@LastEditors: Wang Yao
@LastEditTime: 2020-09-08 17:33:15
"""
import os
import sys
sys.path.append('../..')
import json
import tensorflow as tf

from examples.google_tt.train_google_tt_model import distribute_train_model


def train_on_workers(worker_index):
    """多机分布式训练"""
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': [
                'localhost:9990',
                'localhost:9991',
            ]
        },
        'task': {'type': 'worker', 'index': worker_index}
    })
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    distribute_train_model(strategy)


if __name__ == "__main__":
    index = sys.argv[1]
    train_on_workers(index)
