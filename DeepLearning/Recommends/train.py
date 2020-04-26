"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-04-26 17:47:37
@LastEditors: Wang Yao
@LastEditTime: 2020-04-26 23:00:12
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from deepfm import EmbeddingLayer
from deepfm import OneOrder, TwoOrder, HighOrder
from deepfm import LR
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
    
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


def data_preparing(filepath, n_samples=50000):
    data = []
    n = 1
    progress = tqdm(total=n_samples)
    with open(filepath, "r") as f:
        for line in f:
            if n > n_samples:
                break
            line = line.strip()
            line_data = line.split("\t")
            line_data = [None if x == "" else x for x in line_data ]
            data.append(line_data)
            progress.update(1)
            n += 1
    progress.close()
    df = pd.DataFrame(data)
    integer_cols = [f"I{i}" for i in range(13)]
    categorical_cols = [f"C{i}" for i in range(26)]
    df.columns = ["label"] + integer_cols + categorical_cols
    return df, integer_cols, categorical_cols


def data_process(df, integer_cols, categorical_cols):
    for col in integer_cols:
        df[col] = df[col].fillna(0.)
        df[col] = df[col].astype(np.float32)
        df[col] = df[col].apply(lambda x: np.log(x+1) if x > -1 else -1)
    encoder = LabelEncoder()
    sparse_values_size = []
    for col in categorical_cols:
        df[col] = df[col].fillna("-1")
        df[col] = encoder.fit_transform(df[col])
        sparse_values_size.append(df[col].nunique())
    return df, sparse_values_size


def build_model(integer_cols, categorical_cols, sparse_values_size, embedding_dim=10):
    sparse_inputs = [Input(shape=(1,), dtype=tf.int32, name=col) for col in categorical_cols] 
    dense_inputs = [Input(shape=(1,), dtype=tf.float32, name=col) for col in integer_cols]
    one_order_outputs = OneOrder(sparse_values_size)([sparse_inputs, dense_inputs])
    embeddings = EmbeddingLayer(sparse_values_size, embedding_dim)([sparse_inputs, dense_inputs])
    two_order_outputs = TwoOrder()(embeddings)
    high_order_outputs = HighOrder(2)(embeddings)
    outputs = LR()([one_order_outputs, two_order_outputs, high_order_outputs])
    model = Model(inputs=[sparse_inputs, dense_inputs], outputs=outputs)
    return model


if __name__ == "__main__":
    filepath = "/home/xddz/code/eyepetizer_recommends/data/dac/train.txt"
    # filepath = "/Users/wangyao/Desktop/eyepetizer/dac/train.txt"
    # 数据准备
    df, integer_cols, categorical_cols = data_preparing(filepath, n_samples=1000000)
    # 数据处理
    df, sparse_values_size = data_process(df, integer_cols, categorical_cols)
    # 生成训练数据
    sparse_inputs = [df[col].values for col in categorical_cols]
    dense_inputs = [df[col].values for col in integer_cols]
    targets = df['label'].astype(np.float32).values
    # 构建模型
    model = build_model(integer_cols, categorical_cols, sparse_values_size)
    model.summary()
    plot_model(model, "deepfm.png")
    # 训练模型
    es = EarlyStopping(patience=5)
    model.compile(
        loss="binary_crossentropy", 
        optimizer="adam", 
        metrics=['accuracy'])
    model.fit(
        sparse_inputs + dense_inputs, 
        targets, 
        batch_size=256,
        epochs=10, 
        validation_split=0.2,
        callbacks=[es])
