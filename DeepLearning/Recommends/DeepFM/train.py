"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-04-26 17:47:37
@LastEditors: Wang Yao
@LastEditTime: 2020-04-29 16:48:40
"""
import os
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from deepfm import build_deepfm


def data_preparing(filepath,
                   categorical_cols,
                   numerical_cols,
                   n_samples=50000):
    """数据准备"""
    data = []
    total = 1
    progress = tqdm(total=n_samples)
    with open(filepath, "r") as f:
        for line in f:
            if total > n_samples:
                break
            line = line.strip()
            line_data = line.split("\t")
            line_data = [None if x == "" else x for x in line_data ]
            data.append(line_data)
            progress.update(1)
            total += 1
    progress.close()
    data = pd.DataFrame(data)
    data.columns = ["label"] + numerical_cols + categorical_cols
    return data


def data_process(data, integer_cols, categorical_cols):
    for col in integer_cols:
        data[col] = data[col].fillna(0.)
        data[col] = data[col].astype(np.float32)
        data[col] = data[col].apply(lambda x: np.log(x+1) if x > -1 else -1)
    encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = data[col].fillna("-1")
        data[col] = encoder.fit_transform(data[col])
    return data


def main(_):
    data_root = "/home/xddz/eyepetizer_data/dac"
    data_root = pathlib.Path(data_root)
    train_data_fn = str(data_root/'train.txt')
    test_data_fn = str(data_root/'test.txt')
    models_fp = data_root/'models'

    model_config = {
        'linear_regularizer': 'l1',
        'linear_trainable': True,
        'embed_dim': 10,
        'embed_regularizer': 'l2',
        'embed_trainable': True,
        'embed_numerical_embedding': False,
        'fm_numerical_interactive': False,
        'dnn_num_layers': 2,
        'dnn_dropout_rate': 0.5,
        'dnn_activation': 'relu'
    }

    train_config = {
        'batch_size': 256,
        'num_epochs': 10,
        'loss': tf.keras.losses.binary_crossentropy,
        'optimizer': tf.keras.optimizers.Adam(),
        'metrics': [
            tf.keras.metrics.AUC(),
            tf.keras.metrics.binary_accuracy
        ],
        'callbacks': [tf.keras.callbacks.EarlyStopping(patience=3)],
        'shuffle': True,
        'valid_split': 0.1,
        'version': 1,
        'model_save_fp': str(models_fp / 'deepfm'),
    }

    numerical_cols = [f"I{i}" for i in range(13)]
    categorical_cols = [f"C{i}" for i in range(26)]

    # Train data
    train_data = data_preparing(train_data_fn,
                                categorical_cols,
                                numerical_cols,
                                n_samples=50000)
    train_data = data_process(train_data, categorical_cols, numerical_cols)
    train_sparse_inputs = [train_data[col].values for col in categorical_cols]
    train_dense_inputs = [train_data[col].values for col in numerical_cols]
    train_inputs = [train_sparse_inputs, train_dense_inputs]
    targets = train_data['label'].astype(np.float32).values

    # Test data
    test_data = data_preparing(test_data_fn,
                               categorical_cols,
                               numerical_cols,
                               n_samples=5000)
    test_data = data_process(test_data, categorical_cols, numerical_cols)
    test_sparse_inputs = [test_data[col].values for col in categorical_cols]
    test_dense_inputs = [test_data[col].values for col in numerical_cols]
    test_inputs = [test_sparse_inputs, test_dense_inputs]

    # Build model
    categorical_config = {
        col: train_data[col].nunique()
        for col in categorical_cols
    }
    model = build_deepfm(model_config,
                         categorical_config,
                         numerical_cols)

    model.compile(
        loss=train_config.get('loss'),
        optimizer=train_config.get('optimizer'),
        metrics=train_config.get('metrics')
    )

    # Train model
    print("Model Training ... ")
    history = model.fit(
        train_inputs, targets,
        batch_size=train_config.get('batch_size'),
        epochs=train_config.get('num_epochs'),
        validation_split=train_config.get('valid_split'),
        callbacks=train_config.get('callbacks'),
        shuffle=train_config.get('shuffle')
    )

    # Evaluate model
    print("Model Evaluating ... ")
    model.evaluate(
        test_inputs,
        batch_size=train_config.get('batch_size')
    )

    if train_config.get('model_save_fp') is not None:
        model_save_fp = os.path.join(train_config.get('model_save_fp'),
                                     str(train_config.get('version')))
        tf.keras.models.save_model(model, model_save_fp)


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.app.run()
