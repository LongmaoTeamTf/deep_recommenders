"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-11-30 11:31:00
@LastEditors: Wang Yao
@LastEditTime: 2020-11-30 16:06:18
"""
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from dataset.criteo import criteoDataFLow


class ReportValidationStatus(Callback):
    """ 输出验证集进度 """

    def __init__(self, valid_steps):
        self.valid_steps = valid_steps

    def on_test_begin(self, logs=None):
        print("\nValidating ... ")

    def on_test_end(self, logs=None): 
        print("\nValidating: done.")

    def on_test_batch_end(self, batch, logs):
        print('Validating: batch[{}/{}] val_loss={:.4f} val_auc={:.4f}'.format(
            batch+1, self.valid_steps, logs['loss'], logs['auc']), end='\r')


class CtrModelTrainer(object):
    """ 点击率模型训练 """

    def __init__(self, keras_model: Model, dataset_name: str, batch_size=512, epochs=10, **kwargs):
        super(CtrModelTrainer, self).__init__(**kwargs)
        assert isinstance(keras_model, Model), \
            f"Model type expected be Keras Model, but is {type(keras_model)}"
        self.model = keras_model
        assert dataset_name in ["criteo"], \
            f"Unsupport dataset name. Should be in (criteo,)"
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.epochs = epochs

    def __call__(self, train_filepaths, valid_filepaths, test_filepaths):
        """ 执行 """
        train_steps = self.calc_steps(train_filepaths)
        valid_steps = self.calc_steps(valid_filepaths)
        test_steps = self.calc_steps(test_filepaths)

        train_dataset = self.create_dataset(train_filepaths)
        valid_dataset = self.create_dataset(valid_filepaths)
        test_dataset = self.create_dataset(test_filepaths)

        self.train(
            train_dataset,
            valid_dataset,
            train_steps,
            valid_steps,
        )
        self.eval(
            test_dataset, 
            test_steps
        )

    def calc_steps(self, filenames, skip_header=True):
        """ 计算训练步数 """
        _total_num = 0
        for fn in filenames:
            cmd = "wc -l < {}".format(fn)
            cmd_res = os.popen(cmd)
            _num_lines = int(cmd_res.read().strip())
            if skip_header is True:
                _num_lines -= 1
            _total_num += _num_lines
        _steps = _total_num // self.batch_size
        return _steps

    def create_dataset(self, filepath):
        """ 创建数据集 """
        if self.dataset_name == "criteo":
            criteo_dataflow = criteoDataFLow(
                filepath, 
                batch_size=self.batch_size,
                epochs=1)
            dataset = criteo_dataflow.create_criteo_dataset_from_generator()
        return dataset

    def train(self, train_dataset, valid_dataset, train_steps, valid_steps):
        """ 训练 """
        self.model.fit(
            train_dataset.repeat(self.epochs),
            steps_per_epoch=train_steps,
            validation_data=valid_dataset.repeat(self.epochs),
            validation_steps=valid_steps,
            epochs=self.epochs,
            callbacks=[
                EarlyStopping(patience=3), 
                ReportValidationStatus(valid_steps)
            ],
            shuffle=True,
            verbose=1
        )

    def eval(self, test_dataset, test_steps):
        """ 验证 """
        self.model.evaluate(
            test_dataset, steps=test_steps
        )
