"""
@Description: 训练器
@version: 1.0.0
@License: MIT
@Author: Wang Yao
@Date: 2020-11-30 11:31:00
@LastEditors: Wang Yao
@LastEditTime: 2020-12-02 17:36:01
"""
import os
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping


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


class ModelTrainer(object):
    """ 点击率模型训练 """

    def __init__(self, 
                 keras_model: Model, 
                 train_steps: int,
                 valid_steps: int, 
                 test_steps: int,
                 save_path: str,
                 version: str,
                 epochs: int=10, 
                 **kwargs):
        super(ModelTrainer, self).__init__(**kwargs)
        assert isinstance(keras_model, Model), \
            f"Model type expected be Keras Model, but is {type(keras_model)}"
        self._model = keras_model
        self._train_steps = train_steps
        self._valid_steps = valid_steps
        self._test_steps = test_steps
        self._save_path = save_path
        self._version = version
        self._epochs = epochs

    def __call__(self, 
                 train_dataset,
                 valid_dataset,
                 test_dataset,
                 **kwargs):
        """ build trainer """
        print("Start training ... ")
        self.train(train_dataset, valid_dataset)
        print("Start evaluating ... ")
        self.eval(test_dataset)
        print("Saving model ... ")
        self.save()
        print(f"Saved model to: {os.path.join(self._save_path, self._version)}")
        
    def train(self, train_dataset, valid_dataset):
        """ 训练 """
        self._model.fit(
            train_dataset.repeat(self._epochs),
            steps_per_epoch=self._train_steps,
            validation_data=valid_dataset.repeat(self._epochs),
            validation_steps=self._valid_steps,
            epochs=self._epochs,
            callbacks=[
                EarlyStopping(),
                ReportValidationStatus(self._valid_steps)
            ],
            shuffle=True,
            verbose=1
        )

    def eval(self, test_dataset):
        """ 验证 """
        self._model.evaluate(
            test_dataset, steps=self._test_steps
        )

    def save(self):
        """ 保存模型 """
        self._model.save(os.path.join(self._save_path, self._version))
