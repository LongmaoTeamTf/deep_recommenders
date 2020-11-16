"""
@Description: DCN train
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-11-10 11:14:39
@LastEditors: Wang Yao
@LastEditTime: 2020-11-12 11:48:49
"""
import sys
sys.path.append("..")
from dataset.criteo import criteoDataFLow
from dcn import build_dcn 

from math import floor, ceil
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping


class ReportValidationStatus(Callback):

    def __init__(self, valid_steps):
        self.valid_steps = valid_steps

    def on_test_begin(self, logs=None):
        print("\nValidating ... ")

    def on_test_end(self, logs=None): 
        print("\nValidating: done.")

    def on_test_batch_end(self, batch, logs):
        print('Validating: batch[{}/{}] val_loss={:.4f} val_auc={:.4f}'.format(
            batch+1, self.valid_steps, logs['loss'], logs['auc']), end='\r')


data_path = "/Users/wangyao/Desktop/Recommend/dac/train.txt"
total_examples_num = 10000
batch_size = 512
epochs = 10
cross_layers_num = 3
dnn_layers_num = 3

# 创建Criteo数据集
criteo_dataflow = criteoDataFLow(data_path, batch_size=batch_size, epochs=1)
dataset = criteo_dataflow.create_criteo_dataset_from_generator()
train_steps = floor(total_examples_num * 0.8) // batch_size
test_steps = ceil(total_examples_num * 0.2) // batch_size
train_dataset = dataset.take(train_steps)
test_dataset = dataset.skip(train_steps)

valid_steps = ceil(train_steps * 0.2)
train_steps = train_steps - valid_steps
valid_dataset = train_dataset.skip(train_steps)
train_dataset = train_dataset.take(train_steps)
print("Train steps: {}".format(train_steps))
print("Valid steps: {}".format(valid_steps))
print("Test steps: {}".format(test_steps))

# 创建DCN网络
print("Build and complie DCN model ... ")
dcn = build_dcn(cross_layers_num, dnn_layers_num)
dcn.compile(loss="binary_crossentropy", optimizer="adam", metrics=[AUC()])

# 训练
print("Train DCN model ... ")
dcn.fit(
    train_dataset.repeat(epochs),
    steps_per_epoch=train_steps,
    validation_data=valid_dataset.repeat(epochs),
    validation_steps=valid_steps,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=3), ReportValidationStatus(valid_steps)],
    shuffle=True,
    verbose=1
)
# 测试集验证
print("Evaluate DCN model ... ")
dcn.evaluate(
    test_dataset, steps=test_steps
)

