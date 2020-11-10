"""
@Description: DCN train
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-11-10 11:14:39
@LastEditors: Wang Yao
@LastEditTime: 2020-11-10 16:15:53
"""
import sys
sys.path.append("..")
from data.criteo import criteoDataFLow
from dcn import build_dcn

from math import floor, ceil
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping


data_path = "/root/dac/train.txt"
total_examples_num = 45840617
batch_size = 512
epochs = 10
cross_layers_num = 3
dnn_layers_num = 3

# 创建Criteo数据集
criteo_dataflow = criteoDataFLow(data_path, batch_size=batch_size, epochs=epochs)
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
    train_dataset,
    steps_per_epoch=train_steps,
    validation_data=valid_dataset,
    validation_steps=valid_steps,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=3)],
    shuffle=True
)
# 测试集验证
print("Evaluate DCN model ... ")
dcn.evaluate(
    test_dataset, steps=test_steps
)

