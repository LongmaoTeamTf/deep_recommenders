"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-11-30 15:07:09
@LastEditors: Wang Yao
@LastEditTime: 2020-11-30 19:52:54
"""
from .dcn.dcn import build_dcn
from .xdeepfm.xdeepfm import build_xdeepfm
from .trainner import CtrModelTrainer
from tensorflow.keras.metrics import AUC


train_filepaths = ["/Users/wangyao/Desktop/Recommend/dac/train_10w.txt"]
valid_filepaths = ["/Users/wangyao/Desktop/Recommend/dac/valid_1w.txt"]
test_filepaths = ["/Users/wangyao/Desktop/Recommend/dac/test_1w.txt"]


# model = build_dcn(3, 3)
model = build_xdeepfm()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[AUC()])

trainer = CtrModelTrainer(model, "criteo")

trainer(train_filepaths, valid_filepaths, test_filepaths)