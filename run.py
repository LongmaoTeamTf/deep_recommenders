"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-12-01 16:26:40
@LastEditors: Wang Yao
@LastEditTime: 2020-12-01 17:41:47
"""
import sys
sys.dont_write_bytecode = True
import yaml
from deep_recommend.recommend.ctr.embedding_mlp import EmbeddingMLP
from deep_recommend.recommend.ctr.trainner import CtrModelTrainer
from deep_recommend.recommend.ctr.xdeepfm.xdeepfm import CIN


train_filepaths = ["/Users/wangyao/Desktop/Recommend/dac/train_10w.txt"]
valid_filepaths = ["/Users/wangyao/Desktop/Recommend/dac/valid_1w.txt"]
test_filepaths = ["/Users/wangyao/Desktop/Recommend/dac/test_1w.txt"]

criteo_yaml = "/Users/wangyao/code/DeepRecommend/deep_recommend/recommend/ctr/dataset/criteo.yml"
xdeepfm_yaml = "/Users/wangyao/code/DeepRecommend/deep_recommend/recommend/ctr/xdeepfm/xdeepfm.yml"

criteo_config = yaml.load(open(criteo_yaml))
model_config = yaml.load(open(xdeepfm_yaml))

embedding_mlp = EmbeddingMLP(
    criteo_config.get("dense_features"),
    criteo_config.get("sparse_features"),
    model_config.get("ff").get("hidden_sizes").split(","),
    model_config.get("ff").get("hidden_activation"),
    model_config.get("ff").get("hidden_dropout_rates").split(","),
    model_config.get("logits").get("size"),
    model_config.get("logits").get("activation"),
    model_config.get("model").get("name"),
    model_config.get("model").get("loss"),
    model_config.get("model").get("optimizer")
)
model = embedding_mlp(
    CIN(
        model_config.get("cin").get("feature_maps").split(","),
        model_config.get("cin").get("feature_embedding_dim")
    )
)

trainer = CtrModelTrainer(model, "criteo")

trainer(train_filepaths, valid_filepaths, test_filepaths)
