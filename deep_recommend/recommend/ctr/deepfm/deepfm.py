"""
@Description: Deep & FM Network
@version: https://arxiv.org/abs/1703.04247
@License: MIT
@Author: Wang Yao
@Date: 2020-12-01 18:57:30
@LastEditors: Wang Yao
@LastEditTime: 2020-12-03 19:59:08
"""
from deep_recommend.recommend.ctr.fm.fm import FactorizationMachine
from deep_recommend.recommend.ctr.embedding_mlp import EmbeddingMLP
from deep_recommend.recommend.ctr.embedding_layer import EmbeddingLayer


class DeepFM(object):
    """ Deep & Factorization Machine """
    
    def __init__(self, dataset_config: dict, model_config: dict, **kwargs):
        super(DeepFM, self).__init__(**kwargs)
        self._dataset_config = dataset_config
        self._model_config = model_config

    def __call__(self):
        embedding_layer = EmbeddingLayer(
            self._dataset_config.get("features").get("sparse_features"),
            self._dataset_config.get("features").get("dense_features"),
            return_raw_features=True
        )
        embedding_mlp = EmbeddingMLP(
            self._model_config.get("ff").get("hidden_sizes").split(","),
            self._model_config.get("ff").get("hidden_activation"),
            self._model_config.get("ff").get("hidden_dropout_rates").split(","),
            self._model_config.get("logits").get("size"),
            self._model_config.get("logits").get("activation"),
            self._model_config.get("model").get("name"),
            self._model_config.get("model").get("loss"),
            self._model_config.get("model").get("optimizer"),
            need_raw_features=True
        )
        return embedding_mlp(FactorizationMachine(self._model_config.get("fm").get("factors")),
                             embedding_layer)
                            