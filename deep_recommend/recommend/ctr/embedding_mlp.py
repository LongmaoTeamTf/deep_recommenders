"""
@Description: Embedding & MLP
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-12-01 11:48:59
@LastEditors: Wang Yao
@LastEditTime: 2020-12-03 20:40:09
"""
from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras import activations
from tensorflow.keras.layers import ReLU, Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.metrics import AUC


class EmbeddingMLP(object):
    """ Embedding & MLP """

    __linear_layer__ = "linear"
    __dnn_layer__ = "dnn_layer_"
    __logits_layer__ = "logits_layer"

    def __init__(self,
            ff_hidden_sizes: list,
            ff_hidden_activation: str,
            ff_hidden_dropout_rates: list,
            logits_size: int,
            logits_activation: str,
            model_name: str,
            model_loss: str,
            model_optimizer: str,
            need_raw_features=False,
            **kwargs):
        super(EmbeddingMLP, self).__init__(**kwargs)
        self._ff_hidden_sizes = ff_hidden_sizes
        self._ff_hidden_activation = ff_hidden_activation
        self._ff_hidden_dropout_rates = ff_hidden_dropout_rates
        self._logits_size = logits_size
        self._logits_activation = logits_activation
        self._model_name = model_name
        self._model_loss = model_loss
        self._model_optimizer = model_optimizer
        self._need_raw_features = need_raw_features

    def __call__(self, explicit_part, embedding_layer) -> Model:
        """ Joint explicit part & dnn & linear """
        logits_input = []
        if self._need_raw_features is True:
            concat_embeddings, model_inputs, raw_features = embedding_layer()
            linear_output = self.build_linear(raw_features)
            logits_input.append(linear_output)
        else:
            concat_embeddings, model_inputs = embedding_layer()
        explicit_part_output = explicit_part(concat_embeddings)
        dnn_output = self.build_dnn(concat_embeddings)
        logits_input += [explicit_part_output, dnn_output]
        model_outputs = self.build_logits_output(logits_input)
        model = Model(model_inputs, model_outputs, name=self._model_name)
        model.compile(
            loss=self._model_loss, 
            optimizer=self._model_optimizer, 
            metrics=[AUC()])
        return model

    def build_linear(self, raw_features: Tensor) -> Tensor:
        """ linear layer: wx """
        return Dense(1, name=self.__linear_layer__)(raw_features)

    def build_dnn(self, inputs: Tensor) -> Tensor:
        """ dnn: implicit part """ 
        x = inputs
        for i, (ff_hidden_size, ff_hidden_dropout_rate) in \
            enumerate(zip(self._ff_hidden_sizes, self._ff_hidden_dropout_rates)):
            x = Dense(int(ff_hidden_size), name=self.__dnn_layer__ + str(i))(x)
            x = ReLU()(x)
            x = Dropout(float(ff_hidden_dropout_rate))(x)
        return x

    def build_logits_output(self, inputs: list) -> Tensor:
        """ logits output layer """
        concat_inputs = Concatenate(axis=-1)(inputs)
        x = Dense(
            self._logits_size, 
            activation=activations.get(self._logits_activation),
            name=self.__logits_layer__
        )(concat_inputs)
        return x
      