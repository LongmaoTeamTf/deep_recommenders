"""
@Description:
@version:
@License: MIT
@Author: Wang Yao
@Date: 2020-04-21 20:21:19
@LastEditors: Wang Yao
@LastEditTime: 2020-05-06 19:39:07
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer


class Linear(Layer):

    def __init__(self, regularizer='l2', trainable=True, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self._regularizer = regularizers.get(regularizer)
        self._trainable = trainable

    def build(self, input_shape):
        categorical_shapes, numerical_shapes = input_shape
        self._categorical_weights = [
            self.add_weight(
                shape=(cat_shape[-1], 1),
                initializer=initializers.glorot_uniform,
                regularizer=self._regularizer,
                trainable=self._trainable,
                name=f'categorical_weights_{i}'
            ) for i, cat_shape in enumerate(categorical_shapes)]
        self._numerical_weights = self.add_weight(
            shape=(len(numerical_shapes), 1),
            initializer=initializers.glorot_uniform,
            regularizer=self._regularizer,
            trainable=self._trainable,
            name='numerical_weights')
        super(Linear, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        categorical_inputs, numerical_inputs = inputs
        outputs = 0.
        for cat_input, cat_weight in zip(categorical_inputs, self._categorical_weights):
            sparse_w = K.dot(cat_input, cat_weight)
            outputs += sparse_w
        if numerical_inputs:
            numerical_inputs = K.concatenate(numerical_inputs)
            outputs += K.dot(numerical_inputs, self._numerical_weights)
        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'regularizer': self._regularizer,
            'trainable': self._trainable
        })
        return config


class Embedding(Layer):

    def __init__(self, embedding_dim=10, regularizer='l2', trainable=True, numerical_embedding=True, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self._regularizer = regularizers.get(regularizer)
        self._embedding_dim = embedding_dim
        self._trainable = trainable
        self._numerical_embedding = numerical_embedding

    def build(self, input_shape):
        categorical_shapes, numerical_shapes = input_shape
        self._categorical_weights = [
            self.add_weight(
                shape=(cat_shape[-1], self._embedding_dim),
                initializer=initializers.glorot_uniform,
                regularizer=self._regularizer,
                trainable=self._trainable,
                name=f'categorical_weights_{i}'
            ) for i, cat_shape in enumerate(categorical_shapes)]
        if self._numerical_embedding is True:
            self._numerical_weights = [
                self.add_weight(
                    shape=(1, self._embedding_dim),
                    initializer=initializers.glorot_uniform,
                    regularizer=self._regularizer,
                    trainable=self._trainable,
                    name=f'numerical_weights_{i}'
                ) for i in range(len(numerical_shapes))]
        super(Embedding, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        categorical_inputs, numerical_inputs = inputs
        categorical_embeddings = []
        for cat_input, cat_weight in zip(categorical_inputs, self._categorical_weights):
            cat_embed = K.dot(cat_input, cat_weight)
            categorical_embeddings.append(cat_embed)
        if self._numerical_embedding is True:
            numerical_embeddings = []
            for num_input, num_weight in zip(numerical_inputs, self._numerical_weights):
                num_embed = K.dot(num_input, num_weight)
                numerical_embeddings.append(num_embed)
            embeddings = (categorical_embeddings, numerical_embeddings)
        else:
            embeddings = (categorical_embeddings, numerical_inputs)
        return embeddings

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'regularizer': self._regularizer,
            'embedding_dim': self._embedding_dim,
            'trainable': self._trainable,
            'numerical_embedding': self._numerical_embedding
        })
        return config


class FM(Layer):

    def __init__(self, numerical_interactive=False, **kwargs):
        super(FM, self).__init__(**kwargs)
        self._numerical_interactive = numerical_interactive

    @tf.function
    def call(self, inputs):
        categorical_inputs, numerical_inputs = inputs
        if categorical_inputs and numerical_inputs and \
                categorical_inputs[0].shape[-1] != numerical_inputs[0].shape[-1] \
                and self._numerical_interactive is True:
            raise ValueError('If `fm_numerical_interactive` is True, '
                             'categorical_inputs`s shape must equals to numerical_inputs`s shape')
        if self._numerical_interactive is True:
            exp_inputs = [K.expand_dims(x, axis=1) for x in categorical_inputs+numerical_inputs]
        else:
            exp_inputs = [K.expand_dims(x, axis=1) for x in categorical_inputs]
        concat_inputs = K.concatenate(exp_inputs, axis=1)
        square_inputs = K.square(K.sum(concat_inputs, axis=1))
        sum_inputs = K.sum(K.square(concat_inputs), axis=1)
        cross_term = square_inputs - sum_inputs
        outputs = 0.5 * K.sum(cross_term, axis=1, keepdims=True)
        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numerical_interactive': self._numerical_interactive
        })
        return config


class FeedForwardDNN(Layer):

    def __init__(self, n_layers, dropout_rate=0., activation='relu', **kwargs):
        super(FeedForwardDNN, self).__init__(**kwargs)
        self._n_layers = n_layers
        self._dropout_rate = dropout_rate
        self._activation = activations.get(activation)

    def build(self, input_shape):
        categorical_shape, numerical_shape = input_shape
        kernel_size = 0
        for shape in categorical_shape + numerical_shape:
            kernel_size += shape[-1]
        self._kernel_weights = [
            self.add_weight(
                shape=(kernel_size, kernel_size),
                initializer=initializers.glorot_uniform,
                trainable=True,
                name=f'kernel_weight_{i}'
            ) for i in range(self._n_layers)]
        self._output_weight = self.add_weight(
            shape=(kernel_size, 1),
            initializer=initializers.glorot_uniform,
            trainable=True,
            name='output_weight')
        super(FeedForwardDNN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        categorical_inputs, numerical_inputs = inputs
        outputs = K.concatenate(categorical_inputs+numerical_inputs, axis=-1)
        for i in range(self._n_layers):
            outputs = K.dot(outputs, self._kernel_weights[i])
            outputs = self._activation(outputs)
            outputs = K.in_train_phase(
                K.dropout(outputs, self._dropout_rate),
                outputs,
            )
        outputs = K.dot(outputs, self._output_weight)
        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_layers': self._n_layers,
            'dropout_rate': self._dropout_rate,
            'activation': self._activation})
        return config


class LR(Layer):

    def __init__(self, **kwargs):
        super(LR, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        outputs = K.concatenate(inputs, axis=1)
        outputs = K.sum(outputs, axis=1, keepdims=True)
        outputs = K.sigmoid(outputs)
        return outputs


class DeepFM(Layer):

    def __init__(self, config, **kwargs):
        super(DeepFM, self).__init__(**kwargs)
        # Linear config
        self._linear_regularizer = config.get('linear_regularizer', 'l2')
        self._linear_trainable = config.get('linear_trainable', True)
        # Embedding config
        self._embed_dim = config.get('embed_dim', 10)
        self._embed_regularizer = config.get('embed_regularizer', 'l2')
        self._embed_trainable = config.get('embed_trainable', True)
        self._embed_numerical_embedding = config.get('embed_numerical_embedding', False)
        # FM config
        self._fm_numerical_interactive = config.get('fm_numerical_interactive', False)
        # DNN config
        self._dnn_num_layers = config.get('dnn_num_layers', 2)
        self._dnn_dropout_rate = config.get('dnn_dropout_rate', 0.5)
        self._dnn_activation = config.get('dnn_activation', 'relu')

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('Input shape is wrong.')
        self._Linear = Linear(
            regularizer=self._linear_regularizer,
            trainable=self._linear_trainable
        )
        self._Embedding = Embedding(
            embedding_dim=self._embed_dim,
            regularizer=self._embed_regularizer,
            trainable=self._embed_trainable,
            numerical_embedding=self._embed_numerical_embedding
        )
        self._FM = FM(
            numerical_interactive=self._fm_numerical_interactive
        )
        self._DNN = FeedForwardDNN(
            self._dnn_num_layers,
            dropout_rate=self._dnn_dropout_rate,
            activation=self._dnn_activation
        )
        self._LR = LR()
        super(DeepFM, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        categorical_inputs, numerical_inputs = inputs
        linear_outputs = self._Linear([categorical_inputs, numerical_inputs])
        embeddings = self._Embedding([categorical_inputs, numerical_inputs])
        fm_outputs = self._FM(embeddings)
        dnn_outputs = self._DNN(embeddings)
        outputs = self._LR([linear_outputs, fm_outputs, dnn_outputs])
        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'linear_regularizer': self._linear_regularizer,
            'linear_trainable': self._linear_trainable,
            'embed_dim': self._embed_dim,
            'embed_regularizer': self._embed_regularizer,
            'embed_trainable': self._embed_trainable,
            'embed_numerical_embedding': self._embed_numerical_embedding,
            'fm_numerical_interactive': self._fm_numerical_interactive,
            'dnn_num_layers': self._dnn_num_layers,
            'dnn_dropout_rate': self._dnn_dropout_rate,
            'dnn_activation': self._dnn_activation
        })
        return config


def build_deepfm(config, categorical_columns_config, numerical_columns_config):

    _Linear = Linear(
        regularizer=config.get('linear_regularizer'),
        trainable=config.get('linear_trainable')
    )
    _Embedding = Embedding(
        embedding_dim=config.get('embed_dim'),
        regularizer=config.get('embed_regularizer'),
        trainable=config.get('embed_trainable'),
        numerical_embedding=config.get('embed_numerical_embedding')
    )
    _FM = FM(numerical_interactive=config.get('fm_numerical_interactive'))
    _DNN = FeedForwardDNN(
            config.get('dnn_num_layers'),
            dropout_rate=config.get('dnn_dropout_rate'),
            activation=config.get('dnn_activation')
    )
    _LR = LR()
    categorical_inputs = [
        tf.keras.layers.Input(shape=(len(conf.get('params'))), name=col)
        for col, conf in categorical_columns_config.items()
    ]
    numerical_inputs = [
        tf.keras.layers.Input(shape=(1,), name=col)
        for col, _ in numerical_columns_config.items()
    ]
    linear_outputs = _Linear([categorical_inputs, numerical_inputs])
    embeddings = _Embedding([categorical_inputs, numerical_inputs])
    fm_outputs = _FM(embeddings)
    dnn_outputs = _DNN(embeddings)
    outputs = _LR([linear_outputs, fm_outputs, dnn_outputs])
    model = tf.keras.models.Model(
        inputs=[categorical_inputs, numerical_inputs], outputs=outputs)
    return model
