'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-30 15:47:00
@LastEditors: Wang Yao
@LastEditTime: 2020-03-31 18:47:15
'''
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer


tf.config.experimental_run_functions_eagerly(True)

class Embedding(Layer):

    def __init__(self, vocab_size, model_dim, **kwargs):
        self._vocab_size = vocab_size
        self._model_dim = model_dim
        super(Embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self._vocab_size, self._model_dim),
            initializer='glorot_uniform',
            name="embeddings")
        super(Embedding, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        embeddings = K.gather(self.embeddings, inputs)
        embeddings *= self._model_dim ** 0.5 # Scale
        return embeddings

    def compute_output_shape(self, input_shape):

        return input_shape + (self._model_dim,)


class RNN(Layer):
    
    def __init__(self, 
            kernel_dim,
            activation='tanh',
            return_outputs=False, 
            return_states=False,
            use_bias=True,
            **kwargs):
        super(RNN, self).__init__(**kwargs)
        self._kernel_dim = kernel_dim
        self._activation = activations.get(activation)
        self._return_outputs = return_outputs
        self._return_states = return_states
        self._use_bias = use_bias
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.U = self.add_weight(
            shape=(input_dim, self._kernel_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_input')
        self.W = self.add_weight(
            shape=(self._kernel_dim, self._kernel_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_hidden')
        if self._use_bias:
            self.b = self.add_weight(
                shape=(self._kernel_dim),
                initializer='zeros',
                trainable=True,
                name='bias_b')
        if self._return_outputs:
            self.V = self.add_weight(
                shape=(self._kernel_dim, self._kernel_dim),
                initializer='glorot_uniform',
                trainable=True,
                name='weights_output')
            if self._use_bias:
                self.c = self.add_weight(
                shape=(self._kernel_dim),
                initializer='zeros',
                trainable=True,
                name='bias_c')
        
        super(RNN, self).build(input_shape)

    def call(self, inputs):
        h_t = K.zeros((1, self._kernel_dim))
        ots, hts = [], []
        for t in range(inputs.shape[1]):
            x_t = K.expand_dims(inputs[:, t, :], 1)
            a_t = K.dot(x_t, self.U) + K.dot(h_t, self.W)
            if self._use_bias: a_t += self.b

            if self._activation is not None:
                h_t = self._activation(a_t)
                hts.append(h_t)

            if self._return_outputs:
                o_t = K.dot(h_t, self.V)
                if self._use_bias: o_t += self.c
                o_t = K.softmax(o_t)
                ots.append(o_t)
        outputs = h_t
        if self._return_outputs:
            outputs = ots
        if self._return_states:
            outputs = hts
        if self._return_outputs and self._return_states:
            outputs = ots, hts 
        return outputs

    def compute_output_shape(self, input_shape):
        return [input_shape[:-1] + (self._kernel_dim,), 
                    input_shape[:-1] + (self._kernel_dim,)]


class Attention(Layer):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attetion = self.add_weight(
            shape=(input_shape[1],),
            initializer='glorot_uniform',
            trainable=True,
            name='attetion')
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        attetion = K.softmax(self.attetion)
        outputs = K.dot(attetion, inputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])



if __name__ == "__main__":
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing import sequence
    from tensorflow.keras.utils import to_categorical
    from recurrent import BiDirectional
    # from Embeddings.embeddings import Embedding
    

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    vocab_size = 5000
    max_len = 256
    model_dim = 64
    batch_size = 128
    epochs = 10

    print("Data downloading and pre-processing ... ")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=max_len, num_words=vocab_size)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print('Model building ... ')
    inputs = Input(shape=(max_len,), name="inputs")
    embeddings = Embedding(vocab_size, model_dim)(inputs)
    outputs = BiDirectional(RNN(model_dim, return_states=True))(embeddings)
    x = Attention()(outputs)
    x = Dropout(0.2)(x)
    x = Dense(10, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9), 
        loss='categorical_crossentropy', metrics=['accuracy'])

    print("Model Training ... ")
    es = EarlyStopping(patience=5)
    model.fit(x_train, y_train, 
        batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[es])

    test_metrics = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print("loss on Test: %.4f" % test_metrics[0])
    print("accu on Test: %.4f" % test_metrics[1])

    