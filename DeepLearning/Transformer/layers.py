'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-22 17:48:05
@LastEditors: Wang Yao
@LastEditTime: 2020-03-25 18:05:37
'''
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer



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


class PositionEncoding(Layer):

    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        seq_length = inputs.shape[1]
        position_encodings = np.zeros((seq_length, self._model_dim))
        for pos in range(seq_length):
            for i in range(self._model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i-i%2) / self._model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2]) # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2]) # 2i+1
        position_encodings = K.cast(position_encodings, 'float32')
        return position_encodings

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention(Layer):

    def __init__(self, masking=True, future=False, dropout_rate=0., **kwargs):
        self._masking = masking
        self._future = future
        self._dropout_rate = dropout_rate
        self._masking_num = -2**32+1
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def mask(self, inputs, masks):
        masks = K.cast(masks, 'float32')
        masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])
        masks = K.expand_dims(masks, 1)
        outputs = inputs + masks * self._masking_num
        return outputs
    
    def future_mask(self, inputs):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
        paddings = tf.ones_like(future_masks) * self._masking_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        return outputs

    def call(self, inputs):
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs

        if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
        if K.dtype(keys) != 'float32':  keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':  values = K.cast(values, 'float32')

        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1])) # MatMul
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5  # Scale
        if self._masking:
            scaled_matmul = self.mask(scaled_matmul, masks) # Mask(opt.)

        if self._future:
            scaled_matmul = self.future_mask(scaled_matmul)

        softmax_out = K.softmax(scaled_matmul) # SoftMax
        # Dropout
        out = K.dropout(softmax_out, self._dropout_rate)
        
        outputs = K.batch_dot(out, values)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class MultiHeadAttention(Layer):

    def __init__(self, n_heads, head_dim, masking=True, future=False, trainable=True, **kwargs):
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._masking = masking
        self._future = future
        self._trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_values')
        super(MultiHeadAttention, self).build(input_shape)


    def call(self, inputs):
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs
        
        queries_linear = K.dot(queries, self._weights_queries) 
        keys_linear = K.dot(keys, self._weights_keys)
        values_linear = K.dot(values, self._weights_values)

        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)
        
        if self._masking:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads, masks]
        else:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]
            
        attention = ScaledDotProductAttention(masking=self._masking, future=self._future)
        att_out = attention(att_inputs)

        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)
        
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class PositionWiseFeedForward(Layer):
    
    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self._model_dim = model_dim
        self._inner_dim = inner_dim
        self._trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_inner")
        self.weights_out = self.add_weight(
            shape=(self._inner_dim, self._model_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_out")
        self.bais_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_inner")
        self.bais_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_out")
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bais_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bais_out
        return outputs

    def compute_output_shape(self, input_shape):
        return self._model_dim


class LayerNormalization(Layer):

    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta')
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class Add(Layer):

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def call(self, inputs):
        input_a, input_b = inputs
        return input_a + input_b

    def compute_output_shape(self, input_shape):
        return input_shape[0]


if __name__ == "__main__":
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing import sequence
    from tensorflow.keras.utils import to_categorical

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    vocab_size = 5000
    max_len = 256
    model_dim = 512
    batch_size = 256
    epochs = 20
    
    print("Data downloading and pre-processing ... ")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=max_len, num_words=vocab_size)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    x_train_masks = tf.equal(x_train, 0)
    x_test_masks = tf.equal(x_test, 0)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    print('Model building ... ')
    inputs = Input(shape=(max_len,), name="inputs")
    masks = Input(shape=(max_len,), name='masks')
    embeddings = Embedding(vocab_size, model_dim)(inputs)
    encodings = PositionEncoding(model_dim)(embeddings)
    encodings = Add()([embeddings, encodings])
    x = MultiHeadAttention(8, 64)([encodings, encodings, encodings, masks])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(10, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=[inputs, masks], outputs=outputs)
    model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9), 
        loss='categorical_crossentropy', metrics=['accuracy'])

    print("Model Training ... ")
    es = EarlyStopping(patience=5)
    model.fit([x_train, x_train_masks], y_train, 
        batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[es])
