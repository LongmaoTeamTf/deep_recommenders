'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-23 19:42:15
@LastEditors: Wang Yao
@LastEditTime: 2020-03-27 17:50:33
'''
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import Callback
from multi_head_attention import MultiHeadAttention


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


class Add(Layer):

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def call(self, inputs):
        input_a, input_b = inputs
        return input_a + input_b

    def compute_output_shape(self, input_shape):
        return input_shape[0]


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


class Transformer(Layer):

    def __init__(self, vocab_size, model_dim, 
            n_heads=8, encoder_stack=6, decoder_stack=6, feed_forward_size=2048, dropout_rate=0.1, **kwargs):
        self._vocab_size = vocab_size
        self._model_dim = model_dim
        self._n_heads = n_heads
        self._encoder_stack = encoder_stack
        self._decoder_stack = decoder_stack
        self._feed_forward_size = feed_forward_size
        self._dropout_rate = dropout_rate
        super(Transformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self._vocab_size, self._model_dim),
            initializer='glorot_uniform',
            trainable=True,
            name="embeddings")
        self.EncoderPositionEncoding = PositionEncoding(self._model_dim)
        self.EncoderMultiHeadAttetions = [
            MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
            for _ in range(self._encoder_stack)
        ]
        self.EncoderLayerNorms0 = [
            LayerNormalization()
            for _ in range(self._encoder_stack)
        ]
        self.EncoderPositionWiseFeedForwards = [
            PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            for _ in range(self._encoder_stack)
        ]
        self.EncoderLayerNorms1 = [
            LayerNormalization()
            for _ in range(self._encoder_stack)
        ]
        self.DecoderPositionEncoding = PositionEncoding(self._model_dim)
        self.DecoderMultiHeadAttetions0 = [
            MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads, future=True)
            for _ in range(self._decoder_stack)
        ]
        self.DecoderLayerNorms0 = [
            LayerNormalization()
            for _ in range(self._decoder_stack)
        ]
        self.DecoderMultiHeadAttetions1 = [
            MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
            for _ in range(self._decoder_stack)
        ]
        self.DecoderLayerNorms1 = [
            LayerNormalization()
            for _ in range(self._decoder_stack)
        ]
        self.DecoderPositionWiseFeedForwards = [
            PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            for _ in range(self._decoder_stack)
        ]
        self.DecoderLayerNorms2 = [
            LayerNormalization()
            for _ in range(self._decoder_stack)
        ]
        super(Transformer, self).build(input_shape)
        
    def encoder(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')

        masks = K.equal(inputs, 0)
        # Embeddings
        embeddings = K.gather(self.embeddings, inputs)
        embeddings *= self._model_dim ** 0.5 # Scale
        # Position Encodings
        position_encodings = self.EncoderPositionEncoding(embeddings)
        # Embedings + Postion-encodings
        encodings = embeddings + position_encodings
        # Dropout
        encodings = K.dropout(encodings, self._dropout_rate)

        for i in range(self._encoder_stack):
            # Multi-head-Attention
            attention = self.EncoderMultiHeadAttetions[i]
            attention_input = [encodings, encodings, encodings, masks]
            attention_out = attention(attention_input)
            # Add & Norm
            attention_out += encodings
            attention_out = self.EncoderLayerNorms0[i](attention_out)
            # Feed-Forward
            ff = self.EncoderPositionWiseFeedForwards[i]
            ff_out = ff(attention_out)
            # Add & Norm
            ff_out += attention_out
            encodings = self.EncoderLayerNorms1[i](ff_out)

        return encodings, masks

    def decoder(self, inputs):
        decoder_inputs, encoder_encodings, encoder_masks = inputs
        if K.dtype(decoder_inputs) != 'int32':
            decoder_inputs = K.cast(decoder_inputs, 'int32')

        decoder_masks = K.equal(decoder_inputs, 0)
        # Embeddings
        embeddings = K.gather(self.embeddings, decoder_inputs)
        embeddings *= self._model_dim ** 0.5 # Scale
        # Position Encodings
        position_encodings = self.DecoderPositionEncoding(embeddings)
        # Embedings + Postion-encodings
        encodings = embeddings + position_encodings
        # Dropout
        encodings = K.dropout(encodings, self._dropout_rate)
        
        for i in range(self._decoder_stack):
            # Masked-Multi-head-Attention
            masked_attention = self.DecoderMultiHeadAttetions0[i]
            masked_attention_input = [encodings, encodings, encodings, decoder_masks]
            masked_attention_out = masked_attention(masked_attention_input)
            # Add & Norm
            masked_attention_out += encodings
            masked_attention_out = self.DecoderLayerNorms0[i](masked_attention_out)

            # Multi-head-Attention
            attention = self.DecoderMultiHeadAttetions1[i]
            attention_input = [masked_attention_out, encoder_encodings, encoder_encodings, encoder_masks]
            attention_out = attention(attention_input)
            # Add & Norm
            attention_out += masked_attention_out
            attention_out = self.DecoderLayerNorms1[i](attention_out)

            # Feed-Forward
            ff = self.DecoderPositionWiseFeedForwards[i]
            ff_out = ff(attention_out)
            # Add & Norm
            ff_out += attention_out
            encodings = self.DecoderLayerNorms2[i](ff_out)

        # Pre-Softmax 与 Embeddings 共享参数
        linear_projection = K.dot(encodings, K.transpose(self.embeddings))
        outputs = K.softmax(linear_projection)
        return outputs

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder_encodings, encoder_masks = self.encoder(encoder_inputs)
        encoder_outputs = self.decoder([decoder_inputs, encoder_encodings, encoder_masks])
        return encoder_outputs

    def compute_output_shape(self, input_shape):
        return  (input_shape[0][0], input_shape[0][1], self._vocab_size)


class Noam(Callback):

    def __init__(self, model_dim, step_num=0, warmup_steps=4000, verbose=False, **kwargs):
        self._model_dim = model_dim
        self._step_num = step_num
        self._warmup_steps = warmup_steps
        self.verbose = verbose
        super(Noam, self).__init__(**kwargs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        init_lr = self._model_dim ** -.5 * self._warmup_steps ** -1.5
        K.set_value(self.model.optimizer.lr, init_lr)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self._step_num += 1
        lrate = self._model_dim ** -.5 * K.minimum(self._step_num ** -.5, self._step_num * self._warmup_steps ** -1.5)
        K.set_value(self.model.optimizer.lr, lrate)

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            lrate = K.get_value(self.model.optimizer.lr)
            print(f"epoch {epoch} lr: {lrate}")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
    

def label_smoothing(inputs, epsilon=0.1):
    """目标平滑"""
    output_dim = inputs.shape[-1]
    smooth_label = (1 - epsilon) * inputs + (epsilon / output_dim)
    return smooth_label


if __name__ == "__main__":
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing import sequence
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    vocab_size = 5000
    max_seq_len = 256
    model_dim = 8
    batch_size = 128
    epochs = 10

    print("Data downloading and pre-processing ... ")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=max_seq_len, num_words=vocab_size)
    x_train = sequence.pad_sequences(x_train, maxlen=max_seq_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_seq_len)
    x_train_masks = tf.equal(x_train, 0)
    x_test_masks = tf.equal(x_test, 0)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print('Model building ... ')
    encoder_inputs = Input(shape=(max_seq_len,), name='encoder_inputs')
    decoder_inputs = Input(shape=(max_seq_len,), name='decoder_inputs')
    outputs = Transformer(
        vocab_size, 
        model_dim, 
        n_heads=2, 
        encoder_stack=2,
        decoder_stack=2, 
        feed_forward_size=50
    )([encoder_inputs, decoder_inputs])
    outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(outputs)
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

    model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9), 
        loss='categorical_crossentropy', metrics=['accuracy'])

    print("Model Training ... ")
    es = EarlyStopping(patience=5)
    model.fit([x_train, x_train_masks], y_train, 
        batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[es])

    test_metrics = model.evaluate([x_test, x_test_masks], y_test, batch_size=batch_size, verbose=0)
    print("loss on Test: %.4f" % test_metrics[0])
    print("accu on Test: %.4f" % test_metrics[1])
    
