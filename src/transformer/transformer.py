'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-23 19:42:15
@LastEditors: Wang Yao
@LastEditTime: 2020-03-27 17:50:33
'''
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from layers import PositionEncoding
from layers import MultiHeadAttention, PositionWiseFeedForward
from layers import Add, LayerNormalization

tf.config.experimental_run_functions_eagerly(True)

class Transformer(tf.keras.layers.Layer):

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
        super(Transformer, self).build(input_shape)


    def encoder(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')

        masks = K.equal(inputs, 0)
        # Embeddings
        embeddings = K.gather(self.embeddings, inputs)
        embeddings *= self._model_dim ** 0.5 # Scale
        # Position Encodings
        position_encodings = PositionEncoding(self._model_dim)(embeddings)
        # Embedings + Postion-encodings
        encodings = embeddings + position_encodings
        # Dropout
        encodings = K.dropout(encodings, self._dropout_rate)

        for i in range(self._encoder_stack):
            # Multi-head-Attention
            attention = MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
            attention_input = [encodings, encodings, encodings, masks]
            attention_out = attention(attention_input)
            # Add & Norm
            attention_out += encodings
            attention_out = LayerNormalization()(attention_out)
            # Feed-Forward
            ff = PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            ff_out = ff(attention_out)
            # Add & Norm
            ff_out += attention_out
            encodings = LayerNormalization()(ff_out)

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
        position_encodings = PositionEncoding(self._model_dim)(embeddings)
        # Embedings + Postion-encodings
        encodings = embeddings + position_encodings
        # Dropout
        encodings = K.dropout(encodings, self._dropout_rate)
        
        for i in range(self._decoder_stack):
            # Masked-Multi-head-Attention
            masked_attention = MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads, future=True)
            masked_attention_input = [encodings, encodings, encodings, decoder_masks]
            masked_attention_out = masked_attention(masked_attention_input)
            # Add & Norm
            masked_attention_out += encodings
            masked_attention_out = LayerNormalization()(masked_attention_out)

            # Multi-head-Attention
            attention = MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
            attention_input = [masked_attention_out, encoder_encodings, encoder_encodings, encoder_masks]
            attention_out = attention(attention_input)
            # Add & Norm
            attention_out += masked_attention_out
            attention_out = LayerNormalization()(attention_out)

            # Feed-Forward
            ff = PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            ff_out = ff(attention_out)
            # Add & Norm
            ff_out += attention_out
            encodings = LayerNormalization()(ff_out)

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

    output_dim = inputs.shape[-1]
    smooth_label = (1 - epsilon) * inputs + (epsilon / output_dim)
    return smooth_label


if __name__ == "__main__":
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    from tensorflow.keras.utils import plot_model

    vocab_size = 5000
    max_seq_len = 256
    model_dim = 512

    encoder_inputs = Input(shape=(max_seq_len,), name='encoder_inputs')
    decoder_inputs = Input(shape=(max_seq_len,), name='decoder_inputs')
    outputs = Transformer(vocab_size, model_dim)([encoder_inputs, decoder_inputs])
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

    model.summary()
    plot_model(model, 'transformer.png')
