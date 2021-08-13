#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import Callback

from deep_recommenders.keras.models.nlp import MultiHeadAttention


@tf.keras.utils.register_keras_serializable()
class PositionEncoding(Layer):

    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
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


@tf.keras.utils.register_keras_serializable()
class Add(Layer):

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        input_a, input_b = inputs
        return input_a + input_b

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@tf.keras.utils.register_keras_serializable()
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
        self.bias_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bias_inner")
        self.bias_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bias_out")
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bias_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bias_out
        return outputs

    def compute_output_shape(self, input_shape):
        return self._model_dim


@tf.keras.utils.register_keras_serializable()
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

    def call(self, inputs, **kwargs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable()
class Transformer(Layer):

    def __init__(self,
                 vocab_size,
                 model_dim,
                 n_heads=8,
                 encoder_stack=6,
                 decoder_stack=6,
                 feed_forward_size=2048,
                 dropout_rate=0.1,
                 **kwargs):

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
        self.EncoderMultiHeadAttentions = [
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
        self.DecoderMultiHeadAttentions0 = [
            MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads, future=True)
            for _ in range(self._decoder_stack)
        ]
        self.DecoderLayerNorms0 = [
            LayerNormalization()
            for _ in range(self._decoder_stack)
        ]
        self.DecoderMultiHeadAttentions1 = [
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
        # Embeddings + Position-encodings
        encodings = embeddings + position_encodings
        # Dropout
        encodings = K.dropout(encodings, self._dropout_rate)

        for i in range(self._encoder_stack):
            # Multi-head-Attention
            attention = self.EncoderMultiHeadAttentions[i]
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
        # Embeddings + Position-encodings
        encodings = embeddings + position_encodings
        # Dropout
        encodings = K.dropout(encodings, self._dropout_rate)
        
        for i in range(self._decoder_stack):
            # Masked-Multi-head-Attention
            masked_attention = self.DecoderMultiHeadAttentions0[i]
            masked_attention_input = [encodings, encodings, encodings, decoder_masks]
            masked_attention_out = masked_attention(masked_attention_input)
            # Add & Norm
            masked_attention_out += encodings
            masked_attention_out = self.DecoderLayerNorms0[i](masked_attention_out)

            # Multi-head-Attention
            attention = self.DecoderMultiHeadAttentions1[i]
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

        # Pre-SoftMax 与 Embeddings 共享参数
        linear_projection = K.dot(encodings, K.transpose(self.embeddings))
        outputs = K.softmax(linear_projection)
        return outputs

    def call(self, encoder_inputs, decoder_inputs, **kwargs):
        encoder_encodings, encoder_masks = self.encoder(encoder_inputs)
        encoder_outputs = self.decoder([decoder_inputs, encoder_encodings, encoder_masks])
        return encoder_outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self._vocab_size

    def get_config(self):
        config = {
            "vocab_size": self._vocab_size,
            "model_dim": self._model_dim,
            "n_heads": self._n_heads,
            "encoder_stack": self._encoder_stack,
            "decoder_stack": self._decoder_stack,
            "feed_forward_size": self._feed_forward_size,
            "dropout_rate": self._dropout_rate
        }
        base_config = super(Transformer, self).get_config()
        return {**base_config, **config}


class Noam(Callback):

    def __init__(self, model_dim, step_num=0, warmup_steps=4000, verbose=False):
        self._model_dim = model_dim
        self._step_num = step_num
        self._warmup_steps = warmup_steps
        self.verbose = verbose
        super(Noam, self).__init__()

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

