'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-23 19:42:15
@LastEditors: Wang Yao
@LastEditTime: 2020-03-24 17:51:07
'''
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from layers import PositionEncoding
from layers import MultiHeadAttention, PositionWiseFeedForward
from layers import LayerNormalization



class Transformer(tf.keras.layers.Layer):

    def __init__(self, 
            vocab_size, 
            model_dim, 
            n_heads=8, 
            encoder_stack=6, 
            decoder_stack=6, 
            feed_forward_size=2048, **kwargs):
        self._vocab_size = vocab_size
        self._model_dim = model_dim
        self._n_heads = n_heads
        self._encoder_stack = encoder_stack
        self._decoder_stack = decoder_stack
        self._feed_forward_size = feed_forward_size
        
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
        embeddings_out = K.gather(self.embeddings, inputs)
        embeddings_out *= self._model_dim ** 0.5 # Scale
        # Position Encodings
        position_encodings = PositionEncoding(self._model_dim)(embeddings_out)
        # Embedings + Postion-encodings
        encodings = embeddings_out + position_encodings

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
            inputs = K.cast(decoder_inputs, 'int32')

        decoder_masks = K.equal(decoder_inputs, 0)
        # Embeddings
        embeddings_out = K.gather(self.embeddings, decoder_inputs)
        embeddings_out *= self._model_dim ** 0.5 # Scale
        # Position Encodings
        position_encodings = PositionEncoding(self._model_dim)(embeddings_out)
        # Embedings + Postion-encodings
        encodings = embeddings_out + position_encodings
        
        for i in range(self._decoder_stack):
            # Masked-Multi-head-Attention
            masked_attention = MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
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
        return input_shape
    

if __name__ == "__main__":
    transformer = Transformer(1000, 512)
    encoder_inputs = np.array([[1, 2, 3, 0, 0],[4, 5 ,6, 0, 0],[7, 8, 0, 0, 0]])
    decoder_inputs = np.array([[1, 2, 3, 0, 0],[4, 5 ,6, 0, 0],[7, 8, 0, 0, 0]])
    outputs = transformer([encoder_inputs, decoder_inputs])
    print(outputs)