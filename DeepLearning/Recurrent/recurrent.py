'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-31 17:51:24
@LastEditors: Wang Yao
@LastEditTime: 2020-04-02 14:27:08
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

tf.config.experimental_run_functions_eagerly(True)

class BiDirectional(Layer):
    
    def __init__(self, rnn_cell, merge_mode='concat', **kwargs):
        super(BiDirectional, self).__init__(**kwargs)
        if merge_mode not in ['sum', 'mul', 'ave', 'concat', None]:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"sum", "mul", "ave", "concat", None}')
        self._rnn_cell = rnn_cell
        self._merge_mode = merge_mode
        self._return_outputs = rnn_cell._return_outputs
        self._return_states = rnn_cell._return_states

    def call(self, inputs):
        reverse_inputs = K.reverse(inputs, 1)
        rnn_fw_outputs = self._rnn_cell(inputs)
        rnn_bw_outputs = self._rnn_cell(reverse_inputs)
        if self._return_states:
            if self._return_outputs:
                _, fw_states = rnn_fw_outputs
                _, bw_states = rnn_bw_outputs
            else:
                fw_states = rnn_fw_outputs
                bw_states = rnn_bw_outputs
        else:
            raise ValueError("BiDirectional rnn cell must set `_return_states` to True.")
        
        fw_states = K.concatenate(fw_states, axis=-2)
        bw_states = K.concatenate(bw_states, axis=-2)
        fw_states, bw_states = K.reverse(fw_states, 1), K.reverse(bw_states, 1)
        
        if self._merge_mode == 'concat':
            outputs = K.concatenate([fw_states, bw_states], axis=-1)
        elif self._merge_mode == 'sum':
            outputs = fw_states + bw_states
        elif self._merge_mode == 'ave':
            outputs = (fw_states + bw_states) / 2
        elif self._merge_mode == 'mul':
            outputs = fw_states * bw_states
        elif self._merge_mode is None:
            outputs = [fw_states, bw_states]
        else:
            raise ValueError('Unrecognized value for argument '
                             'merge_mode: %s' % (self._merge_mode))
        return outputs


    def compute_output_shape(self, input_shape):
        output_shape = self._rnn_cell.compute_output_shape(input_shape)
        if self._return_states:
            if self._return_outputs:
                _, output_shape = output_shape
        if self._merge_mode == 'concat':
            output_shape = output_shape[:-1] + (output_shape[-1]*2)
        elif self._merge_mode is None:
            output_shape = [output_shape, output_shape]
        return output_shape