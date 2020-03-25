'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-25 13:55:59
@LastEditors: Wang Yao
@LastEditTime: 2020-03-25 16:32:45
'''
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


class Noam(Callback):

    def __init__(self, model_dim, step_num, warmup_steps=4000, verbose=False, **kwargs):
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
            lrate = K.get_value(K.model.optimizer.lr)
            print(f"epoch {epoch} lr: {lrate}")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(K.model.optimizer.lr)
    

def label_smoothing(inputs, epsilon=0.1):

    output_dim = inputs.shape[-1]
    smooth_label = (1 - epsilon) * inputs + (epsilon / output_dim)
    return smooth_label


if __name__ == "__main__":
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    from tensorflow.keras.optimizers import Adam
    from transformer import Transformer

    encoder_inputs = Input(shape=(256,), name='encoder_inputs')
    decoder_inputs = Input(shape=(256,), name='decoder_inputs')
    outputs = Transformer(1000, 512)([encoder_inputs, decoder_inputs])
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

    model.compile(
        optimizer=Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9), 
        loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    
    
    
    
