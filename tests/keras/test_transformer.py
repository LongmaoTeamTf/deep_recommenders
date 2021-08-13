#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True

import os
import tempfile
import numpy as np
import tensorflow as tf

from deep_recommenders.keras.models.nlp import Transformer


class TestTransformer(tf.test.TestCase):

    def test_save_model(self):

        def get_model():
            encoder_inputs = tf.keras.Input(shape=(256,), name='encoder_inputs')
            decoder_inputs = tf.keras.Input(shape=(256,), name='decoder_inputs')
            outputs = Transformer(5000,
                                  model_dim=8,
                                  n_heads=2,
                                  encoder_stack=2,
                                  decoder_stack=2,
                                  feed_forward_size=50)(encoder_inputs, decoder_inputs)
            outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(outputs)
            return tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

        model = get_model()
        encoder_random_input = np.random.randint(size=(10, 256), low=0, high=5000)
        decoder_random_input = np.random.randint(size=(10, 256), low=0, high=5000)
        model_pred = model.predict([encoder_random_input, decoder_random_input])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "transformer_model")
            model.save(path)
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict([encoder_random_input, decoder_random_input])
        for i in range(len(model.layers)):
            assert model.layers[i].get_config() == loaded_model.layers[i].get_config()
        self.assertAllClose(model_pred, loaded_pred)


if __name__ == '__main__':
    tf.test.main()
