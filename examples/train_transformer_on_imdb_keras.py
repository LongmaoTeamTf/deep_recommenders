#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from deep_recommenders.keras.models.nlp import Transformer


def load_dataset(vocab_size, max_len):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(maxlen=max_len, num_words=vocab_size)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
    x_train_masks = tf.equal(x_train, 0)
    x_test_masks = tf.equal(x_test, 0)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    return (x_train, x_train_masks, y_train), (x_test, x_test_masks, y_test)


def build_model(vocab_size, max_len, model_dim=8, n_heads=2, encoder_stack=2, decoder_stack=2, ff_size=50):
    encoder_inputs = tf.keras.Input(shape=(max_len,), name='encoder_inputs')
    decoder_inputs = tf.keras.Input(shape=(max_len,), name='decoder_inputs')
    outputs = Transformer(
        vocab_size,
        model_dim,
        n_heads=n_heads,
        encoder_stack=encoder_stack,
        decoder_stack=decoder_stack,
        feed_forward_size=ff_size
    )(encoder_inputs, decoder_inputs)
    outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(outputs)
    return tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)


def train_model(vocab_size=5000, max_len=128, batch_size=128, epochs=10):

    train, test = load_dataset(vocab_size, max_len)

    x_train, x_train_masks, y_train = train
    x_test, x_test_masks, y_test = test

    model = build_model(vocab_size, max_len)

    model.compile(optimizer=tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    es = tf.keras.callbacks.EarlyStopping(patience=3)
    model.fit([x_train, x_train_masks], y_train,
              batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[es])

    test_metrics = model.evaluate([x_test, x_test_masks], y_test, batch_size=batch_size, verbose=0)
    print("loss on Test: %.4f" % test_metrics[0])
    print("accu on Test: %.4f" % test_metrics[1])


if __name__ == '__main__':
    train_model()
