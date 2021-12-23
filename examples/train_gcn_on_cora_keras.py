#!/usr/bin/python3
# -*- coding: utf-8 -*-

# TensorFlow 2.3 Passing.

import tensorflow as tf

from deep_recommenders.datasets import Cora
from deep_recommenders.keras.models.retrieval import GCN


def train_model():
    cora = Cora()
    ids, features, labels = cora.load_content()
    graph = cora.build_graph(ids)
    spectral_graph = cora.spectral_graph(graph)
    cora.sample_train_nodes(labels)
    train, valid, test = cora.split_labels(labels)

    def build_model():
        g = tf.keras.layers.Input(shape=(None,))
        feats = tf.keras.layers.Input(shape=(features.shape[-1],))
        x = GCN(32)(feats, g)
        outputs = GCN(cora.num_classes, activation="softmax")(x, g)
        return tf.keras.Model([g, feats], outputs)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss="categorical_crossentropy",
        weighted_metrics=["acc"]
    )

    train_labels, train_mask = train
    valid_labels, valid_mask = valid
    test_labels, test_mask = test

    batch_size = graph.shape[0]

    model.fit([spectral_graph, features],
              train_labels,
              sample_weight=train_mask,
              validation_data=([spectral_graph, features], valid_labels, valid_mask),
              batch_size=batch_size,
              epochs=200,
              shuffle=False,
              verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])

    eval_results = model.evaluate([spectral_graph, features],
                                  test_labels,
                                  sample_weight=test_mask,
                                  batch_size=batch_size,
                                  verbose=0)
    print("Test Loss: {:.4f}".format(eval_results[0]))
    print("Test Accuracy: {:.4f}".format(eval_results[1]))


def get_embeddings(model, graph, features):
    input_layer, output_layer = model.input, model.layers[-1].output
    embedding_model = tf.keras.Model(input_layer, output_layer)
    embeddings = embedding_model.predict([graph, features], batch_size=graph.shape[0])
    return embeddings


if __name__ == "__main__":
    train_model()
