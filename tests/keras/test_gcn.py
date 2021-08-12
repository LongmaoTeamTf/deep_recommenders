#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True

import os
import tempfile
import numpy as np
import scipy as sp
import tensorflow as tf
from absl.testing import parameterized

from deep_recommenders.keras.models.retrieval import GCN


class TestGCN(tf.test.TestCase, parameterized.TestCase):

    def test_gcn_adj_sparse_matrix(self):
        adj = np.asarray([
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1]
        ]).astype(np.float32)
        embeddings = np.asarray([
            [0.1, 0.2, 0.3, 0.0],
            [0.4, 0.5, 0.6, 0.0],
            [0.7, 0.8, 0.9, 0.0]
        ]).astype(np.float32)
        
        W = np.ones(shape=(4, 2))
        agg_embeddings = adj @ embeddings
        dense_outputs = agg_embeddings @ W
        expect_outputs = tf.nn.relu(dense_outputs)
        
        coo = sp.sparse.coo_matrix(adj)
        indices = np.mat([coo.row, coo.col]).transpose()
        sparse_adj = tf.SparseTensor(indices, coo.data, coo.shape)
    
        outputs = GCN(2, kernel_init="ones")(embeddings, sparse_adj)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(outputs, expect_outputs)

    def test_gcn_adj_full_matrix(self):
        adj = np.asarray([
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1]
        ]).astype(np.float32)
        embeddings = np.asarray([
            [0.1, 0.2, 0.3, 0.0],
            [0.4, 0.5, 0.6, 0.0],
            [0.7, 0.8, 0.9, 0.0]
        ]).astype(np.float32)
        
        W = np.ones(shape=(4, 2))
        agg_embeddings = adj @ embeddings
        dense_outputs = agg_embeddings @ W
        expect_outputs = tf.nn.relu(dense_outputs)
        
        outputs = GCN(2, kernel_init="ones")(embeddings, adj)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(outputs, expect_outputs)

    @parameterized.parameters(
        (8, 4),
        (16, 8),
        (32, 16),
    )
    def test_gcn_train(self, num_nodes, embeddings_dim):
        
        def get_model():
            adj = tf.keras.layers.Input(shape=(num_nodes,), sparse=True)
            embeddings = tf.keras.layers.Input(shape=(embeddings_dim,))

            x = GCN(16)(embeddings, adj)
            x = GCN(16)(x, adj)
            outputs = GCN(2, activation="softmax")(x, adj)
            return tf.keras.Model([adj, embeddings], outputs)

        np.random.seed(42)

        adj = sp.sparse.random(num_nodes, num_nodes).tocsr()
        adj.sort_indices()
        
        embeddings = np.random.normal(size=(num_nodes, embeddings_dim)).astype(np.float32)
        
        targets = np.random.randint(2, size=num_nodes).astype(np.float32)
        targets = np.stack([targets, 1 - targets], axis=1)

        model = get_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="categorical_crossentropy")
        model.fit(x=[adj, embeddings], y=targets, batch_size=num_nodes, verbose=0, shuffle=False)

        model_pred = model.predict([adj, embeddings])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gcn")
            model.save(
                path,
                options=tf.saved_model.SaveOptions(namespace_whitelist=["GCN"]))
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict([adj, embeddings], batch_size=num_nodes)
        for model_layer, loaded_layer in zip(model.layers, loaded_model.layers):
            assert model_layer.get_config() == loaded_layer.get_config()
        self.assertAllEqual(model_pred, loaded_pred)


if __name__ == "__main__":
    tf.test.main()