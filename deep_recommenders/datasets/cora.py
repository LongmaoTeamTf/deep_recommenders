#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.sparse as sp


class Cora(object):

    def __init__(self, extract_path="."):
        self._download_url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
        self._extract_path = extract_path
        self._cora_path = os.path.join(extract_path, "cora")
        self._cora_cites = os.path.join(self._cora_path, "cora.cites")
        self._cora_content = os.path.join(self._cora_path, "cora.content")

        if not os.path.exists(self._cora_cites) or \
                not os.path.exists(self._cora_content):
            self._download()

        self._cora_classes = [
            "Case_Based",
            "Genetic_Algorithms",
            "Neural_Networks",
            "Probabilistic_Methods",
            "Reinforcement_Learning",
            "Rule_Learning",
            "Theory"
        ]

    @property
    def num_classes(self):
        return len(self._cora_classes)

    def _download(self, filename="cora.tgz"):
        import requests
        import tarfile
        r = requests.get(self._download_url)
        with open(filename, "wb") as f:
            f.write(r.content)
        tarobj = tarfile.open(filename, "r:gz")
        for tarinfo in tarobj:
            tarobj.extract(tarinfo.name, self._extract_path)
        tarobj.close()

    def load_content(self, normalize=True):
        content = np.genfromtxt(self._cora_content, dtype=np.str)
        ids, features, labels = content[:, 0], content[:, 1:-1], content[:, -1]
        features = sp.csr_matrix(features, dtype=np.float32)
        if normalize is True:
            features /= features.sum(axis=1).reshape(-1, 1)
        return ids, features, labels

    def build_graph(self, nodes):
        idx_map = {int(j): i for i, j in enumerate(nodes)}
        edges_unordered = np.genfromtxt(self._cora_cites, dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        graph = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                              shape=(nodes.shape[0], nodes.shape[0]), dtype=np.float32)
        graph += graph.T - sp.diags(graph.diagonal())  # Convert symmetric matrix
        return graph

    @staticmethod
    def spectral_graph(graph):
        graph = graph + sp.eye(graph.shape[0])  # graph G with added self-connections
        # D^{-1/2} * A * D^{-1/2}
        d = sp.diags(np.power(np.array(graph.sum(1)), -0.5).flatten(), 0)
        spectral_graph = graph.dot(d).transpose().dot(d).tocsr()
        return spectral_graph

    def sample_train_nodes(self, labels, num_per_class=20):
        train_nodes = []
        for cls in self._cora_classes:
            cls_index = np.where(labels == cls)[0]
            cls_sample = np.random.choice(cls_index, num_per_class, replace=False)
            train_nodes += cls_sample.tolist()
        return train_nodes

    def encode_labels(self, labels):
        labels_map = {}
        num_classes = len(self._cora_classes)
        for i, cls in enumerate(self._cora_classes):
            cls_label = np.zeros(shape=(num_classes,))
            cls_label[i] = 1.
            labels_map[cls] = cls_label
        encoded_labels = list(map(labels_map.get, labels))
        return np.array(encoded_labels, dtype=np.int32)

    def split_labels(self, labels, num_valid_nodes=500):
        num_nodes = labels.shape[0]
        all_index = np.arange(num_nodes)
        train_index = self.sample_train_nodes(labels)
        valid_index = list(set(all_index) - set(train_index))
        valid_index, test_index = valid_index[:num_valid_nodes], valid_index[num_valid_nodes:]

        encoded_labels = self.encode_labels(labels)

        def _sample_mask(index_ls):
            mask = np.zeros(num_nodes)
            mask[index_ls] = 1
            return np.array(mask, dtype=np.bool)

        def _get_labels(index_ls):
            _labels = np.zeros(encoded_labels.shape, dtype=np.int32)
            _labels[index_ls] = encoded_labels[index_ls]
            _mask = _sample_mask(index_ls)
            return _labels, _mask

        train_labels, train_mask = _get_labels(train_index)
        valid_labels, valid_mask = _get_labels(valid_index)
        test_labels, test_mask = _get_labels(test_index)

        return (train_labels, train_mask), \
               (valid_labels, valid_mask), \
               (test_labels, test_mask)
