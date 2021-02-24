"""
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2021-02-18 14:13:41
@LastEditors: Wang Yao
@LastEditTime: 2021-02-24 10:52:15
"""
from typing import Dict, Optional, Text, Tuple, Union

import abc
import contextlib
import tensorflow as tf

import os
import mkl
import faiss
import numpy as np
from absl.testing import parameterized



@contextlib.contextmanager
def _warp_batch_too_small_error(k: int):
    """ Candidate batch too small error """
    try:
        yield
    except tf.errors.InvalidArgumentError as e:
        error_msg = str(e)
        if "input must have at least k columns" in error_msg:
            raise ValueError("Tried to retrieve k={k} top items, but candidate batch too small."
                             "To resolve this, 1. increase batch-size, 2. set `drop_remainder`=True, "
                             "3. set `handle_incomplete_batches`=True in constructor.".format(k=k))


def _take_long_axis(arr: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
    """从原始数据arr中，根据indices指定的下标，取出元素
    Args:
        arr: 原始数据，2D
        indices: 下标，2D
    Returns:
        根据下标取出的数据，2D
    """
    row_indices = tf.tile(
        tf.expand_dims(tf.range(tf.shape(indices)[0]), 1),
        [1, tf.shape(indices)[1]])
    gather_indices = tf.concat(
        [tf.reshape(row_indices, (-1, 1)),
        tf.reshape(indices, (-1, 1))], axis=1)

    return tf.reshape(tf.gather_nd(arr, gather_indices), tf.shape(indices))


def _exclude(scores: tf.Tensor, 
             identifiers: tf.Tensor, 
             exclude: tf.Tensor, 
             k: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """从TopK中的items移除指定的候选item
    Args:
        scores: candidate scores. 2D
        identifiers: candidate identifiers. 2D
        exclude: identifiers to exclude. 2D
        k: 返回候选个数
    Returns:
        Tuple(top k candidates scores, top k candidates indentifiers)   
    """
    indents = tf.expand_dims(identifiers, -1)
    exclude = tf.expand_dims(exclude, 1)

    isin = tf.math.reduce_any(tf.math.equal(indents, exclude), -1)

    adjusted_scores = (scores - tf.cast(isin, tf.float32) * 1.0e5)

    k = tf.math.minimum(k, tf.shape(scores)[1])

    _, indices = tf.math.top_k(adjusted_scores, k=k)
    return _take_long_axis(scores, indices), _take_long_axis(identifiers, indices)


class TopK(tf.keras.Model, abc.ABC):
    """TopK layer 接口
    注意，必须实现两个方法
    1、index: 创建索引
    2、call: 检索索引
    """

    def __init__(self, k: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._k = k

    @abc.abstractmethod
    def index(self, 
              candidates: Union[tf.Tensor, tf.data.Dataset],
              identifiers: Optional[Union[tf.Tensor, tf.data.Dataset]] = None) -> "TopK":
        """创建索引 
        args:
            candidates: 候选 embeddings
            indentifiers: 候选 embeddings对应标识 (Opt)
        returns:
            Self.
        """

        raise NotImplementedError("Implementers must provide `index` method.")

    @abc.abstractmethod
    def call(self,
             queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
             k: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """检索索引
        args:
            queries: queries embeddings,
            k: 返回候选个数
        returns:
            Tuple(top k candidates scores, top k candidates indentifiers)
        """

        raise NotImplementedError()

    @tf.function
    def query_with_exclusions(  # pylint: disable=method-hidden
            self,
            queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
            exclusions: tf.Tensor,
            k: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """检索索引并过滤exclusions
        Args:
            queries: queries embeddings,
            exclusions: candidates identifiers. 从TopK的候选集中过滤指定的item.
            k: 返回候选个数
        Returns:
            Tuple(top k candidates scores, top k candidates indetifiers)
        """
        k = k if k is not None else self._k

        adjusted_k = k + exclusions.shape[1]
        scores, indentifiers = self(queries=queries, k=adjusted_k)
        return _exclude(scores, indentifiers, exclusions, adjusted_k)

    def _reset_tf_function_cache(self):
        """Resets the tf.function cache."""
        
        if hasattr(self.query_with_exclusions, "python_function"):
            self.query_with_exclusions = tf.function(
                self.query_with_exclusions.python_function)


class Streaming(TopK):
    """Retrieves top k scoring items and identifiers from large dataset."""

    def __init__(self, 
                 k: int = 10, 
                 query_model: Optional[tf.keras.Model] = None,
                 handle_incomplete_batches: bool = True,
                 num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
                 sorted_order: bool = True,
                 *args, 
                 **kwargs):
        super().__init__(k, *args, **kwargs)

        self._query_model = query_model
        self._handle_incomplete_batches = handle_incomplete_batches
        self._num_parallel_calls = num_parallel_calls
        self._sorted_order = sorted_order

        self._candidates = None
        self._identifiers = None

        self._counter = self.add_weight("counter", dtype=tf.int32, trainable=False)

    def index(self, 
              candidates: tf.data.Dataset, 
              identifiers: Optional[tf.data.Dataset] = None) -> "Streaming":
        """构建索引
        Args:
            candidates: 候选embeddings的Dataset
            indentifiers: 候选 embeddings对应标识的Dataset(Opt)
        Returns:
            Self.
        """
        self._candidates = candidates
        self._identifiers = identifiers

        return self

    def call(self,
             queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
             k: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """检索索引
        args:
            queries: queries embeddings,
            k: 返回候选个数
        returns:
            Tuple(top k candidates scores, top k candidates indentifiers)
        """
        k = k if k is not None else self._k

        if self._candidates is None:
            raise ValueError("The `index` method must be called first to "
                             "create the retrieval index.")
        
        if self._query_model is not None:
            queries = self._query_model(queries)

        # 重置计数器
        self._counter.assign(0)

        def top_scores(candidate_index: tf.Tensor,
                       candidate_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            """计算一个batch的候选集中的topK的scores和indices"""
            scores = tf.matmul(queries, candidate_batch, transpose_b=True)

            if self._handle_incomplete_batches is True:
                k_ = tf.math.minimum(k, tf.shape(scores)[1])
            else:
                k_ = k
            
            scores, indices = tf.math.top_k(scores, k=k_, sorted=self._sorted_order)

            return scores, tf.gather(candidate_index, indices)

        def top_k(state: Tuple[tf.Tensor, tf.Tensor],
                  x: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
            """Reduction fucntion.
            合并现在的topk和新的topk，重新从中选出topk
            """
            state_scores, state_indices = state
            x_scores, x_indices = x

            joined_scores = tf.concat([state_scores, x_scores], axis=1)
            joined_indices = tf.concat([state_indices, x_indices], axis=1)

            if self._handle_incomplete_batches is True:
                k_ = tf.math.minimum(k, tf.shape(joined_scores)[1])
            else:
                k_ = k
            
            scores, indices = tf.math.top_k(joined_scores, k=k_, sorted=self._sorted_order)

            return scores, tf.gather(joined_indices, indices, batch_dims=1)

        # 初始化state
        if self._identifiers is not None:
            index_dtype = self._identifiers.element_spec.dtype
        else:
            index_dtype = tf.int32
        initial_state = (tf.zeros((tf.shape(queries)[0], 0), dtype=tf.float32),
                         tf.zeros((tf.shape(queries)[0], 0), dtype=index_dtype))
        
        def enumerate_rows(batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            """Enumerates rows in each batch using a total element counter."""
            starting_counter = self._counter.read_value()
            end_counter = self._counter.assign_add(tf.shape(batch)[0])

            return tf.range(starting_counter, end_counter), batch

        if self._identifiers is not None:
            dataset = tf.data.Dataset.zip((self._identifiers, self._candidates))
        else:
            dataset = self._candidates.map(enumerate_rows)

        with _warp_batch_too_small_error(k):
            result = (dataset
                # Map: 计算每个batch的TopK
                .map(top_scores, num_parallel_calls=self._num_parallel_calls)
                # Reduce: 计算全局TopK
                .reduce(initial_state, top_k))
        return result


class BruteForce(TopK):
    """暴力检索"""

    def __init__(self, 
                 k: int = 10,
                 query_model: Optional[tf.keras.Model] = None,
                 *args, 
                 **kwargs):
        super().__init__(k, *args, **kwargs)

        self._query_model = query_model


    def index(self, 
              candidates: Union[tf.Tensor, tf.data.Dataset],
              identifiers: Optional[Union[tf.Tensor, tf.data.Dataset]] = None) -> "BruteForce":
        
        if isinstance(candidates, tf.data.Dataset):
            candidates = tf.concat(list(candidates), axis=0)

        if identifiers is None:
            identifiers = tf.range(candidates.shape[0])

        if isinstance(identifiers, tf.data.Dataset):
            identifiers = tf.concat(list(identifiers), axis=0)

        if tf.rank(candidates) != 2:
            raise ValueError("`candidates` ndim should be 2. "
                             "Got `ndim` = {}".format(tf.rank(candidates)))

        self._candidates = self.add_weight(
            name="candidates",
            dtype=candidates.dtype,
            shape=candidates.shape,
            initializer=tf.keras.initializers.Zeros(),
            trainable=False
        )

        identifiers_initial_value = tf.zeros((), dtype=identifiers.dtype)

        self._identifiers = self.add_weight(
            name="identifiers",
            dtype=identifiers.dtype,
            shape=identifiers.shape,
            initializer=tf.keras.initializers.Constant(value=identifiers_initial_value),
            trainable=False
        )

        self._candidates.assign(candidates)
        self._identifiers.assign(identifiers)

        self._reset_tf_function_cache()
        return self

    def call(self,
             queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
             k: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        
        k = k if k is not None else self._k

        if self._candidates is None:
            raise ValueError("The `index` method must be called first to "
                             "create the retrieval index.")
        
        if self._query_model is not None:
            queries = self._query_model(queries)

        scores = tf.matmul(queries, self._candidates, transpose_b=True)
        
        scores, indices = tf.math.top_k(scores, k=k)

        return scores, tf.gather(self._identifiers, indices)


class ScaNN(TopK):
    """(Google)ScaNN approximate retrieval index for a factorized retrieval model"""


class Faiss(TopK):
    """(Facebook)Faiss retrieval index for a factorized retrieval model"""

    def __init__(self, 
                 k: int = 10,
                 query_model: Optional[tf.keras.Model] = None,
                 nlist: Optional[int] = 1,
                 nprobe: Optional[int] = 1,
                 normalize: bool = False,
                 *args, 
                 **kwargs):
        super().__init__(k, *args, **kwargs)

        self._query_model = query_model
        self._nlist = nlist
        self._nprobe = nprobe
        self._normalize = normalize

        mkl.get_max_threads()

        def build_searcher(
            candidates: Union[np.ndarray, tf.Tensor],
            identifiers: Optional[Union[np.ndarray, tf.Tensor]] = None,
        ) -> Union[faiss.swigfaiss.IndexIDMap, faiss.swigfaiss.IndexIVFFlat]:
            
            if isinstance(candidates, tf.Tensor):
                candidates = candidates.numpy()

            if candidates.dtype != "float32":
                candidates = candidates.astype(np.float32)
    
            d = candidates.shape[1]
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, self._nlist, faiss.METRIC_INNER_PRODUCT)
            if self._normalize is True:
                faiss.normalize_L2(candidates)
            index.train(candidates) # pylint: disable=no-value-for-parameter

            if identifiers is not None:
                if isinstance(identifiers, tf.Tensor):
                    identifiers = identifiers.numpy()
                if identifiers.dtype != np.int64:
                    try:
                        identifiers = identifiers.astype(np.int64)
                    except:
                        raise ValueError("`identifiers` dtype must be `int64`."
                                    "Got `dtype` = {}".format(identifiers.dtype))

                index = faiss.IndexIDMap(index)
                index.add_with_ids(candidates, identifiers) # pylint: disable=no-value-for-parameter
            else:
                index.add(candidates)

            return index

        self._build_searcher = build_searcher
        self._searcher = None
        self._identifiers = None
            
    def index(self, 
              candidates: Union[tf.Tensor, tf.data.Dataset],
              identifiers: Optional[Union[tf.Tensor, tf.data.Dataset]] = None) -> "Faiss":

        if isinstance(candidates, tf.data.Dataset):
            candidates = tf.concat(list(candidates), axis=0)

        if identifiers is None:
            identifiers = tf.range(candidates.shape[0])

        if isinstance(identifiers, tf.data.Dataset):
            identifiers = tf.concat(list(identifiers), axis=0)

        if tf.rank(candidates) != 2:
            raise ValueError("`candidates` ndim should be 2. "
                             "Got `ndim` = {}".format(tf.rank(candidates)))

        if identifiers.dtype not in ("int8", "int16", "int32", "int64"):
            self._searcher = self._build_searcher(candidates, identifiers=None)
            # 初始化identifiers
            identifiers_initial_value = tf.zeros((), dtype=identifiers.dtype)
            
            self._identifiers = self.add_weight(
                name="identifiers",
                dtype=identifiers.dtype,
                shape=identifiers.shape,
                initializer=tf.keras.initializers.Constant(
                    value=identifiers_initial_value),
                trainable=False)
            self._identifiers.assign(identifiers)
        else:
            self._searcher = self._build_searcher(candidates, identifiers=identifiers)
            
        self._reset_tf_function_cache()
    
        return self

    def call(self,
             queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
             k: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        
        k = k if k is not None else self._k

        if self._searcher is None:
            raise ValueError("The `index` method must be called first to "
                            "create the retrieval index.")
        
        if self._query_model is not None:
            queries = self._query_model(queries)

        if not isinstance(queries, tf.Tensor):
            raise ValueError(f"Queries must be a tensor, got {type(queries)}.")
        
        def _search(queries, k):
            queries = tf.make_ndarray(tf.make_tensor_proto(queries))
            
            if self._normalize is True:
                faiss.normalize_L2(queries)

            self._searcher.nprobe = self._nprobe
            distances, indices = self._searcher.search(queries, int(k))
            return distances, indices
        
        distances, indices = tf.py_function(_search, [queries, k], [tf.float32, tf.int32])

        if self._identifiers is None:
            return distances, indices

        return distances, tf.gather(self._identifiers, indices)


class TestTopK(tf.test.TestCase, parameterized.TestCase):

    def test_take_long_axis(self):
        arr = tf.constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        indices = tf.constant([[0, 1], [2, 1]])
        out = _take_long_axis(arr, indices)
        expected_out = tf.constant([[0.1, 0.2], [0.6, 0.5]])
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(out, expected_out)

    def test_exclude(self):
        scores = tf.constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        identifiers = tf.constant([[0, 1, 2], [3, 4, 5]])
        exclude = tf.constant([[1, 2], [3, 5]])
        k = 1
        x, y = _exclude(scores, identifiers, exclude, k)
        expected_x = tf.constant([[0.1], [0.5]])
        expected_y = tf.constant([[0], [4]])
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose((x, y), (expected_x, expected_y))

    @parameterized.parameters(np.str, np.float32, np.float64, np.int32, np.int64)
    def test_faiss(self, identifier_dtype):
        num_candidates, num_queries = (5000, 4)

        rng = np.random.RandomState(42) # pylint: disable=no-member
        candidates = rng.normal(size=(num_candidates, 4)).astype(np.float32)
        query = rng.normal(size=(num_queries, 4)).astype(np.float32)
        candidate_names = np.arange(num_candidates).astype(identifier_dtype)

        faiss_topk = Faiss(k=10)
        faiss_topk.index(candidates, candidate_names)

        for _ in range(100):
            pre_serialization_results = faiss_topk(query[:2])

        path = os.path.join(self.get_temp_dir(), "query_model")
        faiss_topk.save(
            path,
            options=tf.saved_model.SaveOptions(namespace_whitelist=["Faiss"]))
        loaded = tf.keras.models.load_model(path)

        for _ in range(100):
            post_serialization_results = loaded(tf.constant(query[:2]))

        self.assertAllEqual(post_serialization_results, pre_serialization_results)
    
    @parameterized.parameters(np.float32, np.float64)
    def test_faiss_with_no_identifiers(self, candidate_dtype):
        """ 测试构建无唯一标识索引 """
        num_candidates = 5000

        candidates = np.random.normal(size=(num_candidates, 4)).astype(candidate_dtype)
        faiss_topk = Faiss(k=10)
        faiss_topk.index(candidates, identifiers=None)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(num_candidates, faiss_topk._searcher.ntotal)

    @parameterized.parameters(np.str, np.float32, np.float64, np.int32, np.int64)
    def test_faiss_with_dataset(self, identifier_dtype):
        num_candidates = 5000
        
        candidates = tf.data.Dataset.from_tensor_slices(
            np.random.normal(size=(num_candidates, 4)).astype(np.float32))
        identifiers = tf.data.Dataset.from_tensor_slices(
            np.arange(num_candidates).astype(identifier_dtype))
        faiss_topk = Faiss(k=10)
        faiss_topk.index(candidates.batch(100), identifiers=identifiers)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(num_candidates, faiss_topk._searcher.ntotal)

        
if __name__ == "__main__":
    tf.test.main()
