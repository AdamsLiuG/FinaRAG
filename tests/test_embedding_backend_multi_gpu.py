import unittest
import threading
import time
from unittest.mock import patch

import numpy as np

from src.embedding_backend import BGEM3SparseEmbeddingBackend, EmbeddingBackend, _parse_devices


class DummySentenceTransformer:
    def __init__(self, device: str):
        self.device = device

    def encode(self, texts, **kwargs):
        text_list = [texts] if isinstance(texts, str) else list(texts)
        device_score = float(self.device.split(":")[-1]) if ":" in self.device else 0.0
        rows = np.asarray([[float(len(text)), device_score] for text in text_list], dtype=np.float32)
        if isinstance(texts, str):
            return rows[0]
        return rows


class DummyBGEM3Model:
    def __init__(self, device: str):
        self.device = device

    def encode(self, texts, **kwargs):
        text_list = list(texts)
        return {
            "lexical_weights": [
                {"text": text, "device": self.device, "length": len(text)}
                for text in text_list
            ]
        }

    def compute_lexical_matching_score(self, query_weights_list, document_weights):
        query_size = len(query_weights_list[0].get("text", "")) or 1
        return [float(query_size + item.get("length", 0)) for item in document_weights]


class ConcurrentUnsafeBGEM3Model:
    def __init__(self):
        self._guard = threading.Lock()
        self.active_calls = 0
        self.overlap_detected = False

    def encode(self, texts, **kwargs):
        with self._guard:
            self.active_calls += 1
            if self.active_calls > 1:
                self.overlap_detected = True
        try:
            time.sleep(0.05)
            text_list = list(texts)
            return {
                "lexical_weights": [
                    {"text": text, "length": len(text)}
                    for text in text_list
                ]
            }
        finally:
            with self._guard:
                self.active_calls -= 1

    def compute_lexical_matching_score(self, query_weights_list, document_weights):
        with self._guard:
            self.active_calls += 1
            if self.active_calls > 1:
                self.overlap_detected = True
        try:
            time.sleep(0.05)
            query_size = len(query_weights_list[0].get("text", "")) or 1
            return [float(query_size + item.get("length", 0)) for item in document_weights]
        finally:
            with self._guard:
                self.active_calls -= 1


class EmbeddingBackendMultiGpuTests(unittest.TestCase):
    def test_parse_devices_accepts_comma_separated_cuda_list(self):
        self.assertEqual(_parse_devices("cuda:0, cuda:1"), ["cuda:0", "cuda:1"])
        self.assertEqual(_parse_devices("0,1"), ["cuda:0", "cuda:1"])

    def test_dense_embedding_backend_shards_texts_across_devices_and_preserves_order(self):
        with patch("src.embedding_backend._load_model", side_effect=lambda model_name, device, trust: DummySentenceTransformer(device)):
            backend = EmbeddingBackend(model_name="dummy", device="cuda:0, cuda:1", batch_size=2)

            embeddings = backend.embed_texts(["aa", "bbb", "c", "dddd"])

        self.assertEqual(backend.devices, ["cuda:0", "cuda:1"])
        self.assertEqual(
            embeddings,
            [
                [2.0, 0.0],
                [3.0, 1.0],
                [1.0, 0.0],
                [4.0, 1.0],
            ],
        )

    def test_dense_embedding_backend_uses_primary_device_for_queries(self):
        with patch("src.embedding_backend._load_model", side_effect=lambda model_name, device, trust: DummySentenceTransformer(device)):
            backend = EmbeddingBackend(model_name="dummy", device="cuda:0, cuda:1")

            embedding = backend.embed_query("hello")

        self.assertEqual(embedding.tolist(), [5.0, 0.0])

    def test_sparse_embedding_backend_shards_texts_across_devices_and_preserves_order(self):
        with patch("src.embedding_backend._load_bgem3_model", side_effect=lambda **kwargs: DummyBGEM3Model(kwargs["device"])):
            backend = BGEM3SparseEmbeddingBackend(model_name="dummy", device="cuda:0, cuda:1", batch_size=2)

            lexical_weights = backend.encode_texts(["a", "bb", "ccc", "dddd"])

        self.assertEqual(
            lexical_weights,
            [
                {"text": "a", "device": "cuda:0", "length": 1},
                {"text": "bb", "device": "cuda:1", "length": 2},
                {"text": "ccc", "device": "cuda:0", "length": 3},
                {"text": "dddd", "device": "cuda:1", "length": 4},
            ],
        )

    def test_sparse_embedding_backend_uses_primary_device_for_queries(self):
        with patch("src.embedding_backend._load_bgem3_model", side_effect=lambda **kwargs: DummyBGEM3Model(kwargs["device"])):
            backend = BGEM3SparseEmbeddingBackend(model_name="dummy", device="cuda:0, cuda:1")

            query_weights = backend.encode_query("xyz")

        self.assertEqual(query_weights, {"text": "xyz", "device": "cuda:0", "length": 3})

    def test_sparse_embedding_backend_serializes_shared_model_queries(self):
        shared_model = ConcurrentUnsafeBGEM3Model()
        barrier = threading.Barrier(2)

        with patch("src.embedding_backend._load_bgem3_model", side_effect=lambda **kwargs: shared_model):
            backend_one = BGEM3SparseEmbeddingBackend(model_name="dummy", device="cuda:0")
            backend_two = BGEM3SparseEmbeddingBackend(model_name="dummy", device="cuda:0")

            results = []

            def worker(backend, text):
                barrier.wait()
                results.append(backend.encode_query(text))

            thread_one = threading.Thread(target=worker, args=(backend_one, "alpha"))
            thread_two = threading.Thread(target=worker, args=(backend_two, "beta"))
            thread_one.start()
            thread_two.start()
            thread_one.join()
            thread_two.join()

        self.assertEqual(len(results), 2)
        self.assertFalse(shared_model.overlap_detected)


if __name__ == "__main__":
    unittest.main()
