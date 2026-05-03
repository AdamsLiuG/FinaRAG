import threading
import time
import unittest
import os
from unittest.mock import patch

import numpy as np
import torch

from src.embedding_backend import (
    BGEM3SparseEmbeddingBackend,
    EmbeddingBackend,
    _disable_bgem3_unused_pooler,
    _parse_devices,
    _prepare_bgem3_model_for_inference,
)


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


class _FakeBatchEncoding(dict):
    def to(self, device):
        self["__device__"] = device
        return self


class _FakeTokenizer:
    special_tokens_map = {
        "cls_token": "[CLS]",
        "eos_token": "[SEP]",
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
    }

    _token_ids = {
        "[PAD]": 0,
        "[UNK]": 100,
        "[CLS]": 101,
        "[SEP]": 102,
    }

    def convert_tokens_to_ids(self, token):
        return self._token_ids[token]

    def __call__(self, texts, padding=True, truncation=True, max_length=None, return_tensors="pt"):
        text_list = [texts] if isinstance(texts, str) else list(texts)
        rows = []
        masks = []
        max_tokens = 0

        for text in text_list:
            text_length = len(str(text))
            tokens = [101]
            if text_length > 0:
                tokens.extend([1000 + text_length, 2000 + text_length])
            tokens.append(102)
            if max_length is not None:
                tokens = tokens[:max_length]
            rows.append(tokens)
            max_tokens = max(max_tokens, len(tokens))

        for index, tokens in enumerate(rows):
            mask = [1] * len(tokens)
            if padding:
                pad_length = max_tokens - len(tokens)
                if pad_length > 0:
                    rows[index] = tokens + [0] * pad_length
                    mask = mask + [0] * pad_length
            masks.append(mask)

        if return_tensors == "pt":
            return _FakeBatchEncoding(
                {
                    "input_ids": torch.tensor(rows, dtype=torch.long),
                    "attention_mask": torch.tensor(masks, dtype=torch.long),
                }
            )
        return {"input_ids": rows, "attention_mask": masks}


class _FakeParameter:
    def __init__(self):
        self.device = "cpu"
        self.dtype = torch.float32


class _FakeInferenceModel:
    def __init__(self, device: str, *, max_supported_batch_size: int | None = None):
        self.parameter = _FakeParameter()
        self.device = device
        self.max_supported_batch_size = max_supported_batch_size
        self.forward_calls = []
        self.model = type(
            "_FakeBaseModel",
            (),
            {
                "pooler": object(),
                "config": type("_FakeConfig", (), {"add_pooling_layer": True})(),
            },
        )()

    def parameters(self):
        return iter([self.parameter])

    def half(self):
        self.parameter.dtype = torch.float16
        return self

    def float(self):
        self.parameter.dtype = torch.float32
        return self

    def to(self, device):
        self.parameter.device = device
        return self

    def eval(self):
        return self

    def __call__(self, inputs, return_dense=False, return_sparse=False, return_colbert_vecs=False):
        batch_size = int(inputs["input_ids"].shape[0])
        self.forward_calls.append(
            {
                "batch_size": batch_size,
                "device": self.parameter.device,
                "dtype": self.parameter.dtype,
            }
        )
        if self.max_supported_batch_size is not None and batch_size > self.max_supported_batch_size:
            raise RuntimeError("synthetic batch overflow")

        input_ids = inputs["input_ids"].float()
        attention_mask = inputs["attention_mask"].float()
        device_offset = 10.0 * float(str(self.device).split(":")[-1]) if ":" in str(self.device) else 0.0

        outputs = {}
        if return_sparse:
            sparse_vecs = (input_ids + device_offset).unsqueeze(-1) * attention_mask.unsqueeze(-1)
            outputs["sparse_vecs"] = sparse_vecs
        if return_colbert_vecs:
            colbert_tokens = input_ids[:, 1:]
            outputs["colbert_vecs"] = torch.stack([colbert_tokens, colbert_tokens / 100.0], dim=-1)
        if return_dense:
            outputs["dense_vecs"] = input_ids[:, :2]
        return outputs


class _FakeHeadModule:
    def __init__(self):
        self.device = "cpu"
        self.dtype = torch.float32
        self.last_input_dtype = None

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = str(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(self, tensor):
        self.last_input_dtype = tensor.dtype
        return tensor.to(dtype=self.dtype)


class _PrecisionAwareInferenceModel(_FakeInferenceModel):
    def __init__(self, device: str):
        super().__init__(device)
        self.sparse_linear = _FakeHeadModule()
        self.colbert_linear = _FakeHeadModule()

    def half(self):
        super().half()
        self.sparse_linear.to(dtype=torch.float16)
        self.colbert_linear.to(dtype=torch.float16)
        return self

    def float(self):
        super().float()
        self.sparse_linear.to(dtype=torch.float32)
        self.colbert_linear.to(dtype=torch.float32)
        return self

    def to(self, device):
        super().to(device)
        self.sparse_linear.to(device=device)
        self.colbert_linear.to(device=device)
        return self

    def _sparse_embedding(self, hidden_state, input_ids, return_embedding: bool = True):
        return self.sparse_linear(hidden_state)

    def _colbert_embedding(self, last_hidden_state, mask):
        return self.colbert_linear(last_hidden_state[:, 1:])


class _FakeBGEM3Model:
    def __init__(self, device: str, *, max_supported_batch_size: int | None = None):
        self.device = device
        self.target_devices = [device]
        self.use_fp16 = str(device).startswith("cuda")
        self.tokenizer = _FakeTokenizer()
        self.model = _FakeInferenceModel(device, max_supported_batch_size=max_supported_batch_size)

    def encode(self, *args, **kwargs):  # pragma: no cover - any call means the fix regressed
        raise AssertionError("BGEM3SparseEmbeddingBackend should not call FlagEmbedding.encode anymore")

    def compute_lexical_matching_score(self, query_weights_list, document_weights):
        query_weights = query_weights_list[0]
        return [
            float(
                sum(float(query_weights.get(token_id, 0.0)) * float(document.get(token_id, 0.0)) for token_id in query_weights)
            )
            for document in document_weights
        ]


class _PrecisionAwareBGEM3Model(_FakeBGEM3Model):
    def __init__(self, device: str):
        super().__init__(device)
        self.use_fp16 = True
        self.model = _PrecisionAwareInferenceModel(device)


class _ConcurrentUnsafeInferenceModel(_FakeInferenceModel):
    def __init__(self, parent):
        super().__init__("cpu")
        self.parent = parent

    def __call__(self, inputs, return_dense=False, return_sparse=False, return_colbert_vecs=False):
        with self.parent._guard:
            self.parent.active_calls += 1
            if self.parent.active_calls > 1:
                self.parent.overlap_detected = True
        try:
            time.sleep(0.05)
            return super().__call__(
                inputs,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=return_colbert_vecs,
            )
        finally:
            with self.parent._guard:
                self.parent.active_calls -= 1


class ConcurrentUnsafeBGEM3Model(_FakeBGEM3Model):
    def __init__(self):
        super().__init__("cpu")
        self._guard = threading.Lock()
        self.active_calls = 0
        self.overlap_detected = False
        self.model = _ConcurrentUnsafeInferenceModel(self)

    def compute_lexical_matching_score(self, query_weights_list, document_weights):
        with self._guard:
            self.active_calls += 1
            if self.active_calls > 1:
                self.overlap_detected = True
        try:
            time.sleep(0.05)
            return super().compute_lexical_matching_score(query_weights_list, document_weights)
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
        created_models = {}

        def fake_loader(**kwargs):
            device = kwargs["device"]
            model = _FakeBGEM3Model(device)
            created_models[device] = model
            return model

        with patch("src.embedding_backend._load_bgem3_model", side_effect=fake_loader):
            backend = BGEM3SparseEmbeddingBackend(model_name="dummy", device="cuda:0, cuda:1", batch_size=2)

            lexical_weights = backend.encode_texts(["a", "bb", "ccc", "dddd"])

        self.assertEqual(
            lexical_weights,
            [
                {"1001": 1001.0, "2001": 2001.0},
                {"1002": 1012.0, "2002": 2012.0},
                {"1003": 1003.0, "2003": 2003.0},
                {"1004": 1014.0, "2004": 2014.0},
            ],
        )
        self.assertEqual([call["batch_size"] for call in created_models["cuda:0"].model.forward_calls], [2])
        self.assertEqual([call["batch_size"] for call in created_models["cuda:1"].model.forward_calls], [2])
        self.assertTrue(all(call["dtype"] == torch.float16 for call in created_models["cuda:0"].model.forward_calls))

    def test_sparse_embedding_backend_uses_primary_device_for_queries(self):
        with patch("src.embedding_backend._load_bgem3_model", side_effect=lambda **kwargs: _FakeBGEM3Model(kwargs["device"])):
            backend = BGEM3SparseEmbeddingBackend(model_name="dummy", device="cuda:0, cuda:1")

            query_weights = backend.encode_query("xyz")

        self.assertEqual(query_weights, {"1003": 1003.0, "2003": 2003.0})

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

    def test_sparse_embedding_backend_reduces_batch_size_without_hitting_empty_pad_path(self):
        fragile_model = _FakeBGEM3Model("cpu", max_supported_batch_size=1)

        with patch("src.embedding_backend._load_bgem3_model", side_effect=lambda **kwargs: fragile_model):
            backend = BGEM3SparseEmbeddingBackend(model_name="dummy", device="cpu", batch_size=3)
            lexical_weights = backend.encode_texts(["a", "bb", "ccc"])

        self.assertEqual(
            lexical_weights,
            [
                {"1001": 1001.0, "2001": 2001.0},
                {"1002": 1002.0, "2002": 2002.0},
                {"1003": 1003.0, "2003": 2003.0},
            ],
        )
        self.assertEqual([call["batch_size"] for call in fragile_model.model.forward_calls], [3, 2, 1, 1, 1])

    def test_disable_bgem3_unused_pooler_turns_off_base_pooler(self):
        inference_model = _FakeInferenceModel("cpu")

        _disable_bgem3_unused_pooler(inference_model)

        self.assertIsNone(inference_model.model.pooler)
        self.assertFalse(inference_model.model.config.add_pooling_layer)

    def test_prepare_bgem3_model_keeps_base_fp16_but_forces_sparse_and_colbert_heads_fp32(self):
        fake_model = _PrecisionAwareBGEM3Model("cuda:0")

        with patch.dict(os.environ, {"BGEM3_FP32_HEADS": "true"}, clear=False):
            inference_model = _prepare_bgem3_model_for_inference(
                fake_model,
                "cuda:0",
                return_sparse=True,
                return_colbert_vecs=True,
            )

        self.assertEqual(next(inference_model.parameters()).dtype, torch.float16)
        self.assertEqual(inference_model.sparse_linear.dtype, torch.float32)
        self.assertEqual(inference_model.colbert_linear.dtype, torch.float32)

        hidden_state = torch.ones((1, 3, 2), dtype=torch.float16)
        attention_mask = torch.ones((1, 3), dtype=torch.long)
        input_ids = torch.ones((1, 3), dtype=torch.long)
        inference_model._sparse_embedding(hidden_state, input_ids)
        inference_model._colbert_embedding(hidden_state, attention_mask)

        self.assertEqual(inference_model.sparse_linear.last_input_dtype, torch.float32)
        self.assertEqual(inference_model.colbert_linear.last_input_dtype, torch.float32)

    def test_prepare_bgem3_model_can_disable_fp32_head_override(self):
        fake_model = _PrecisionAwareBGEM3Model("cuda:0")

        with patch.dict(os.environ, {"BGEM3_FP32_HEADS": "false"}, clear=False):
            inference_model = _prepare_bgem3_model_for_inference(
                fake_model,
                "cuda:0",
                return_sparse=True,
                return_colbert_vecs=True,
            )

        self.assertEqual(next(inference_model.parameters()).dtype, torch.float16)
        self.assertEqual(inference_model.sparse_linear.dtype, torch.float16)
        self.assertEqual(inference_model.colbert_linear.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
