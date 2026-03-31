import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Iterable, List

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:  # pragma: no cover - optional dependency in some envs
    BGEM3FlagModel = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_devices(value: str | None, default: str = "cpu") -> List[str]:
    if value is None:
        return [default]

    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        return [default]

    devices: List[str] = []
    for part in parts:
        normalized = part.lower()
        if part.isdigit():
            devices.append(f"cuda:{part}")
        elif normalized in {"cuda", "cpu", "mps"}:
            devices.append(normalized)
        elif normalized.startswith(("cuda:", "cpu", "mps", "npu:", "musa:")):
            devices.append(part)
        else:
            devices.append(f"cuda:{part}")
    return devices


def _split_indexed_items(indexed_items: List[tuple], num_buckets: int) -> List[List[tuple]]:
    if num_buckets <= 1:
        return [indexed_items]
    return [indexed_items[index::num_buckets] for index in range(num_buckets)]


# ---------------------------------------------------------------------------
# Thread-safe model loading
#
# Root cause of the meta-tensor error
# ------------------------------------
# SentenceTransformer.__init__ always calls self.to(device) at the end.
# When transformers loads weights with low_cpu_mem_usage=True (the default),
# it uses accelerate's init_empty_weights context manager, placing every
# parameter on the virtual 'meta' device.  The subsequent self.to(device)
# then raises:
#
#     NotImplementedError: Cannot copy out of meta tensor; no data!
#
# Concurrency makes this worse
# ----------------------------
# lru_cache does NOT block concurrent callers — before the first result is
# cached, N parallel threads all enter the loading function simultaneously.
# accelerate's init_empty_weights uses global/thread-local state that can
# be corrupted when multiple loads race each other, causing the meta-tensor
# error even for small models (e.g. bge-small-en-v1.5) that normally load
# fine in a single-threaded context.
#
# Fix
# ---
# 1. A module-level threading.Lock serialises all concurrent load requests.
#    The first thread acquires the lock and loads the model; every subsequent
#    thread blocks until the lock is released, then hits the lru_cache and
#    returns instantly.
# 2. We also force device="cpu" when constructing SentenceTransformer so
#    that the internal self.to() call never tries to move meta tensors to a
#    GPU.  After construction (all weights in real CPU memory) we move the
#    model to the desired device ourselves.
# 3. low_cpu_mem_usage=False prevents transformers from using
#    init_empty_weights in the first place.
# ---------------------------------------------------------------------------

_model_load_lock = threading.Lock()
_bgem3_model_load_lock = threading.Lock()


@lru_cache(maxsize=4)
def _load_model_cached(model_name: str, device: str, trust_remote_code: bool) -> SentenceTransformer:
    """Actually load the model (called at most once per unique set of args)."""

    def _safe_load(name: str, trust: bool) -> SentenceTransformer:
        # Force CPU during construction so self.to(device) inside __init__
        # never tries to move meta-device tensors to a GPU.
        # low_cpu_mem_usage=False prevents accelerate from using
        # init_empty_weights (which is what creates the meta tensors).
        model = SentenceTransformer(
            name,
            device="cpu",
            trust_remote_code=trust,
            model_kwargs={"low_cpu_mem_usage": False},
        )
        # Move to the desired device *after* all weights are in real CPU memory.
        if device:
            model = model.to(device)
        return model

    try:
        return _safe_load(model_name, trust_remote_code)
    except Exception as err:  # noqa: BLE001
        # Re-raise anything that isn't a meta-tensor error.
        if "meta tensor" not in str(err).lower() and "meta" not in str(err).lower():
            raise

        fallback_model = os.getenv("EMBEDDING_FALLBACK_MODEL_NAME", "BAAI/bge-small-en-v1.5")
        if fallback_model == model_name:
            raise

        print(
            f"Embedding model '{model_name}' failed to load with a meta-tensor error. "
            f"Falling back to '{fallback_model}'."
        )
        return _safe_load(fallback_model, trust=False)


def _load_model(model_name: str, device: str, trust_remote_code: bool) -> SentenceTransformer:
    """Thread-safe wrapper: serialises concurrent loads, then delegates to the
    lru_cache'd function.  After the first load, all threads hit the cache
    and return immediately without acquiring the lock."""
    with _model_load_lock:
        return _load_model_cached(model_name, device, trust_remote_code)


@lru_cache(maxsize=2)
def _load_bgem3_model_cached(
    model_name: str,
    device: str,
    use_fp16: bool,
    batch_size: int,
    query_max_length: int,
    passage_max_length: int,
) -> "BGEM3FlagModel":
    if BGEM3FlagModel is None:
        raise ImportError(
            "FlagEmbedding is required for bge-m3 sparse lexical weights support. "
            "Install it with `pip install FlagEmbedding`."
        )

    return BGEM3FlagModel(
        model_name,
        normalize_embeddings=True,
        use_fp16=use_fp16,
        devices=device or None,
        batch_size=batch_size,
        query_max_length=query_max_length,
        passage_max_length=passage_max_length,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )


def _load_bgem3_model(
    model_name: str,
    device: str,
    use_fp16: bool,
    batch_size: int,
    query_max_length: int,
    passage_max_length: int,
) -> "BGEM3FlagModel":
    with _bgem3_model_load_lock:
        return _load_bgem3_model_cached(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
        )


class EmbeddingBackend:
    def __init__(self, model_name: str = None, device: str = None, batch_size: int = None):
        load_dotenv()
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        raw_device = device if device is not None else os.getenv("EMBEDDING_DEVICE", "")
        self.devices = _parse_devices(raw_device, default="cpu")
        self.device = self.devices[0]
        self.batch_size = batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        self.trust_remote_code = _env_flag("EMBEDDING_TRUST_REMOTE_CODE", default=False)
        self.models = [
            _load_model(self.model_name, target_device, self.trust_remote_code)
            for target_device in self.devices
        ]
        self.model = self.models[0]

    def _encode_batch(self, model: SentenceTransformer, texts: List[str]) -> np.ndarray:
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings_array = np.asarray(embeddings, dtype=np.float32)
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        return embeddings_array

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        text_list = [text for text in texts if text and text.strip()]
        if not text_list:
            return []
        if len(self.models) == 1:
            return self._encode_batch(self.model, text_list).tolist()

        indexed_texts = list(enumerate(text_list))
        chunks = _split_indexed_items(indexed_texts, len(self.models))
        ordered_embeddings: List[np.ndarray | None] = [None] * len(text_list)

        def encode_chunk(model: SentenceTransformer, chunk: List[tuple]) -> List[tuple]:
            if not chunk:
                return []
            indices = [index for index, _ in chunk]
            chunk_texts = [text for _, text in chunk]
            chunk_embeddings = self._encode_batch(model, chunk_texts)
            return list(zip(indices, chunk_embeddings))

        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = [
                executor.submit(encode_chunk, model, chunk)
                for model, chunk in zip(self.models, chunks)
            ]
            for future in futures:
                for index, embedding in future.result():
                    ordered_embeddings[index] = embedding

        if any(embedding is None for embedding in ordered_embeddings):
            raise RuntimeError("Dense embedding sharding produced incomplete results.")
        return [np.asarray(embedding, dtype=np.float32).tolist() for embedding in ordered_embeddings]

    def embed_query(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty.")

        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embedding, dtype=np.float32)


class BGEM3SparseEmbeddingBackend:
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None,
        query_max_length: int = None,
        passage_max_length: int = None,
    ):
        load_dotenv()
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_SPARSE_MODEL_NAME")
            or os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        )
        raw_device = device or os.getenv("EMBEDDING_SPARSE_DEVICE") or os.getenv("EMBEDDING_DEVICE", "cpu")
        self.devices = _parse_devices(raw_device, default="cpu")
        self.device = self.devices[0]
        self.batch_size = batch_size or int(os.getenv("EMBEDDING_SPARSE_BATCH_SIZE", "32"))
        self.query_max_length = query_max_length or int(os.getenv("EMBEDDING_SPARSE_QUERY_MAX_LENGTH", "256"))
        self.passage_max_length = passage_max_length or int(os.getenv("EMBEDDING_SPARSE_PASSAGE_MAX_LENGTH", "512"))
        use_fp16_default = any(device_name.startswith("cuda") for device_name in self.devices)
        self.use_fp16 = _env_flag("EMBEDDING_SPARSE_USE_FP16", default=use_fp16_default)
        self.models = [
            _load_bgem3_model(
                model_name=self.model_name,
                device=target_device,
                use_fp16=self.use_fp16,
                batch_size=self.batch_size,
                query_max_length=self.query_max_length,
                passage_max_length=self.passage_max_length,
            )
            for target_device in self.devices
        ]
        self.model = self.models[0]

    def _encode_sparse_batch(self, model: "BGEM3FlagModel", texts: List[str], max_length: int) -> List[dict]:
        outputs = model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return list(outputs["lexical_weights"])

    def encode_texts(self, texts: Iterable[str]) -> List[dict]:
        text_list = [text for text in texts if text and text.strip()]
        if not text_list:
            return []
        if len(self.models) == 1:
            return self._encode_sparse_batch(self.model, text_list, self.passage_max_length)

        indexed_texts = list(enumerate(text_list))
        chunks = _split_indexed_items(indexed_texts, len(self.models))
        ordered_weights: List[dict | None] = [None] * len(text_list)

        def encode_chunk(model: "BGEM3FlagModel", chunk: List[tuple]) -> List[tuple]:
            if not chunk:
                return []
            indices = [index for index, _ in chunk]
            chunk_texts = [text for _, text in chunk]
            chunk_weights = self._encode_sparse_batch(model, chunk_texts, self.passage_max_length)
            return list(zip(indices, chunk_weights))

        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = [
                executor.submit(encode_chunk, model, chunk)
                for model, chunk in zip(self.models, chunks)
            ]
            for future in futures:
                for index, lexical_weight in future.result():
                    ordered_weights[index] = lexical_weight

        if any(weight is None for weight in ordered_weights):
            raise RuntimeError("Sparse embedding sharding produced incomplete results.")
        return list(ordered_weights)

    def encode_query(self, text: str) -> dict:
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty.")

        outputs = self.model.encode(
            [text],
            batch_size=1,
            max_length=self.query_max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return outputs["lexical_weights"][0]

    def score_query_against_documents(self, query_weights: dict, document_weights: List[dict]) -> List[float]:
        if not document_weights:
            return []

        scores = self.model.compute_lexical_matching_score([query_weights], document_weights)
        scores_array = np.asarray(scores, dtype=np.float32).reshape(-1)
        return scores_array.tolist()
