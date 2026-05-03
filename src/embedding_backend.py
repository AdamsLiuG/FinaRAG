import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Iterable, List

import numpy as np
import torch
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


def _env_flag_any(*names: str, default: bool = False) -> bool:
    for name in names:
        if not name:
            continue
        value = os.getenv(name)
        if value is None:
            continue
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


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


def _env_positive_int(*names: str) -> int | None:
    for name in names:
        raw_value = os.getenv(name)
        if raw_value is None or str(raw_value).strip() == "":
            continue
        try:
            value = int(str(raw_value).strip())
        except ValueError:
            continue
        if value > 0:
            return value
    return None


def _limit_devices(devices: List[str], *env_names: str) -> List[str]:
    max_devices = _env_positive_int(*env_names)
    if max_devices is None:
        return devices
    return devices[:max(1, max_devices)]


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
_bgem3_inference_locks_guard = threading.Lock()
_bgem3_inference_locks: dict[int, threading.Lock] = {}


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


@lru_cache(maxsize=8)
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


def _get_bgem3_inference_lock(model: "BGEM3FlagModel") -> threading.Lock:
    model_id = id(model)
    with _bgem3_inference_locks_guard:
        lock = _bgem3_inference_locks.get(model_id)
        if lock is None:
            lock = threading.Lock()
            _bgem3_inference_locks[model_id] = lock
        return lock


def _resolve_bgem3_device(model: "BGEM3FlagModel", device: str | None = None) -> str:
    if device:
        return device
    target_devices = list(getattr(model, "target_devices", []) or [])
    if target_devices:
        return str(target_devices[0])
    return "cpu"


def _disable_bgem3_unused_pooler(inference_model) -> None:
    base_model = getattr(inference_model, "model", None)
    if base_model is None:
        return

    if getattr(base_model, "pooler", None) is not None:
        base_model.pooler = None

    config = getattr(base_model, "config", None)
    if config is not None and hasattr(config, "add_pooling_layer"):
        config.add_pooling_layer = False


def _bgem3_should_use_fp32_sparse_head(model: "BGEM3FlagModel", device: str) -> bool:
    return str(device).startswith("cuda") and getattr(model, "use_fp16", False) and _env_flag_any(
        "EMBEDDING_SPARSE_FP32_HEAD",
        "BGEM3_FP32_HEADS",
        default=True,
    )


def _bgem3_should_use_fp32_colbert_head(model: "BGEM3FlagModel", device: str) -> bool:
    return str(device).startswith("cuda") and getattr(model, "use_fp16", False) and _env_flag_any(
        "COLBERT_FP32_HEAD",
        "BGEM3_FP32_HEADS",
        default=True,
    )


def _configure_bgem3_head_precision(
    inference_model,
    *,
    device: str,
    default_head_dtype: torch.dtype,
    use_fp32_sparse_head: bool,
    use_fp32_colbert_head: bool,
) -> None:
    sparse_linear = getattr(inference_model, "sparse_linear", None)
    colbert_linear = getattr(inference_model, "colbert_linear", None)

    if sparse_linear is not None:
        sparse_target_dtype = torch.float32 if use_fp32_sparse_head else default_head_dtype
        sparse_linear.to(device=device, dtype=sparse_target_dtype)
    if colbert_linear is not None:
        colbert_target_dtype = torch.float32 if use_fp32_colbert_head else default_head_dtype
        colbert_linear.to(device=device, dtype=colbert_target_dtype)

    if hasattr(inference_model, "_sparse_embedding") and not hasattr(inference_model, "_finarag_original_sparse_embedding"):
        inference_model._finarag_original_sparse_embedding = inference_model._sparse_embedding
    if hasattr(inference_model, "_colbert_embedding") and not hasattr(inference_model, "_finarag_original_colbert_embedding"):
        inference_model._finarag_original_colbert_embedding = inference_model._colbert_embedding

    if use_fp32_sparse_head and hasattr(inference_model, "_finarag_original_sparse_embedding"):
        def _sparse_embedding_with_fp32_head(hidden_state, input_ids, return_embedding: bool = True):
            return inference_model._finarag_original_sparse_embedding(
                hidden_state.float(),
                input_ids,
                return_embedding=return_embedding,
            )

        inference_model._sparse_embedding = _sparse_embedding_with_fp32_head
    elif hasattr(inference_model, "_finarag_original_sparse_embedding"):
        inference_model._sparse_embedding = inference_model._finarag_original_sparse_embedding

    if use_fp32_colbert_head and hasattr(inference_model, "_finarag_original_colbert_embedding"):
        def _colbert_embedding_with_fp32_head(last_hidden_state, mask):
            return inference_model._finarag_original_colbert_embedding(
                last_hidden_state.float(),
                mask,
            )

        inference_model._colbert_embedding = _colbert_embedding_with_fp32_head
    elif hasattr(inference_model, "_finarag_original_colbert_embedding"):
        inference_model._colbert_embedding = inference_model._finarag_original_colbert_embedding


def _prepare_bgem3_model_for_inference(
    model: "BGEM3FlagModel",
    device: str,
    *,
    return_sparse: bool = False,
    return_colbert_vecs: bool = False,
):
    inference_model = model.model
    _disable_bgem3_unused_pooler(inference_model)
    target_dtype = torch.float16 if getattr(model, "use_fp16", False) and str(device).startswith("cuda") else torch.float32

    try:
        reference_param = next(inference_model.parameters())
        current_device = str(reference_param.device)
        current_dtype = reference_param.dtype
    except StopIteration:  # pragma: no cover - model without parameters is not expected
        current_device = None
        current_dtype = None

    if current_dtype != target_dtype:
        if target_dtype == torch.float16:
            inference_model.half()
        else:
            inference_model.float()

    if current_device != str(device):
        inference_model.to(device)

    _configure_bgem3_head_precision(
        inference_model,
        device=str(device),
        default_head_dtype=target_dtype,
        use_fp32_sparse_head=return_sparse and _bgem3_should_use_fp32_sparse_head(model, device),
        use_fp32_colbert_head=return_colbert_vecs and _bgem3_should_use_fp32_colbert_head(model, device),
    )

    inference_model.eval()
    return inference_model


def _collect_bgem3_special_token_ids(tokenizer) -> set[int]:
    special_token_ids: set[int] = set()
    for token_name in ("cls_token", "eos_token", "pad_token", "unk_token"):
        token = tokenizer.special_tokens_map.get(token_name)
        if token is None:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            continue
        special_token_ids.add(int(token_id))
    return special_token_ids


def _process_bgem3_lexical_weights(
    token_weights: np.ndarray,
    input_ids: list[int],
    special_token_ids: set[int],
) -> dict[str, float]:
    lexical_weights: dict[str, float] = {}
    for weight, token_id in zip(token_weights, input_ids):
        normalized_token_id = int(token_id)
        normalized_weight = float(weight)
        if normalized_token_id in special_token_ids or normalized_weight <= 0:
            continue
        key = str(normalized_token_id)
        previous_weight = lexical_weights.get(key, 0.0)
        if normalized_weight > previous_weight:
            lexical_weights[key] = normalized_weight
    return lexical_weights


def _process_bgem3_colbert_vecs(colbert_vecs: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    tokens_num = int(np.asarray(attention_mask).sum())
    usable_tokens = max(tokens_num - 1, 0)
    return np.asarray(colbert_vecs[:usable_tokens], dtype=np.float32)


@torch.no_grad()
def _bgem3_encode_texts(
    model: "BGEM3FlagModel",
    texts: List[str] | str,
    *,
    batch_size: int,
    max_length: int,
    return_dense: bool = False,
    return_sparse: bool = False,
    return_colbert_vecs: bool = False,
    device: str | None = None,
) -> dict:
    if not (return_dense or return_sparse or return_colbert_vecs):
        raise ValueError("At least one BGE-M3 output mode must be enabled.")

    input_was_string = isinstance(texts, str)
    text_list = [texts] if input_was_string else list(texts)
    if not text_list:
        return {
            "dense_vecs": np.asarray([], dtype=np.float32) if return_dense else None,
            "lexical_weights": [] if return_sparse else None,
            "colbert_vecs": [] if return_colbert_vecs else None,
        }

    current_batch_size = max(1, int(batch_size))
    resolved_device = _resolve_bgem3_device(model, device)
    tokenizer = model.tokenizer
    special_token_ids = _collect_bgem3_special_token_ids(tokenizer)

    dense_outputs: list[np.ndarray] = []
    lexical_outputs: list[dict[str, float]] = []
    colbert_outputs: list[np.ndarray] = []

    with _get_bgem3_inference_lock(model):
        inference_model = _prepare_bgem3_model_for_inference(
            model,
            resolved_device,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs,
        )

        start_index = 0
        while start_index < len(text_list):
            effective_batch_size = min(current_batch_size, len(text_list) - start_index)
            batch_texts = text_list[start_index:start_index + effective_batch_size]
            try:
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(resolved_device)
                outputs = inference_model(
                    inputs,
                    return_dense=return_dense,
                    return_sparse=return_sparse,
                    return_colbert_vecs=return_colbert_vecs,
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                if effective_batch_size == 1:
                    raise
                if str(resolved_device).startswith("cuda"):
                    torch.cuda.empty_cache()
                current_batch_size = max(1, effective_batch_size * 3 // 4)
                continue

            if return_dense:
                dense_outputs.append(outputs["dense_vecs"].detach().float().cpu().numpy())

            if return_sparse:
                token_weights = outputs["sparse_vecs"].detach().float().cpu().numpy().squeeze(-1)
                input_ids = inputs["input_ids"].detach().cpu().numpy().tolist()
                lexical_outputs.extend(
                    _process_bgem3_lexical_weights(weights, ids, special_token_ids)
                    for weights, ids in zip(token_weights, input_ids)
                )

            if return_colbert_vecs:
                colbert_vecs = outputs["colbert_vecs"].detach().float().cpu().numpy()
                attention_mask = inputs["attention_mask"].detach().cpu().numpy()
                colbert_outputs.extend(
                    _process_bgem3_colbert_vecs(vecs, mask)
                    for vecs, mask in zip(colbert_vecs, attention_mask)
                )

            start_index += effective_batch_size

    dense_result = None
    if return_dense:
        if dense_outputs:
            dense_result = np.concatenate(dense_outputs, axis=0)
        else:  # pragma: no cover - guarded by non-empty text_list
            dense_result = np.asarray([], dtype=np.float32)
        if input_was_string:
            dense_result = dense_result[0]

    lexical_result = lexical_outputs if return_sparse else None
    if return_sparse and input_was_string:
        lexical_result = lexical_outputs[0] if lexical_outputs else {}

    colbert_result = colbert_outputs if return_colbert_vecs else None
    if return_colbert_vecs and input_was_string:
        if colbert_outputs:
            colbert_result = colbert_outputs[0]
        else:
            colbert_result = np.asarray([], dtype=np.float32)

    return {
        "dense_vecs": dense_result,
        "lexical_weights": lexical_result,
        "colbert_vecs": colbert_result,
    }


class EmbeddingBackend:
    def __init__(self, model_name: str = None, device: str = None, batch_size: int = None):
        load_dotenv()
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        raw_device = device if device is not None else os.getenv("EMBEDDING_DEVICE", "")
        self.devices = _limit_devices(_parse_devices(raw_device, default="cpu"), "EMBEDDING_MAX_DEVICES")
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
        self.devices = _limit_devices(
            _parse_devices(raw_device, default="cpu"),
            "EMBEDDING_SPARSE_MAX_DEVICES",
            "EMBEDDING_MAX_DEVICES",
        )
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

    @staticmethod
    def _normalize_sparse_text(text: object) -> str:
        return " ".join(str(text or "").split())

    def _encode_sparse_batch(self, model: "BGEM3FlagModel", texts: List[str], max_length: int) -> List[dict]:
        outputs = _bgem3_encode_texts(
            model,
            texts,
            batch_size=self.batch_size,
            max_length=max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return list(outputs["lexical_weights"] or [])

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
        normalized_text = self._normalize_sparse_text(text)
        if not normalized_text:
            raise ValueError("Query text cannot be empty.")

        outputs = _bgem3_encode_texts(
            self.model,
            [normalized_text],
            batch_size=1,
            max_length=self.query_max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        lexical_weights = list(outputs.get("lexical_weights") or [])
        if not lexical_weights:
            return {}
        return lexical_weights[0]

    def score_query_against_documents(self, query_weights: dict, document_weights: List[dict]) -> List[float]:
        if not document_weights:
            return []
        if not query_weights:
            return [0.0] * len(document_weights)

        with _get_bgem3_inference_lock(self.model):
            scores = self.model.compute_lexical_matching_score([query_weights], document_weights)
        scores_array = np.asarray(scores, dtype=np.float32).reshape(-1)
        return scores_array.tolist()
