from __future__ import annotations

import difflib
import math
import re
from collections import Counter
from typing import Any, List

try:
    import jieba
except ImportError:  # pragma: no cover - fallback depends on local env
    jieba = None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return " ".join(_normalize_text(item) for item in value)
    return " ".join(str(value).strip().lower().split())


def _is_cjk_text(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _tokenize_text(value: Any) -> List[str]:
    text = _normalize_text(value)
    if not text:
        return []
    if _is_cjk_text(text):
        if " " in text:
            return [token for token in text.split(" ") if token]
        if jieba is not None:
            return [token.strip() for token in jieba.lcut(text) if token.strip()]
        if len(text) <= 4:
            return [text]
        return [char for char in text if char.strip()]
    return [token for token in re.split(r"[\s,;|/]+", text) if token]


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _parse_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return None

    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace(",", "").replace("，", "").replace("%", "")
    normalized = normalized.replace("人民币", "").replace("元", "").replace("万元", "").replace("亿元", "")
    normalized = normalized.strip()
    if not normalized:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def exact_match_score(prediction: Any, reference: Any) -> float:
    return 1.0 if _normalize_text(prediction) == _normalize_text(reference) else 0.0


def list_f1_score(prediction: Any, reference: Any) -> float | None:
    if not isinstance(prediction, list) and not isinstance(reference, list):
        return None

    pred_counter = Counter(_normalize_text(item) for item in (prediction or []))
    ref_counter = Counter(_normalize_text(item) for item in (reference or []))
    if not pred_counter and not ref_counter:
        return 1.0
    overlap = sum((pred_counter & ref_counter).values())
    precision = _safe_divide(overlap, sum(pred_counter.values()))
    recall = _safe_divide(overlap, sum(ref_counter.values()))
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def token_f1_score(prediction: Any, reference: Any) -> float:
    pred_counter = Counter(_tokenize_text(prediction))
    ref_counter = Counter(_tokenize_text(reference))
    if not pred_counter and not ref_counter:
        return 1.0
    overlap = sum((pred_counter & ref_counter).values())
    precision = _safe_divide(overlap, sum(pred_counter.values()))
    recall = _safe_divide(overlap, sum(ref_counter.values()))
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def sequence_similarity_score(prediction: Any, reference: Any) -> float:
    pred_text = _normalize_text(prediction)
    ref_text = _normalize_text(reference)
    if not pred_text and not ref_text:
        return 1.0
    return round(difflib.SequenceMatcher(a=pred_text, b=ref_text).ratio(), 4)


def numeric_similarity_score(prediction: Any, reference: Any) -> float | None:
    pred_number = _parse_number(prediction)
    ref_number = _parse_number(reference)
    if pred_number is None or ref_number is None:
        return None
    if math.isclose(pred_number, ref_number, rel_tol=1e-9, abs_tol=1e-9):
        return 1.0
    denominator = max(abs(ref_number), 1.0)
    relative_error = abs(pred_number - ref_number) / denominator
    return round(max(0.0, 1.0 - relative_error), 4)


class EmbeddingSimilarityScorer:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        return self._model

    def score(self, prediction: Any, reference: Any) -> float | None:
        pred_text = _normalize_text(prediction)
        ref_text = _normalize_text(reference)
        if not pred_text or not ref_text:
            return None
        model = self._ensure_model()
        embeddings = model.encode([pred_text, ref_text], normalize_embeddings=True)
        similarity = float(embeddings[0] @ embeddings[1])
        return round(similarity, 4)


def score_semantic_similarity(
    prediction: Any,
    reference: Any,
    *,
    kind: str | None = None,
    embedding_scorer: EmbeddingSimilarityScorer | None = None,
) -> dict:
    exact_score = exact_match_score(prediction, reference)
    list_score = list_f1_score(prediction, reference)
    numeric_score = numeric_similarity_score(prediction, reference)
    token_score = token_f1_score(prediction, reference)
    sequence_score = sequence_similarity_score(prediction, reference)

    if isinstance(reference, bool) or kind == "boolean":
        semantic_score = exact_score
        backend = "exact"
    elif list_score is not None:
        semantic_score = max(exact_score, list_score)
        backend = "list_f1"
    elif numeric_score is not None and kind in {"number", "ratio", "comparative"}:
        semantic_score = max(exact_score, numeric_score)
        backend = "numeric"
    else:
        semantic_score = max(exact_score, token_score, sequence_score)
        backend = "token_sequence"

    embedding_score = None
    if embedding_scorer is not None and kind not in {"boolean", "number", "ratio"}:
        embedding_score = embedding_scorer.score(prediction, reference)
        if embedding_score is not None and embedding_score > semantic_score:
            semantic_score = embedding_score
            backend = "embedding"

    return {
        "semantic_score": round(semantic_score, 4),
        "backend": backend,
        "exact_match": exact_score,
        "list_f1": list_score,
        "token_f1": token_score,
        "sequence_similarity": sequence_score,
        "numeric_similarity": numeric_score,
        "embedding_similarity": embedding_score,
    }
