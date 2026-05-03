from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv

try:
    from ragas.llms.base import InstructorLLM, InstructorTypeVar
except Exception:
    InstructorLLM = object  # type: ignore[assignment]
    InstructorTypeVar = Any  # type: ignore[misc,assignment]


DEFAULT_RAGAS_LLM_MODEL = "Qwen3.5-35B-A3B-AWQ-4bit"
DEFAULT_RAGAS_EMBEDDING_MODEL = "BAAI/bge-m3"

RAGAS_METRIC_WEIGHTS = {
    "answer_correctness": 0.30,
    "faithfulness": 0.20,
    "answer_relevancy": 0.10,
    "context_recall": 0.25,
    "context_precision": 0.15,
}


def _env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value not in {None, ""}:
            return value
    return default


def _env_bool(*names: str, default: bool = True) -> bool:
    value = _env_value(*names)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _format_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {message}"


def _is_official_openai_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    normalized = base_url.strip().lower()
    return "api.openai.com" in normalized


def _normalize_embedding_device(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = device.strip()
    if not normalized:
        return None
    if "," in normalized:
        return normalized.split(",", 1)[0].strip()
    return normalized


def _round_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            return None
        return round(numeric_value, 4)
    extracted = None
    if isinstance(value, dict):
        extracted = value.get("value", value.get("score"))
    else:
        extracted = getattr(value, "value", getattr(value, "score", None))
    if isinstance(extracted, (int, float)):
        numeric_value = float(extracted)
        if not math.isfinite(numeric_value):
            return None
        return round(numeric_value, 4)
    return None


def _weighted_score(scores: Dict[str, float | None], weights: Dict[str, float]) -> float | None:
    weighted_sum = 0.0
    active_weight = 0.0
    for key, score in scores.items():
        if score is None:
            continue
        weight = weights.get(key, 0.0)
        if weight <= 0:
            continue
        weighted_sum += score * weight
        active_weight += weight
    if active_weight == 0:
        return None
    return round(weighted_sum / active_weight, 4)


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            item_text = _stringify_value(item)
            if item_text:
                parts.append(item_text)
        return "；".join(parts)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def _extract_json_candidate(text: str) -> str:
    stripped = _strip_code_fences(text)
    if not stripped:
        return stripped
    first_curly = stripped.find("{")
    last_curly = stripped.rfind("}")
    if first_curly != -1 and last_curly != -1 and last_curly > first_curly:
        return stripped[first_curly : last_curly + 1]
    first_square = stripped.find("[")
    last_square = stripped.rfind("]")
    if first_square != -1 and last_square != -1 and last_square > first_square:
        return stripped[first_square : last_square + 1]
    return stripped


def _parse_response_model(response_model: Any, raw_text: str) -> Any:
    candidate = _extract_json_candidate(raw_text)

    validation_methods = [
        getattr(response_model, "model_validate_json", None),
        getattr(response_model, "parse_raw", None),
    ]
    for method in validation_methods:
        if method is None:
            continue
        try:
            return method(candidate)
        except Exception:
            pass

    try:
        from json_repair import repair_json

        repaired_candidate = repair_json(candidate)
    except Exception:
        repaired_candidate = candidate

    for method in validation_methods:
        if method is None:
            continue
        try:
            return method(repaired_candidate)
        except Exception:
            pass

    raise ValueError(f"unable_to_parse_streamed_json:{candidate[:500]}")


@dataclass(frozen=True)
class RagasRuntimeConfig:
    enabled: bool = True
    llm_provider: str = "openai"
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_timeout: float = 120.0
    llm_max_retries: int = 2
    llm_adapter: str = "auto"
    llm_force_stream: bool = False
    embedding_provider: str = "huggingface"
    embedding_model: str | None = None
    embedding_device: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    context_limit: int = 5

    @classmethod
    def from_env(cls) -> "RagasRuntimeConfig":
        load_dotenv()
        timeout_raw = _env_value("RAGAS_LLM_TIMEOUT", "RAGAS_TIMEOUT", default="120")
        retries_raw = _env_value("RAGAS_LLM_MAX_RETRIES", default="2")
        context_limit_raw = _env_value("RAGAS_CONTEXT_LIMIT", default="5")
        llm_provider = (_env_value("RAGAS_LLM_PROVIDER", "RAGAS_PROVIDER", default="openai") or "openai").strip().lower()
        embedding_provider = (_env_value("RAGAS_EMBEDDING_PROVIDER", default="huggingface") or "huggingface").strip().lower()
        return cls(
            enabled=_env_bool("RAGAS_ENABLED", default=True),
            llm_provider=llm_provider,
            llm_model=_env_value("RAGAS_LLM_MODEL", default=DEFAULT_RAGAS_LLM_MODEL),
            llm_base_url=_env_value(
                "RAGAS_LLM_BASE_URL",
                "RAGAS_BASE_URL",
                "OPENAI_BASE_URL" if llm_provider == "openai" else "",
            ),
            llm_api_key=_env_value(
                "RAGAS_LLM_API_KEY",
                "RAGAS_API_KEY",
                "OPENAI_API_KEY" if llm_provider == "openai" else "",
            ),
            llm_timeout=float(timeout_raw) if timeout_raw else 120.0,
            llm_max_retries=int(retries_raw) if retries_raw else 2,
            llm_adapter=_env_value("RAGAS_LLM_ADAPTER", default="auto") or "auto",
            llm_force_stream=_env_bool("RAGAS_LLM_FORCE_STREAM", default=False),
            embedding_provider=embedding_provider,
            embedding_model=_env_value("RAGAS_EMBEDDING_MODEL", default=DEFAULT_RAGAS_EMBEDDING_MODEL),
            embedding_device=_env_value("RAGAS_EMBEDDING_DEVICE", "EMBEDDING_DEVICE"),
            embedding_base_url=_env_value(
                "RAGAS_EMBEDDING_BASE_URL",
                "OPENAI_BASE_URL" if embedding_provider == "openai" else "",
            ),
            embedding_api_key=_env_value(
                "RAGAS_EMBEDDING_API_KEY",
                "OPENAI_API_KEY" if embedding_provider == "openai" else "",
            ),
            context_limit=max(int(context_limit_raw or "5"), 1),
        )


class OpenAIStreamingInstructorLLM(InstructorLLM):  # type: ignore[misc]
    def __init__(
        self,
        *,
        client: Any,
        model: str,
        temperature: float = 0.0,
    ):
        self.temperature = temperature
        super().__init__(
            client=client,
            model=model,
            provider="openai",
            temperature=temperature,
        )

    async def _astream_json_text(self, prompt: str) -> str:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=True,
        )

        chunks: List[str] = []
        async for event in stream:
            for choice in getattr(event, "choices", []) or []:
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            chunks.append(item["text"])
                        elif hasattr(item, "text") and isinstance(item.text, str):
                            chunks.append(item.text)

        text = "".join(chunks).strip()
        if not text:
            raise ValueError("empty_stream_content")
        return text

    async def agenerate(self, prompt: str, response_model: Any) -> Any:
        raw_text = await self._astream_json_text(prompt)
        return _parse_response_model(response_model, raw_text)

    def generate(
        self,
        prompt: str,
        response_model: Any,
    ) -> Any:
        if self.is_async:
            return self._run_async_in_current_loop(self.agenerate(prompt, response_model))
        return asyncio.run(self.agenerate(prompt, response_model))


class RagasRuntime:
    def __init__(self, config: RagasRuntimeConfig):
        self.config = config
        self._metrics: Dict[str, Any] = {}
        self._initialize()

    def _initialize(self) -> None:
        try:
            from ragas.dataset_schema import SingleTurnSample
            from openai import AsyncOpenAI
            from ragas.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AnswerCorrectness,
                AnswerRelevancy,
                ContextPrecisionWithReference,
                ContextRecall,
                Faithfulness,
            )
        except ImportError as exc:
            raise ImportError("ragas_runtime_dependencies_missing") from exc

        if self.config.llm_provider != "openai":
            raise ValueError(f"unsupported_ragas_llm_provider:{self.config.llm_provider}")

        llm_base_url = self.config.llm_base_url
        llm_api_key = self.config.llm_api_key
        if not llm_base_url and not llm_api_key:
            raise ValueError("missing_ragas_llm_configuration")

        llm_client = AsyncOpenAI(
            api_key=llm_api_key or "EMPTY",
            base_url=llm_base_url,
            timeout=self.config.llm_timeout,
            max_retries=self.config.llm_max_retries,
        )
        llm_adapter = (self.config.llm_adapter or "auto").strip().lower()
        use_streaming_json_adapter = (
            self.config.llm_force_stream
            or llm_adapter in {"streaming_json", "proxy_stream", "stream"}
            or (llm_adapter == "auto" and llm_base_url and not _is_official_openai_base_url(llm_base_url))
        )

        if use_streaming_json_adapter:
            llm = OpenAIStreamingInstructorLLM(
                client=llm_client,
                model=self.config.llm_model or DEFAULT_RAGAS_LLM_MODEL,
            )
        else:
            llm = llm_factory(
                self.config.llm_model or DEFAULT_RAGAS_LLM_MODEL,
                provider="openai",
                client=llm_client,
                adapter=self.config.llm_adapter,
            )

        embedding_provider = self.config.embedding_provider
        if embedding_provider in {"huggingface", "hf"}:
            embeddings = HuggingFaceEmbeddings(
                model=self.config.embedding_model or DEFAULT_RAGAS_EMBEDDING_MODEL,
                device=_normalize_embedding_device(self.config.embedding_device),
            )
        elif embedding_provider == "openai":
            embedding_base_url = self.config.embedding_base_url or llm_base_url
            embedding_api_key = self.config.embedding_api_key or llm_api_key
            if not embedding_base_url and not embedding_api_key:
                raise ValueError("missing_ragas_embedding_configuration")
            embedding_client = AsyncOpenAI(
                api_key=embedding_api_key or "EMPTY",
                base_url=embedding_base_url,
                timeout=self.config.llm_timeout,
                max_retries=self.config.llm_max_retries,
            )
            embeddings = OpenAIEmbeddings(
                client=embedding_client,
                model=self.config.embedding_model or "text-embedding-3-small",
            )
        else:
            raise ValueError(f"unsupported_ragas_embedding_provider:{embedding_provider}")

        self._metrics = {
            "answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings),
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "context_recall": ContextRecall(llm=llm),
            "context_precision": ContextPrecisionWithReference(llm=llm),
        }
        self._sample_type = SingleTurnSample

    def _score_metric(self, metric: Any, sample_kwargs: Dict[str, Any]) -> float | None:
        if hasattr(metric, "single_turn_score"):
            sample = self._sample_type(**sample_kwargs)
            return _round_score(metric.single_turn_score(sample))
        if hasattr(metric, "score"):
            return _round_score(metric.score(**sample_kwargs))
        raise AttributeError(f"{metric.__class__.__name__} has no supported scoring method")

    def score(
        self,
        *,
        question_text: str,
        answer: Any,
        reference: Any,
        contexts: List[str],
    ) -> Dict[str, Any]:
        answer_text = _stringify_value(answer)
        reference_text = _stringify_value(reference)
        question_kwargs = {
            "user_input": question_text.strip(),
        }
        response_kwargs = {
            **question_kwargs,
            "response": answer_text,
        }
        metric_kwargs = {
            "answer_correctness": {
                **response_kwargs,
                "reference": reference_text,
            },
            "faithfulness": {
                **response_kwargs,
                "retrieved_contexts": contexts,
            },
            "answer_relevancy": response_kwargs,
            "context_recall": {
                **question_kwargs,
                "retrieved_contexts": contexts,
                "reference": reference_text,
            },
            "context_precision": {
                **question_kwargs,
                "retrieved_contexts": contexts,
                "reference": reference_text,
            },
        }

        metric_scores: Dict[str, float | None] = {}
        metric_errors: List[str] = []
        for metric_name, metric in self._metrics.items():
            try:
                metric_scores[metric_name] = self._score_metric(metric, metric_kwargs[metric_name])
            except Exception as exc:
                metric_scores[metric_name] = None
                metric_errors.append(f"{metric_name}: {_format_error(exc)}")

        ragas_score = _weighted_score(metric_scores, RAGAS_METRIC_WEIGHTS)
        if ragas_score is None:
            return {
                "available": False,
                "reason": "metric_scoring_failed",
                "error": metric_errors[0] if metric_errors else None,
                "errors": metric_errors,
                "contexts_used": len(contexts),
                "answer_correctness": None,
                "faithfulness": None,
                "answer_relevancy": None,
                "context_recall": None,
                "context_precision": None,
                "ragas_score": None,
            }

        return {
            "available": True,
            "reason": "ok" if not metric_errors else "partial_metric_failure",
            "error": metric_errors[0] if metric_errors else None,
            "errors": metric_errors,
            "contexts_used": len(contexts),
            "answer_correctness": metric_scores.get("answer_correctness"),
            "faithfulness": metric_scores.get("faithfulness"),
            "answer_relevancy": metric_scores.get("answer_relevancy"),
            "context_recall": metric_scores.get("context_recall"),
            "context_precision": metric_scores.get("context_precision"),
            "ragas_metric_weights": RAGAS_METRIC_WEIGHTS,
            "ragas_score": ragas_score,
        }


def prepare_ragas_runtime(
    config: RagasRuntimeConfig | None = None,
) -> tuple[RagasRuntime | None, str | None, str | None]:
    runtime_config = config or RagasRuntimeConfig.from_env()
    if not runtime_config.enabled:
        return None, "ragas_disabled", None

    try:
        return RagasRuntime(runtime_config), None, None
    except ImportError as exc:
        return None, "ragas_not_installed", _format_error(exc)
    except Exception as exc:
        if isinstance(exc, ValueError):
            reason = str(exc)
        else:
            reason = "ragas_runtime_init_failed"
        return None, reason, _format_error(exc)


def collect_ragas_contexts(pred_answer: Dict, debug_detail: Dict | None = None, limit: int = 5) -> List[str]:
    contexts: List[str] = []
    seen: set[str] = set()

    for citation in pred_answer.get("citations", []) or []:
        snippet = " ".join(str(citation.get("evidence_snippet", "")).split())
        if snippet and snippet not in seen:
            seen.add(snippet)
            contexts.append(snippet)

    for result in (debug_detail or {}).get("retrieval_results", []) or []:
        text = " ".join(str(result.get("text", "")).split())
        if text and text not in seen:
            seen.add(text)
            contexts.append(text)

    return contexts[:limit]


def score_with_ragas(
    *,
    question_text: str,
    answer: Any,
    reference: Any,
    contexts: List[str],
    runtime: RagasRuntime | None = None,
    unavailable_reason: str | None = None,
    unavailable_error: str | None = None,
) -> Dict[str, Any]:
    base_result = {
        "available": False,
        "reason": None,
        "error": None,
        "errors": [],
        "contexts_used": len(contexts),
        "answer_correctness": None,
        "faithfulness": None,
        "answer_relevancy": None,
        "context_recall": None,
        "context_precision": None,
        "ragas_metric_weights": RAGAS_METRIC_WEIGHTS,
        "ragas_score": None,
    }

    if not question_text.strip():
        base_result["reason"] = "empty_question"
        return base_result
    if answer in (None, ""):
        base_result["reason"] = "empty_answer"
        return base_result
    if reference in (None, ""):
        base_result["reason"] = "empty_reference"
        return base_result
    if not contexts:
        base_result["reason"] = "no_contexts"
        return base_result

    if runtime is None:
        base_result["reason"] = unavailable_reason or "ragas_runtime_unavailable"
        base_result["error"] = unavailable_error
        return base_result

    result = runtime.score(
        question_text=question_text,
        answer=answer,
        reference=reference,
        contexts=contexts,
    )
    merged_result = dict(base_result)
    merged_result.update(result)
    return merged_result
