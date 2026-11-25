"""Request-local trajectory observation and bounded JSONL storage."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import fcntl
from functools import lru_cache, wraps
import hashlib
import hmac
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from threading import Lock
import time
import unicodedata

from src.storage.base import validate_safe_path


SCHEMA_VERSION = 1
MAX_RECORD_BYTES = 1 << 20
_MAX_DEPTH = 20
_MAX_ITEMS = 1024
_MAX_STEPS = 384
_MAX_TEXT = 32768
_PROJECT_ITEMS = 80
_DEPLOYMENT_ROOT = Path(__file__).resolve().parents[2]
_FILE_NAME = "trajectories.jsonl"
_REDACTED = "[redacted]"
_LABEL = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}")
_SESSION_ID = re.compile(r"[A-Za-z0-9._~-]{1,128}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_PRIVATE = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
    r"|(?<!\d)(?:\+?\d[ .()-]?){10,19}(?:[xX](?!\w)|(?!\d))"
    r"|(?i:\b(?:bearer|basic)\s+\S+|\b(?:sk|ghp|gho|ghu|github_pat|xox[baprs])[-_][A-Za-z0-9_-]+"
    r"|(?:password|authorization|api[_-]?key|access[_-]?token|credential|secret)[=:]\S+)"
)
_SCHEMA_NAMES = {
    "boolean", "comparative", "json", "long_text", "name", "names", "number", "text", "string",
    "numeric", "entity", "entity_list",
}
_ANSWER_MARKERS = {None, "N/A", "[redacted]"}
_REQUEST_FIELDS = frozenset({
    "question", "schema", "company", "company_name", "companies", "thread_id",
    "turn_id", "checkpoint_id", "decision", "question_meta", "budget_context",
    "budget_summary", "budget_snapshot", "approval_enabled", "resume", "tool_name",
})
_TEXT_FIELDS = frozenset({
    "question", "sub_question", "original_query", "normalized_query", "query",
    "input_text", "search_queries", "queries", "rephrased_questions", "generated_text",
    "model_response", "response", "response_data", "spec", "expression", "prompt",
    "text", "content", "snippet", "evidence_snippet", "raw_response",
    "full_context", "working_context", "reviewed_candidate", "hybrid_context",
    "answer_text", "excerpt", "context_summary",
})
_ID_FIELDS = frozenset({
    "doc_id", "document_id", "source_doc_id", "source", "document", "source_id",
    "sha1_name", "pdf_sha1", "selected_doc_id", "report_id", "chunk_id",
    "parent_chunk_id", "table_id", "chart_id", "picture_id", "logical_table_id",
    "parent_block_id", "point_id", "candidate_doc_ids", "allowed_grounding_doc_ids",
    "doc_ids", "selected_doc_ids", "candidate_ids", "ranked_ids", "source_ids",
    "evidence_ids", "evidence_id", "fact_id", "evidence_doc_id", "matched_child_chunk_ids", "company_name",
    "company", "companies", "mentioned_companies", "entity", "entity_id", "broker_name",
    "security_code", "stock_code",
})
_SOURCE_FIELDS = frozenset({
    "doc_id", "document_id", "source_doc_id", "source", "document", "source_id", "sha1_name",
    "pdf_sha1", "selected_doc_id", "report_id", "candidate_doc_ids", "allowed_grounding_doc_ids",
    "doc_ids", "selected_doc_ids", "candidate_ids", "ranked_ids", "source_ids",
    "evidence_id", "evidence_ids", "evidence_doc_id", "chunk_id", "parent_chunk_id",
    "matched_child_chunk_ids", "table_id", "chart_id",
})
_DOCUMENT_FIELDS = _SOURCE_FIELDS - {
    "evidence_id", "evidence_ids", "chunk_id", "parent_chunk_id", "matched_child_chunk_ids", "table_id", "chart_id",
}
_LABEL_FIELDS = frozenset({
    "stage", "node", "mode", "route", "route_target", "route_mode", "retrieval_mode",
    "operation", "expected_answer_type", "task_type", "sort_order", "fallback_route",
    "metric_atom_id", "atom_id", "stop_reason", "confidence", "status", "operation_status",
    "kind", "type", "tool", "tool_name", "task_id", "approval_status", "error_code",
    "reason_code", "decision", "action", "scope_type", "claim_status", "currency",
    "period", "dimension", "chunk_type", "node_type", "evidence_type", "question_kind",
    "report_type", "doc_source_type", "market_type", "basis", "measurement_basis",
    "match_type", "level", "flags", "approval_reasons", "validation_flags", "validated_flags",
    "topic_flags", "required_topic_flags", "retrieval_sources", "depends_on", "dependencies",
    "fallback_for", "retry_for", "coverage", "intent", "outcome", "terminal_status",
    "task_ids", "trigger", "backend_scope", "parent_retrieval_mode", "relation_type", "operator",
})
_NUMBER_FIELDS = frozenset({
    "ordinal", "iteration", "round", "rounds_used", "adaptive_rounds_used", "wave",
    "wave_count", "page", "page_index", "source_page", "page_start", "page_end",
    "row_idx", "col_idx", "rank", "year", "report_year", "fiscal_year", "n_results",
    "top_k", "top_n", "k", "count", "hit_count", "candidate_pool_size_before_rerank",
    "tree_nav_hits", "kg_hits", "pre_adaptive_rounds", "candidate_count", "result_count",
    "relevant_pages", "page_span", "limit", "score", "combined_score", "ranking_score",
    "rerank_score", "distance", "match_score", "intent_confidence", "latency_ms", "ms",
    "elapsed_ms", "duration_ms", "prompt_tokens", "completion_tokens", "total_tokens",
    "max_tokens", "spent_tokens", "remaining_tokens", "max_latency_ms", "degraded_level",
    "contradicted_count", "verified_count", "supported_count", "unverified_count",
    "claim_count", "priority", "max_rounds", "status_code", "documents_batch_size", "pool_cap",
    "final_relevance_score", "colbert_score", "distance_rrf", "hops", "evidence_page",
    "max_sections", "max_pages", "ndigits", "start_page", "end_page",
    "max_hops", "max_paths",
})
_BOOLEAN_FIELDS = frozenset({
    "is_comparative", "rule_route", "rule_answer", "cache_hit", "budget_exceeded", "fast_path",
    "approval_enabled", "resume", "claim_contradicted", "timed_out", "disconnected", "refused",
    "enabled", "rerank_enabled", "rerank_applied", "gap_closed", "has_table_context",
    "has_chart_context", "execution_covered", "verification_executed", "ascending", "rule",
})
_OBJECT_FIELDS = frozenset({
    "query_plan", "route_info", "route_decision", "filters", "scope", "metric_atom", "task",
    "tasks", "metadata", "results", "retrieval_results", "candidates", "ranked", "ranked_results",
    "citations", "references", "sources", "budget_snapshot", "budget_summary", "cost", "token_usage",
    "degraded_events", "claim_audit", "tool_result", "result", "outputs", "arguments", "input",
    "inputs", "result_data", "outcomes", "evidence", "evidence_ledger", "evidence_entries",
    "deep_research_plan", "deep_research_status", "selected_report", "table_grounding_result",
    "chart_grounding_result", "stage_events", "graph_stage_events", "final_answer_dict",
    "data", "rows", "calculation", "pot_result", "variables", "sections", "pages", "answer", "answers",
    "paths", "edges", "evidence_pages", "claims", "numeric_checks",
})
_STEP_FIELDS = _TEXT_FIELDS | _ID_FIELDS | _LABEL_FIELDS | _NUMBER_FIELDS | _BOOLEAN_FIELDS | _OBJECT_FIELDS | {
    "schema", "final_answer", "metric", "raw_value", "value", "unit", "base_unit", "raw_unit",
    "result_unit", "waves", "rounds", "error",
    "normalized_cny", "threshold_cny",
}
_FINANCIAL_OBJECTS = {
    "table_grounding_result", "chart_grounding_result", "evidence_entries", "citations",
    "rows", "pot_result", "variables",
}
_FINANCIAL_NUMBER = re.compile(r"[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?")
_FINANCIAL_UNITS = {
    "", "%", "元", "万元", "亿元", "千元", "百万元", "元/股", "股", "万股", "亿股", "倍", "家", "个",
    "人民币", "港元", "美元", "CNY", "RMB", "HKD", "USD", "percent", "shares",
}
_FINANCIAL_NUMERIC_KINDS = {"number", "numeric"}
_KNOWN_STATUSES = {
    "ok", "error", "failed", "refused", "timeout", "timed_out", "disconnected", "cancelled",
    "pending", "pending_approval", "approved", "rejected", "resumed", "cached", "cache_hit",
    "completed", "unavailable", "empty", "skipped_budget", "skipped_sufficient", "blocked",
}
_observer: ContextVar[_Observer | None] = ContextVar("finarag_trajectory_observer", default=None)
_scope: ContextVar[tuple[str, tuple[str, ...]]] = ContextVar("finarag_trajectory_scope", default=("", ()))
_stats_lock = Lock()
_capture_failures = 0


class TrajectoryError(ValueError):
    """A trajectory failure whose public message contains only fixed text."""


@dataclass(frozen=True)
class TrajectoryStep:
    step_key: str
    stage: str
    operation: str
    parents: tuple[str, ...] = ()
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    source_ids: tuple[str, ...] = ()
    cost: dict = field(default_factory=dict)
    status: str = "ok"
    error_code: str | None = None


@dataclass(frozen=True)
class TrajectoryRecord:
    trace_id: str
    entry_point: str
    operation_status: str
    created_at: str
    versions: dict = field(default_factory=dict)
    request: dict = field(default_factory=dict)
    query_plan: dict = field(default_factory=dict)
    stage_events: tuple[dict, ...] = ()
    steps: tuple[TrajectoryStep, ...] = ()
    cost: dict = field(default_factory=dict)
    validation_flags: tuple[str, ...] = ()
    final_answer: dict = field(default_factory=dict)
    confidence: str | None = None
    replayable: bool = False
    redactions: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION
    run_key: str | None = None
    continuation_of: str | None = None

    def to_dict(self) -> dict:
        return _json_value(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict) -> TrajectoryRecord:
        payload = _json_value(payload)
        if not isinstance(payload, dict):
            raise TrajectoryError("Invalid trajectory record")
        version = payload.get("schema_version")
        if type(version) is not int or version != SCHEMA_VERSION:
            raise TrajectoryError("Unsupported trajectory schema version")
        if set(payload) - {item.name for item in fields(cls)}:
            raise TrajectoryError("Invalid trajectory record fields")
        for name in ("trace_id", "entry_point", "operation_status", "created_at"):
            if not isinstance(payload.get(name), str) or not payload[name] or len(payload[name]) > 256:
                raise TrajectoryError("Invalid trajectory record header")
        for name in ("versions", "request", "query_plan", "cost", "final_answer"):
            if name in payload and not isinstance(payload[name], dict):
                raise TrajectoryError("Invalid trajectory record projection")
        for name in ("confidence", "run_key", "continuation_of"):
            item = payload.get(name)
            if item is not None and (not isinstance(item, str) or len(item) > 256):
                raise TrajectoryError("Invalid trajectory record metadata")
        if "replayable" in payload and type(payload["replayable"]) is not bool:
            raise TrajectoryError("Invalid trajectory replay status")
        for name in ("stage_events", "validation_flags", "redactions"):
            payload[name] = _tuple_of(payload.get(name, []), dict if name == "stage_events" else str)
        given_steps = payload.get("steps", [])
        if not isinstance(given_steps, list) or len(given_steps) > _MAX_STEPS:
            raise TrajectoryError("Invalid trajectory steps")
        steps = tuple(_parse_step(step) for step in given_steps)
        if len({step.step_key for step in steps}) != len(steps):
            raise TrajectoryError("Duplicate trajectory step")
        payload["steps"] = steps
        return cls(**payload)


def _tuple_of(value, item_type):
    if not isinstance(value, list) or any(not isinstance(item, item_type) for item in value):
        raise TrajectoryError("Invalid trajectory sequence")
    return tuple(value)


def _valid_step_key(value):
    return (
        isinstance(value, str) and 0 < len(value) <= 512
        and all(_LABEL.fullmatch(part) for part in value.split("/"))
    )


def _parse_step(payload) -> TrajectoryStep:
    if not isinstance(payload, dict) or set(payload) - {item.name for item in fields(TrajectoryStep)}:
        raise TrajectoryError("Invalid trajectory step fields")
    if not _valid_step_key(payload.get("step_key")):
        raise TrajectoryError("Invalid trajectory step key")
    for name in ("stage", "operation", "status"):
        if name == "status" and name not in payload:
            continue
        value = payload.get(name)
        if not isinstance(value, str) or not _LABEL.fullmatch(value):
            raise TrajectoryError("Invalid trajectory step header")
    code = payload.get("error_code")
    if code is not None and (not isinstance(code, str) or not _LABEL.fullmatch(code)):
        raise TrajectoryError("Invalid trajectory error code")
    for name in ("inputs", "outputs", "cost"):
        if name in payload and not isinstance(payload[name], dict):
            raise TrajectoryError("Invalid trajectory step projection")
    for name in ("source_ids", "parents"):
        payload[name] = _tuple_of(payload.get(name, []), str)
    if any(not _valid_step_key(parent) for parent in payload["parents"]):
        raise TrajectoryError("Invalid trajectory dependency")
    return TrajectoryStep(**payload)


def _json_value(value, depth=0):
    """Validate and copy a JSON value with explicit limits."""
    if depth > _MAX_DEPTH:
        raise TrajectoryError("Trajectory nesting limit exceeded")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str) and len(value) <= _MAX_TEXT:
        return value
    if type(value) is int and value.bit_length() <= 63:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping) and len(value) <= _MAX_ITEMS:
        if any(not isinstance(key, str) or len(key) > 256 for key in value):
            raise TrajectoryError("Invalid trajectory field name")
        return {key: _json_value(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and len(value) <= _MAX_ITEMS:
        return [_json_value(item, depth + 1) for item in value]
    raise TrajectoryError("Invalid or oversized trajectory value")


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise TrajectoryError("Duplicate trajectory field")
    return result


def observation_failure() -> None:
    """Count a capture failure without recording request or exception data."""
    global _capture_failures
    with _stats_lock:
        _capture_failures += 1


def observation_stats() -> dict[str, int]:
    with _stats_lock:
        return {"capture_failures": _capture_failures}


def _validated_path(path: Path | str, root: Path) -> Path:
    try:
        raw = Path(path).expanduser().absolute()
        if validate_safe_path(raw, root) == raw:
            return raw
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    raise TrajectoryError("Invalid trajectory storage path") from None


@contextmanager
def _open_directory(base, target, *, create=False):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(base, flags)
    try:
        for part in target.relative_to(base).parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child_fd
        yield fd
    finally:
        os.close(fd)


class TrajectoryStore:
    """Write service-owned trajectories and read bounded versioned records."""

    def __init__(self, root: Path | str, *, allowed_base_dir: Path | str):
        if not allowed_base_dir:
            raise TrajectoryError("Invalid trajectory storage root")
        try:
            self.allowed_base_dir = Path(allowed_base_dir).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            raise TrajectoryError("Invalid trajectory storage root") from None
        self.root = _validated_path(root, self.allowed_base_dir)
        self.path = _validated_path(self.root / _FILE_NAME, self.root)

    def append(self, record: TrajectoryRecord) -> Path:
        try:
            payload = record.to_dict()
            TrajectoryRecord.from_dict(payload)
            line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            data = (line + "\n").encode("utf-8")
        except (AttributeError, TypeError, UnicodeError, RecursionError):
            raise TrajectoryError("Invalid trajectory record") from None
        if len(data) > MAX_RECORD_BYTES:
            raise TrajectoryError("Trajectory record size limit exceeded")

        with self._open(self.path, write=True) as handle:
            self._repair_trailing_line(handle)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return self.path

    def iter_records(self, path: Path | str, *, strict_trailing=False):
        target = Path(path).expanduser().absolute()
        if not target.is_relative_to(self.root) and not Path(path).is_absolute():
            target = self.root / path
        target = _validated_path(target, self.root)
        with self._open(target) as handle:
            index = 0
            while raw := handle.readline(MAX_RECORD_BYTES + 1):
                index += 1
                try:
                    if len(raw) > MAX_RECORD_BYTES:
                        raise TrajectoryError("Trajectory record size limit exceeded")
                    if not raw.endswith(b"\n"):
                        observation_failure()
                        if strict_trailing:
                            raise TrajectoryError("Torn trailing trajectory record")
                        return
                    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
                    record = TrajectoryRecord.from_dict(payload)
                except TrajectoryError as exc:
                    raise TrajectoryError(f"{exc} on line {index}") from None
                except (UnicodeError, ValueError, RecursionError):
                    raise TrajectoryError(f"Invalid trajectory record on line {index}") from None
                yield record

    @contextmanager
    def _open(self, path: Path, *, write=False):
        file_fd = None
        try:
            _validated_path(self.root, self.allowed_base_dir)
            path = _validated_path(path, self.root)
            flags = (os.O_CREAT | os.O_RDWR if write else os.O_RDONLY) | os.O_NOFOLLOW | os.O_NONBLOCK
            with _open_directory(self.allowed_base_dir, path.parent, create=write) as dir_fd:
                if write:
                    if os.fstat(dir_fd).st_uid != os.geteuid():
                        raise TrajectoryError("Invalid trajectory storage owner")
                    os.fchmod(dir_fd, 0o700)
                file_fd = os.open(path.name, flags, 0o600, dir_fd=dir_fd)
            info = os.fstat(file_fd)
            if not stat.S_ISREG(info.st_mode):
                raise TrajectoryError("Invalid trajectory storage file")
            if write:
                if info.st_uid != os.geteuid() or info.st_nlink != 1:
                    raise TrajectoryError("Invalid trajectory storage owner")
                os.fchmod(file_fd, 0o600)
            with os.fdopen(file_fd, "r+b" if write else "rb") as handle:
                file_fd = None
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX if write else fcntl.LOCK_SH)
                yield handle
        except OSError:
            raise TrajectoryError("Trajectory storage unavailable") from None
        finally:
            if file_fd is not None:
                os.close(file_fd)

    @staticmethod
    def _repair_trailing_line(handle) -> None:
        end = handle.seek(0, os.SEEK_END)
        if not end:
            return
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
        observation_failure()
        start = max(0, end - MAX_RECORD_BYTES)
        handle.seek(start)
        boundary = handle.read().rfind(b"\n")
        if boundary == -1 and start:
            raise TrajectoryError("Torn trajectory record exceeds size limit")
        end = start + boundary + 1
        handle.truncate(end)
        handle.seek(end)

    def _hmac_key(self) -> bytes:
        configured = os.getenv("FINARAG_TRAJECTORY_HMAC_KEY")
        if configured:
            return configured.encode("utf-8")
        with self._open(self.root / ".trajectory-hmac-key", write=True) as handle:
            key = handle.read(33)
            if not key:
                key = secrets.token_bytes(32)
                handle.write(key)
                handle.flush()
                os.fsync(handle.fileno())
            if len(key) != 32:
                raise TrajectoryError("Invalid trajectory pseudonym key")
        return key


def harness_enabled() -> bool:
    return os.getenv("FINARAG_HARNESS_ENABLED", "0").strip().lower() in {"1", "true", "yes"}


def current_observer() -> _Observer | None:
    return _observer.get()


def capture_active() -> bool:
    return _observer.get() is not None


def current_prefix() -> str:
    return _scope.get()[0]


@contextmanager
def bind_observer(observer):
    """Bind an explicitly propagated observer in a graph worker."""
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


@contextmanager
def step_scope(prefix, *, parents=()):
    """Qualify step keys and provide absolute default dependencies."""
    current, inherited = _scope.get()
    scoped = f"{current}/{prefix}" if current else prefix
    token = _scope.set((scoped, tuple(parents) or inherited))
    try:
        yield
    finally:
        _scope.reset(token)


def _label(value):
    if not isinstance(value, str) or len(value) > 256:
        return None
    normalized = unicodedata.normalize("NFKC", value).strip()
    if _LABEL.fullmatch(normalized) and not _PRIVATE.search(normalized):
        return normalized
    return None


def _number(value):
    if type(value) is int and value.bit_length() <= 63:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    return None


def _known_mapping(value):
    if isinstance(value, Mapping):
        return value
    if type(value).__module__ in {"src.query_plan", "src.retrieval_filters"} and type(value).__name__ in {
        "QueryPlan", "RetrievalFilters",
    }:
        return {item.name: getattr(value, item.name) for item in fields(value)}
    return None


def _fixture_value(value, depth=0):
    """Copy reviewed fixtures using only JSON types and the known planning records."""
    if depth > _MAX_DEPTH:
        raise TrajectoryError("Trajectory nesting limit exceeded")
    mapping = _known_mapping(value)
    if mapping is not None:
        if len(mapping) > _MAX_ITEMS or any(not isinstance(key, str) or len(key) > 256 for key in mapping):
            raise TrajectoryError("Invalid trajectory field name")
        return {key: _fixture_value(item, depth + 1) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_ITEMS:
            raise TrajectoryError("Trajectory sequence limit exceeded")
        return [_fixture_value(item, depth + 1) for item in value]
    return _json_value(value)


class _LiveProjection:
    def __init__(self, key: bytes):
        self._key = key
        self.redactions = set()

    def _hide(self, reason):
        self.redactions.add(reason)
        return _REDACTED

    def pseudonym(self, domain, value):
        if type(value) is int and value.bit_length() <= 63:
            value = str(value)
        if not isinstance(value, str) or not 0 < len(value) <= 256:
            return None
        value = unicodedata.normalize("NFKC", value).strip()
        if not value or any(char in value for char in "/\\\r\n\0") or _PRIVATE.search(value):
            self.redactions.add("private_identifier")
            return None
        self.redactions.add("identifiers_pseudonymized")
        return self.digest(domain, value)

    def digest(self, domain, value):
        return "hmac:" + hmac.new(self._key, f"{domain}\0{value}".encode("utf-8"), hashlib.sha256).hexdigest()[:32]

    def session_id(self, domain, value):
        if isinstance(value, str) and _SESSION_ID.fullmatch(value):
            self.redactions.add("identifiers_pseudonymized")
            return self.digest(domain, value)
        return None

    def mapping(self, value, *, context="", depth=0, request=False, operation=None):
        mapping = _known_mapping(value)
        if mapping is None or depth > 10:
            self.redactions.add("unsupported_value")
            return {}
        allow = _REQUEST_FIELDS if request else _STEP_FIELDS
        projected = {}
        for name in itertools.islice(mapping, _PROJECT_ITEMS * 2):
            if name not in allow:
                self.redactions.add("unclassified_fields")
                continue
            item = mapping[name]
            result = self.value(name, item, context=context, depth=depth + 1, operation=operation)
            if result is not None or item is None:
                projected[name] = result
        if len(mapping) > _PROJECT_ITEMS * 2:
            self.redactions.add("projection_limit")
        return projected

    def value(self, name, item, *, context, depth, operation):
        if depth > 10:
            self.redactions.add("projection_limit")
            return None
        if item is None:
            return None
        if name in _TEXT_FIELDS or name == "question_meta":
            return self._hide("private_text")
        if name in {"thread_id", "turn_id", "checkpoint_id", "error"}:
            return self._hide("private_fields")
        if name == "schema":
            label = _label(item)
            return label if label in _SCHEMA_NAMES else self._hide("unknown_schema")
        if name == "final_answer":
            if type(item) is bool or (isinstance(item, str) and item in _ANSWER_MARKERS):
                return item
            return self._hide("answer_text")
        if name in {"unit", "base_unit", "raw_unit", "result_unit"}:
            unit = unicodedata.normalize("NFKC", item).strip() if isinstance(item, str) and len(item) <= 32 else None
            return unit if unit in _FINANCIAL_UNITS else self._hide("unclassified_text")
        if name == "metric":
            return self._hide("unclassified_text")
        if name in {"raw_value", "value", "normalized_cny", "threshold_cny"}:
            if (
                context in _FINANCIAL_OBJECTS or (context == "arguments" and name == "threshold_cny")
                or (context == "data" and name == "value" and operation == "pot_calculate")
            ):
                if _number(item) is not None:
                    return item
                if (
                    isinstance(item, str) and len(item) <= 40
                    and _FINANCIAL_NUMBER.fullmatch(item) and not _PRIVATE.search(item)
                ):
                    return item
            return self._hide("unclassified_value")
        if name == "budget_context":
            if type(item).__module__ == "src.service.budget_controller" and type(item).__name__ == "BudgetContext":
                item = item.summary()
            return self.mapping(item, context="budget_summary", depth=depth)
        if isinstance(item, (tuple, list)):
            if len(item) > _PROJECT_ITEMS:
                self.redactions.add("projection_limit")
            return [
                result for element in item[:_PROJECT_ITEMS]
                if (result := self.value(name, element, context=context, depth=depth + 1, operation=operation)) is not None
            ]
        if name in _ID_FIELDS:
            domain = "document" if name in _DOCUMENT_FIELDS else name
            return self.pseudonym(domain, item)
        if name in _BOOLEAN_FIELDS:
            return item if type(item) is bool else None
        if name in _NUMBER_FIELDS or name == "rounds":
            return _number(item)
        if name in _LABEL_FIELDS:
            return _label(item) or self._hide("unclassified_text")
        if name in _OBJECT_FIELDS:
            if name in {"answer", "answers"} and _known_mapping(item) is None:
                return self._hide("answer_text")
            return self.mapping(item, context=name, depth=depth, operation=operation)
        if name == "waves":
            return _label(item)
        return None


def _source_ids(*projections):
    ordered = {}

    def visit(value, name=None):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, key)
        elif isinstance(value, list):
            for item in value:
                visit(item, name)
        elif name in _SOURCE_FIELDS and isinstance(value, str) and value != _REDACTED:
            ordered[value] = None

    for projection in projections:
        visit(projection)
    return tuple(ordered)


@lru_cache(maxsize=1)
def _repository_revision():
    git = _DEPLOYMENT_ROOT / ".git"
    try:
        if git.is_file():
            pointer = git.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir: "):
                return "unknown"
            git = _DEPLOYMENT_ROOT / pointer.removeprefix("gitdir: ")
        revision = (git / "HEAD").read_text(encoding="utf-8").strip()
        if revision.startswith("ref: "):
            reference = revision.removeprefix("ref: ")
            if not re.fullmatch(r"refs/[A-Za-z0-9_./-]+", reference) or ".." in Path(reference).parts:
                return "unknown"
            common_dir = git / "commondir"
            if common_dir.is_file():
                git = git / common_dir.read_text(encoding="utf-8").strip()
            loose = git / reference
            if loose.is_file():
                revision = loose.read_text(encoding="utf-8").strip()
            else:
                with (git / "packed-refs").open(encoding="utf-8") as packed:
                    for line in packed:
                        if line.rstrip().endswith(f" {reference}"):
                            revision = line.split(" ", 1)[0]
                            break
    except (OSError, UnicodeError):
        return "unknown"
    return revision if _GIT_REVISION.fullmatch(revision) else "unknown"


def _versions(*, fixture, request):
    if fixture:
        versions = request.get("versions") if isinstance(request, Mapping) else None
        return {**versions, "replay_policy_version": "1"} if isinstance(versions, dict) else {
            "replay_policy_version": "1",
        }
    from src.service.semantic_cache import resolve_cache_versions

    versions = asdict(resolve_cache_versions())
    versions = {name: _label(value) or "unknown" for name, value in versions.items()}
    versions["program_revision"] = _label(os.getenv("FINARAG_PROGRAM_REVISION")) or _repository_revision()
    versions["replay_policy_version"] = "1"
    return versions


def _status(answer, status, entry_point):
    if status is not None:
        return status if isinstance(status, str) and status in _KNOWN_STATUSES else "error"
    if answer.get("approval_status") == "pending":
        return "pending_approval"
    if answer.get("approval_status") == "rejected":
        return "rejected"
    if answer.get("cache_hit") is True:
        return "cached"
    if answer.get("final_answer") == "N/A":
        return "refused"
    if "resume" in entry_point:
        return "resumed"
    return "ok"


def _exception_status(exc):
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return "disconnected"
    return "error"


def _versioned_trace_id(projection):
    from src.service.telemetry import current_trace_id

    current = current_trace_id()
    if current:
        value = projection.pseudonym("trace", current)
        if value:
            return value
    return secrets.token_hex(16)


def _provided_cost(answer, projection):
    from src.service.tracing import token_usage_of

    try:
        usage = token_usage_of(answer)
    except (AttributeError, TypeError, ValueError, OverflowError):
        usage = {}
    if not any(usage.values()) and isinstance(answer.get("token_usage"), dict):
        usage = answer["token_usage"]
    if projection is not None:
        usage = projection.mapping(usage)
    return usage


class _Observer:
    def __init__(self, entry_point, request, *, store=None):
        self._store = store
        self._live = None if store is None else _LiveProjection(store._hmac_key())
        self._start = time.perf_counter()
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._trace_id = secrets.token_hex(16) if self._live is None else _versioned_trace_id(self._live)
        self._versions = _versions(fixture=self._live is None, request=request)
        self._entry_point = _label(entry_point) or "unknown"
        self._request = self._mapping(request, request=True)
        self._thread = None
        self._continuation_of = None
        if self._live is not None and isinstance(request, Mapping):
            self._thread = self._live.session_id("thread", request.get("thread_id"))
            if self._entry_point in {"resume", "continue"} or request.get("resume") is True:
                self._continuation_of = self._run_key(request.get("checkpoint_id"), request.get("turn_id"))
        self._lock = Lock()
        self._steps = {}
        self._finished = False
        self.record: TrajectoryRecord | None = None

    def _mapping(self, value, **kwargs):
        if self._live is not None:
            return self._live.mapping(value, **kwargs)
        result = _fixture_value(value)
        if not isinstance(result, dict):
            raise TrajectoryError("Invalid fixture observation")
        return result

    def _run_key(self, checkpoint, turn):
        if self._live is None or self._thread is None:
            return None
        value = self._live.session_id("checkpoint", checkpoint) or self._live.session_id("turn", turn)
        if value is None:
            return None
        return self._live.digest("continuation", f"{self._thread}:{value}")

    def has_step(self, step_key) -> bool:
        with self._lock:
            return step_key in self._steps

    def add_step(self, *, stage, operation, step_key, parents, inputs, outputs, costs, status, error_code):
        with self._lock:
            if self._finished:
                return
            if (
                not _valid_step_key(step_key) or not _label(stage) or not _label(operation) or not _label(status)
                or any(not _valid_step_key(parent) for parent in parents)
            ):
                raise TrajectoryError("Invalid trajectory step identity")
            if self._live is not None and any(_PRIVATE.search(key) for key in (step_key, *parents)):
                raise TrajectoryError("Invalid trajectory step identity")
            if step_key in self._steps:
                raise TrajectoryError("Duplicate trajectory step")
            if len(self._steps) >= _MAX_STEPS:
                raise TrajectoryError("Trajectory step limit exceeded")
            projected_inputs = self._mapping(inputs, operation=operation)
            projected_outputs = self._mapping(outputs, operation=operation)
            self._steps[step_key] = TrajectoryStep(
                step_key=step_key,
                stage=stage,
                operation=operation,
                parents=tuple(dict.fromkeys(parents)),
                inputs=projected_inputs,
                outputs=projected_outputs,
                source_ids=_source_ids(projected_inputs, projected_outputs),
                cost=self._mapping(costs or {}),
                status=status,
                error_code=_label(error_code),
            )

    def finish(self, answer=None, *, status=None):
        try:
            with self._lock:
                if self._finished:
                    return self.record
                self._finished = True
                self.record = self._build_record(answer, status)
            if self._store is not None:
                self._store.append(self.record)
        except Exception:
            if self._live is None:
                raise
            observation_failure()
        return self.record

    def _build_record(self, answer, status):
        answer = answer if isinstance(answer, Mapping) else {}
        steps = tuple(self._steps[key] for key in sorted(self._steps))
        projected = self._mapping(answer)
        content = projected.get("final_answer", _REDACTED)
        # Preserve numeric answers from the declared answer schema when they are typed.
        if self._live is not None and self._request.get("schema") in _FINANCIAL_NUMERIC_KINDS:
            if _number(answer.get("final_answer")) is not None:
                content = answer["final_answer"]
        final_fields = {"relevant_pages", "references", "citations", "approval_status", "cache_hit"}
        final_answer = {key: projected[key] for key in final_fields if key in projected}
        final_answer["final_answer"] = content
        plan = projected.get("query_plan")
        if not plan:
            for step in steps:
                plan = step.outputs.get("query_plan")
                if not plan and isinstance(step.outputs.get("route_decision"), dict):
                    plan = step.outputs["route_decision"].get("query_plan")
                if isinstance(plan, dict) and plan:
                    break
        events = projected.get("graph_stage_events") or projected.get("stage_events")
        if not isinstance(events, list):
            events = []
            for step in steps:
                for event in step.outputs.get("stage_events") or ():
                    if isinstance(event, dict):
                        events.append(event)
        flags = projected.get("validation_flags") or []
        confidence = projected.get("confidence")
        budget = projected.get("budget_summary") or projected.get("budget_snapshot")
        cost = {
            "elapsed_ms": round((time.perf_counter() - self._start) * 1000, 2),
            "token_usage": _provided_cost(answer, self._live),
        }
        if isinstance(budget, dict):
            cost["budget"] = budget
        run_key = self._run_key(answer.get("checkpoint_id"), answer.get("turn_id"))
        redactions = tuple(sorted(self._live.redactions)) if self._live is not None else ()
        return TrajectoryRecord(
            trace_id=self._trace_id, entry_point=self._entry_point, created_at=self._created_at,
            operation_status=_status(answer, status, self._entry_point),
            request=self._request, versions=self._versions,
            query_plan=plan if isinstance(plan, dict) else {},
            stage_events=tuple(events[:_MAX_ITEMS]), steps=steps, cost=cost,
            validation_flags=tuple(flag for flag in flags if isinstance(flag, str)) if isinstance(flags, list) else (),
            final_answer=final_answer, confidence=confidence if isinstance(confidence, str) else None,
            replayable=self._live is None, redactions=redactions, run_key=run_key, continuation_of=self._continuation_of,
        )


def capture_step(stage, *, operation, step_key, inputs, outputs, parents=(), costs=None, status="ok", error_code=None):
    """Observe an operational boundary and return its fully qualified key."""
    observer = _observer.get()
    if observer is None:
        return step_key
    prefix, scope_parents = _scope.get()
    key = f"{prefix}/{step_key}" if prefix else step_key
    try:
        observer.add_step(
            stage=stage, operation=operation, step_key=key, inputs=inputs, outputs=outputs,
            parents=tuple(parents) or scope_parents, costs=costs, status=status, error_code=error_code,
        )
    except Exception:
        if observer._live is None:
            raise
        observation_failure()
    return key


@contextmanager
def _bind_run(observer):
    scope_token = _scope.set(("", ()))
    try:
        with bind_observer(observer):
            try:
                yield observer
            except BaseException as exc:
                observer.finish(status=_exception_status(exc))
                raise
            else:
                observer.finish()
    finally:
        _scope.reset(scope_token)


@contextmanager
def record_trajectory(*, store=None, entry_point, request):
    """Own a live trajectory at an outermost, enabled entry point."""
    if _observer.get() is not None or not harness_enabled():
        yield None
        return
    try:
        if store is None:
            configured = os.getenv("FINARAG_TRAJECTORY_ROOT", "outputs/finarag_trajectories").strip()
            if not configured:
                raise TrajectoryError("Invalid trajectory storage root")
            root = Path(configured).expanduser()
            if root.is_absolute():
                base = root.parent
            else:
                base = _DEPLOYMENT_ROOT
                root = base / root
            store = TrajectoryStore(root, allowed_base_dir=base)
        observer = _Observer(entry_point, request, store=store)
    except Exception:
        observation_failure()
        yield None
        return
    with _bind_run(observer):
        yield observer


@contextmanager
def collect_trajectory(entry_point, request):
    """Collect reviewed synthetic fixture observations in memory."""
    if _observer.get() is not None:
        raise TrajectoryError("Fixture requires an independent trajectory")
    with _bind_run(_Observer(entry_point, request)) as observer:
        yield observer


def observed_run(entry_point, request_names, *, answer=True):
    """Record a synchronous answer entry point while preserving its call signature."""
    def decorate(func):
        signature = inspect.signature(func)

        @wraps(func)
        def wrapped(*args, **kwargs):
            if capture_active() or not harness_enabled():
                return func(*args, **kwargs)
            bound = signature.bind_partial(*args, **kwargs)
            request = {name: bound.arguments[name] for name in request_names if name in bound.arguments}
            with record_trajectory(entry_point=entry_point, request=request) as observer:
                result = func(*args, **kwargs)
                if observer is not None and answer:
                    observer.finish(result)
                return result

        return wrapped

    return decorate
