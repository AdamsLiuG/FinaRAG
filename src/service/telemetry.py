"""ContextVar-isolated tracing, OpenTelemetry-compatible spans, JSONL sinks, and optional Langfuse."""

from __future__ import annotations

import base64
import contextvars
import json
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

TRACE_ID_VAR: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("finarag_trace_id", default=None)
SPAN_ID_VAR: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("finarag_span_id", default=None)
PARENT_SPAN_ID_VAR: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("finarag_parent_span_id", default=None)

_TRACEPARENT_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
_LOW_CONFIDENCE_FLAGS = {
    "metric_evidence_mismatch",
    "metric_caliber_conflict_refused",
    "implicit_intent_evidence_mismatch",
    "claim_contradicted",
}

try:
    from opentelemetry import trace as _otel_trace  # type: ignore
except ImportError:
    _otel_trace = None


def new_trace_id() -> str:
    return secrets.token_hex(16)


def new_span_id() -> str:
    return secrets.token_hex(8)


def current_trace_id() -> Optional[str]:
    return TRACE_ID_VAR.get()


def current_span_id() -> Optional[str]:
    return SPAN_ID_VAR.get()


def parse_traceparent(header_value: Optional[str]) -> Optional[Tuple[str, str]]:
    """Parse W3C traceparent header (`00-<trace_id>-<parent_id>-<flags>`)."""
    raw = str(header_value or "").strip().lower()
    if not raw:
        return None
    match = _TRACEPARENT_RE.match(raw)
    if not match:
        return None
    trace_id, parent_id, _ = match.groups()
    if trace_id == "0" * 32 or parent_id == "0" * 16:
        return None
    return trace_id, parent_id


def bind_trace_context(traceparent: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
    """Bind trace_id and span_id into current contextvars and return reset tokens."""
    parsed = parse_traceparent(traceparent)
    if parsed is not None:
        trace_id, parent_span_id = parsed
    else:
        trace_id, parent_span_id = new_trace_id(), None
    span_id = new_span_id()
    tokens = {
        "trace": TRACE_ID_VAR.set(trace_id),
        "span": SPAN_ID_VAR.set(span_id),
        "parent": PARENT_SPAN_ID_VAR.set(parent_span_id),
    }
    return trace_id, span_id, tokens


def reset_trace_context(tokens: Dict[str, Any]) -> None:
    if "parent" in tokens:
        PARENT_SPAN_ID_VAR.reset(tokens["parent"])
    if "span" in tokens:
        SPAN_ID_VAR.reset(tokens["span"])
    if "trace" in tokens:
        TRACE_ID_VAR.reset(tokens["trace"])


def inject_trace_headers() -> Dict[str, str]:
    """Build outbound W3C traceparent headers from active contextvars."""
    trace_id = TRACE_ID_VAR.get() or new_trace_id()
    span_id = SPAN_ID_VAR.get() or new_span_id()
    return {
        "traceparent": f"00-{trace_id}-{span_id}-01",
        "X-Trace-Id": trace_id,
    }


class TelemetrySpan:
    """Sync/async context manager emitting an OTel-compatible span record."""

    def __init__(
        self,
        name: str,
        *,
        attributes: Optional[Dict[str, Any]] = None,
        collector: Optional[List[Dict[str, Any]]] = None,
    ):
        self.name = name
        self.attributes = dict(attributes or {})
        self.collector = collector
        self.trace_id = TRACE_ID_VAR.get() or new_trace_id()
        self.parent_span_id = SPAN_ID_VAR.get()
        self.span_id = new_span_id()
        self._start_ns = 0
        self._start_iso = ""
        self._span_token: Any = None
        self._parent_token: Any = None
        self._trace_token: Any = None
        self._otel_ctx: Any = None
        self.record: Dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def _enter_common(self) -> "TelemetrySpan":
        self._start_ns = time.perf_counter_ns()
        self._start_iso = datetime.now(timezone.utc).isoformat()
        if TRACE_ID_VAR.get() is None:
            self._trace_token = TRACE_ID_VAR.set(self.trace_id)
        self._parent_token = PARENT_SPAN_ID_VAR.set(self.parent_span_id)
        self._span_token = SPAN_ID_VAR.set(self.span_id)
        if _otel_trace is not None:
            tracer = _otel_trace.get_tracer("finarag")
            self._otel_ctx = tracer.start_as_current_span(self.name)
            self._otel_ctx.__enter__()
        return self

    def _exit_common(self, exc_type, exc, tb) -> bool:
        duration_ms = round((time.perf_counter_ns() - self._start_ns) / 1_000_000.0, 2)
        status = "error" if exc_type is not None else "ok"
        if exc_type is not None:
            self.attributes["error.type"] = exc_type.__name__
        self.record = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self._start_iso,
            "duration_ms": duration_ms,
            "status": status,
            "attributes": dict(self.attributes),
        }
        if self.collector is not None:
            self.collector.append(self.record)
        if self._otel_ctx is not None:
            try:
                self._otel_ctx.__exit__(exc_type, exc, tb)
            except Exception:
                pass
        if self._span_token is not None:
            SPAN_ID_VAR.reset(self._span_token)
        if self._parent_token is not None:
            PARENT_SPAN_ID_VAR.reset(self._parent_token)
        if self._trace_token is not None:
            TRACE_ID_VAR.reset(self._trace_token)
        return False

    def __enter__(self) -> "TelemetrySpan":
        return self._enter_common()

    def __exit__(self, exc_type, exc, tb) -> bool:
        return self._exit_common(exc_type, exc, tb)

    async def __aenter__(self) -> "TelemetrySpan":
        return self._enter_common()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return self._exit_common(exc_type, exc, tb)


class TelemetryManager:
    """Writes request traces + adaptive feedback signals and exports optional Langfuse batches."""

    def __init__(
        self,
        *,
        trace_dir: Optional[Path | str] = None,
        langfuse_host: Optional[str] = None,
        langfuse_public_key: Optional[str] = None,
        langfuse_secret_key: Optional[str] = None,
        http_transport: Optional[httpx.BaseTransport] = None,
    ):
        self.trace_dir = Path(trace_dir or os.getenv("SERVICE_TRACE_DIR", "logs"))
        self._file_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.requests_recorded = 0
        self.adaptive_signals_emitted = 0
        self.langfuse_exports = 0

        raw_host = (langfuse_host if langfuse_host is not None else os.getenv("LANGFUSE_HOST", "")).strip().rstrip("/")
        pub = langfuse_public_key if langfuse_public_key is not None else os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sec = langfuse_secret_key if langfuse_secret_key is not None else os.getenv("LANGFUSE_SECRET_KEY", "")
        self.langfuse_host: Optional[str] = None
        self._langfuse_auth_header: Optional[str] = None
        self._http_transport = http_transport
        if raw_host and pub and sec:
            parsed = urlparse(raw_host)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                self.langfuse_host = raw_host
                token = base64.b64encode(f"{pub}:{sec}".encode("utf-8")).decode("ascii")
                self._langfuse_auth_header = f"Basic {token}"

    def _append_jsonl(self, filename: str, entry: Dict[str, Any]) -> None:
        try:
            with self._file_lock:
                self.trace_dir.mkdir(parents=True, exist_ok=True)
                with open(self.trace_dir / filename, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def record_request(
        self,
        *,
        trace_id: str,
        question: str,
        schema: str,
        answer_dict: Dict[str, Any],
        spans: List[Dict[str, Any]],
        latency_ms: float,
        cache_hit: bool = False,
        degraded_level: int = 0,
    ) -> None:
        flags = list(answer_dict.get("validation_flags") or [])
        confidence = str(answer_dict.get("confidence") or "")
        usage = (answer_dict.get("response_data") or {}).get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "question": question,
            "schema": schema,
            "final_answer": answer_dict.get("final_answer"),
            "confidence": confidence,
            "validation_flags": flags,
            "route_mode": (answer_dict.get("route_info") or {}).get("route_mode"),
            "cache_hit": cache_hit,
            "degraded_level": degraded_level,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "latency_ms": round(float(latency_ms), 2),
            "spans": spans,
        }
        self._append_jsonl("service_trace.jsonl", entry)
        with self._stats_lock:
            self.requests_recorded += 1

        if (_LOW_CONFIDENCE_FLAGS & set(flags)) or confidence == "low":
            qplan = answer_dict.get("query_plan") if isinstance(answer_dict.get("query_plan"), dict) else {}
            atom_payload = qplan.get("metric_atom") if isinstance(qplan.get("metric_atom"), dict) else {}
            matched_flag = next((f for f in flags if f in _LOW_CONFIDENCE_FLAGS), None)
            claim_audit = answer_dict.get("claim_audit") if isinstance(answer_dict.get("claim_audit"), dict) else {}
            feedback_entry = {
                "source": "online_telemetry",
                "timestamp": entry["timestamp"],
                "trace_id": trace_id,
                "question": question,
                "schema": schema,
                "confidence": confidence,
                "validation_flags": flags,
                "trigger": matched_flag or (flags[0] if flags else "low_confidence"),
                "atom_id": atom_payload.get("atom_id"),
                "metric": qplan.get("metric") or atom_payload.get("standard_name"),
                "queries": list(qplan.get("search_queries") or answer_dict.get("search_queries") or []),
                "new_hits": 0,
                "rounds": 0,
                "gap_closed": False,
                "outcome": "telemetry_flagged",
                "claim_contradicted_count": int(claim_audit.get("contradicted_count") or 0),
                "route_mode": entry["route_mode"],
                "query_plan": qplan,
            }
            self._append_jsonl("adaptive_trace.jsonl", feedback_entry)
            with self._stats_lock:
                self.adaptive_signals_emitted += 1

        if self.langfuse_host and self._langfuse_auth_header:
            self._export_langfuse_batch(entry)

    def _export_langfuse_batch(self, entry: Dict[str, Any]) -> None:
        batch = [
            {
                "id": entry["trace_id"],
                "type": "trace-create",
                "timestamp": entry["timestamp"],
                "body": {
                    "id": entry["trace_id"],
                    "name": "finarag_qa",
                    "input": {"question": entry["question"], "schema": entry["schema"]},
                    "output": {"final_answer": entry["final_answer"], "confidence": entry["confidence"]},
                    "metadata": {
                        "validation_flags": entry["validation_flags"],
                        "degraded_level": entry["degraded_level"],
                        "cache_hit": entry["cache_hit"],
                        "latency_ms": entry["latency_ms"],
                    },
                },
            }
        ]
        try:
            with httpx.Client(timeout=2.0, transport=self._http_transport) as client:
                resp = client.post(
                    f"{self.langfuse_host}/api/public/ingestion",
                    headers={"Authorization": self._langfuse_auth_header},
                    json={"batch": batch},
                )
                if 200 <= resp.status_code < 300:
                    with self._stats_lock:
                        self.langfuse_exports += 1
        except Exception:
            pass

    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            return {
                "requests_recorded": self.requests_recorded,
                "adaptive_signals_emitted": self.adaptive_signals_emitted,
                "langfuse_enabled": self.langfuse_host is not None,
                "langfuse_exports": self.langfuse_exports,
                "otel_available": _otel_trace is not None,
            }
