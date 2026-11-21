"""FastAPI async service with true SSE streaming, budget control, semantic cache, and telemetry."""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hmac
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.graph.builder import run_finarag_graph
from src.service.budget_controller import BudgetContext
from src.service.semantic_cache import SemanticCache
from src.service.streaming import AsyncLLMClient, async_stream_qa_events, shutdown_cpu_executor
from src.service.telemetry import (
    TelemetryManager,
    TelemetrySpan,
    bind_trace_context,
    current_trace_id,
    reset_trace_context,
)
from src.service.tracing import token_usage_of

_log = logging.getLogger(__name__)

_DEFAULT_TOKEN_BUDGET = int(os.getenv("SERVICE_TOKEN_BUDGET", "8000"))
_DEFAULT_LATENCY_BUDGET_MS = float(os.getenv("SERVICE_LATENCY_BUDGET_MS", "30000"))

_AllowedSchema = Literal[
    "string",
    "number",
    "boolean",
    "json",
    "name",
    "names",
    "text",
    "long_text",
    "comparative",
]


class QaRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(..., min_length=1, max_length=4000)
    answer_schema: _AllowedSchema = Field(default="string", alias="schema")
    max_tokens: Optional[int] = Field(default=None, ge=100, le=100000)
    max_latency_ms: Optional[float] = Field(default=None, ge=500.0, le=120000.0)
    use_cache: bool = True

    @field_validator("question")
    @classmethod
    def _strip_and_require_nonempty(cls, v: str) -> str:
        cleaned = str(v or "").strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class _ProcessorHolder:
    def __init__(self, processor: Any = None):
        self._processor = processor
        self._lock = threading.Lock()

    def get(self) -> Any:
        if self._processor is None:
            with self._lock:
                if self._processor is None:
                    from src.questions_processing import QuestionsProcessor

                    self._processor = QuestionsProcessor()
        return self._processor


async def _verify_api_key(request: Request) -> None:
    expected = os.getenv("SERVICE_API_KEY", "").strip()
    if not expected:
        # TODO(security): SERVICE_API_KEY is optional only for local loopback dev/testing; enforce in production.
        return
    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else request.headers.get("X-API-Key", "").strip()
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


def _execute_qa_sync(
    processor: Any,
    question: str,
    schema: str,
    budget_ctx: BudgetContext,
    spans: List[Dict[str, Any]],
) -> Dict[str, Any]:
    with TelemetrySpan("qa_sync_execution", attributes={"schema": schema}, collector=spans):
        is_full = hasattr(processor, "_resolve_question_route") and hasattr(processor, "_build_query_plan")
        if is_full:
            return run_finarag_graph(processor, question, schema, budget_context=budget_ctx)
        try:
            return processor.process_question(question, schema, budget_context=budget_ctx)
        except TypeError:
            return processor.process_question(question, schema)


class _TraceAndSecurityHeadersMiddleware:
    """Pure ASGI middleware binding W3C traceparent and security headers with zero SSE buffering."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        inbound_traceparent = None
        for k, v in scope.get("headers") or ():
            if k.lower() == b"traceparent":
                inbound_traceparent = v.decode("latin-1")
                break
        trace_id, span_id, tokens = bind_trace_context(inbound_traceparent)

        async def _send_wrapped(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or ())
                headers.extend(
                    [
                        (b"x-trace-id", trace_id.encode("ascii")),
                        (b"traceparent", f"00-{trace_id}-{span_id}-01".encode("ascii")),
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
                    ]
                )
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, _send_wrapped)
        finally:
            reset_trace_context(tokens)


def create_fastapi_app(
    processor: Any = None,
    *,
    cache: Optional[SemanticCache] = None,
    telemetry: Optional[TelemetryManager] = None,
    llm_client: Optional[AsyncLLMClient] = None,
    default_token_budget: Optional[int] = None,
    default_latency_budget_ms: Optional[float] = None,
) -> FastAPI:
    holder = _ProcessorHolder(processor)
    semantic_cache = cache or SemanticCache()
    telemetry_mgr = telemetry or TelemetryManager()
    owns_llm_client = llm_client is None
    token_budget_default = default_token_budget if default_token_budget is not None else _DEFAULT_TOKEN_BUDGET
    latency_budget_default = (
        default_latency_budget_ms if default_latency_budget_ms is not None else _DEFAULT_LATENCY_BUDGET_MS
    )

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        app.state.processor_holder = holder
        app.state.semantic_cache = semantic_cache
        app.state.telemetry = telemetry_mgr
        app.state.llm_client = llm_client or AsyncLLMClient()
        try:
            yield
        finally:
            if owns_llm_client and getattr(app.state, "llm_client", None) is not None:
                await app.state.llm_client.aclose()
            shutdown_cpu_executor()

    app = FastAPI(title="FinaRAG Async Service", version="1.0.0", lifespan=_lifespan)
    app.add_middleware(_TraceAndSecurityHeadersMiddleware)

    @app.get("/healthz")
    async def healthz():
        from src.adaptive_retrieval import adaptive_retrieval_enabled
        from src.metric_atoms import MetricAtomStore
        from src.retrieval import _INDEX_CACHE

        knowledge = {"valid": False, "detail": "unknown"}
        try:
            MetricAtomStore.load()
            snap_path = Path(__file__).resolve().parents[2] / "config" / "atom_snapshot.json"
            snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
            knowledge = {"valid": True, "schema_version": snapshot.get("schema_version")}
        except Exception:
            knowledge = {"valid": False, "detail": "knowledge validation failed"}

        return {
            "status": "ok" if knowledge["valid"] else "degraded",
            "trace_id": current_trace_id(),
            "knowledge_generation": knowledge,
            "index_cache": _INDEX_CACHE.stats(),
            "semantic_cache": semantic_cache.stats(),
            "retrieval_storage_backend": os.getenv("RETRIEVAL_STORAGE_BACKEND", "file"),
            "document_store_backend": os.getenv("DOCUMENT_STORE_BACKEND", "memory"),
            "adaptive_enabled": adaptive_retrieval_enabled(),
        }

    @app.get("/metrics/cache")
    async def metrics_cache(_auth: None = Depends(_verify_api_key)):
        from src.retrieval import _INDEX_CACHE

        return {
            "trace_id": current_trace_id(),
            "semantic_cache": semantic_cache.stats(),
            "telemetry": telemetry_mgr.stats(),
            "index_cache": _INDEX_CACHE.stats(),
        }

    @app.post("/qa")
    async def qa_endpoint(req: QaRequest, _auth: None = Depends(_verify_api_key)):
        started = time.perf_counter()
        trace_id = current_trace_id() or ""
        spans: List[Dict[str, Any]] = []
        schema = req.answer_schema

        if req.use_cache:
            cached = semantic_cache.get(req.question, schema)
            if cached is not None:
                latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
                usage = token_usage_of(cached)
                await asyncio.to_thread(
                    telemetry_mgr.record_request,
                    trace_id=trace_id,
                    question=req.question,
                    schema=schema,
                    answer_dict=cached,
                    spans=spans,
                    latency_ms=latency_ms,
                    cache_hit=True,
                    degraded_level=0,
                )
                return {
                    "trace_id": trace_id,
                    "question": req.question,
                    "final_answer": cached.get("final_answer"),
                    "confidence": cached.get("confidence"),
                    "references": cached.get("references") or [],
                    "validation_flags": cached.get("validation_flags") or [],
                    "token_usage": usage,
                    "budget_exceeded": False,
                    "degraded_level": 0,
                    "cache_hit": True,
                    "latency_ms": latency_ms,
                }

        budget_ctx = BudgetContext(
            max_tokens=req.max_tokens if req.max_tokens is not None else token_budget_default,
            max_latency_ms=req.max_latency_ms if req.max_latency_ms is not None else latency_budget_default,
        )

        try:
            proc = holder.get()
            ctx = contextvars.copy_context()
            answer_dict = await asyncio.to_thread(
                ctx.run,
                _execute_qa_sync,
                proc,
                req.question,
                schema,
                budget_ctx,
                spans,
            )
        except Exception:
            _log.exception("POST /qa failed trace_id=%s", trace_id)
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "trace_id": trace_id},
            )

        usage = token_usage_of(answer_dict)
        if budget_ctx.spent_tokens == 0 and (usage["prompt_tokens"] or usage["completion_tokens"]):
            budget_ctx.record_usage(usage["prompt_tokens"], usage["completion_tokens"])
        degraded_level = budget_ctx.degradation_level
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if req.use_cache and degraded_level < 3:
            semantic_cache.set(req.question, schema, answer_dict)

        await asyncio.to_thread(
            telemetry_mgr.record_request,
            trace_id=trace_id,
            question=req.question,
            schema=schema,
            answer_dict=answer_dict,
            spans=spans,
            latency_ms=latency_ms,
            cache_hit=False,
            degraded_level=degraded_level,
        )
        return {
            "trace_id": trace_id,
            "question": req.question,
            "final_answer": answer_dict.get("final_answer"),
            "confidence": answer_dict.get("confidence"),
            "references": answer_dict.get("references") or [],
            "validation_flags": answer_dict.get("validation_flags") or [],
            "token_usage": usage,
            "budget_exceeded": degraded_level >= 3,
            "degraded_level": degraded_level,
            "budget_summary": budget_ctx.summary(),
            "cache_hit": False,
            "latency_ms": latency_ms,
        }

    @app.post("/qa/stream")
    async def qa_stream_endpoint(req: QaRequest, request: Request, _auth: None = Depends(_verify_api_key)):
        trace_id = current_trace_id() or ""
        schema = req.answer_schema
        budget_ctx = BudgetContext(
            max_tokens=req.max_tokens if req.max_tokens is not None else token_budget_default,
            max_latency_ms=req.max_latency_ms if req.max_latency_ms is not None else latency_budget_default,
        )
        proc = holder.get()

        async def _sse_generator():
            try:
                async for event in async_stream_qa_events(
                    proc,
                    req.question,
                    schema,
                    budget_context=budget_ctx,
                    is_disconnected=request.is_disconnected,
                    telemetry_manager=telemetry_mgr,
                    cache=semantic_cache if req.use_cache else None,
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception:
                _log.exception("POST /qa/stream failed trace_id=%s", trace_id)
                yield f"data: {json.dumps({'stage': 'error', 'error': 'Internal server error', 'trace_id': trace_id})}\n\n"

        return StreamingResponse(
            _sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_fastapi_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="FinaRAG Async FastAPI Service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    uvicorn.run(
        "src.service.fastapi_app:app",
        host=args.host,
        port=args.port,
        workers=max(1, int(args.workers)),
    )


if __name__ == "__main__":
    main()
