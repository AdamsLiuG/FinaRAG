"""FastAPI async service with true SSE streaming, budget control, semantic cache, and telemetry."""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hmac
import inspect
import json
import logging
import os
import threading
import time
from contextlib import ExitStack, aclosing, asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.graph.builder import run_finarag_graph
from src.harness.trajectory_store import capture_step, harness_enabled, record_trajectory
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
    thread_id: str | None = None

    @field_validator("question")
    @classmethod
    def _strip_and_require_nonempty(cls, v: str) -> str:
        cleaned = str(v or "").strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str
    interrupt_id: str
    decision: Literal["approve", "reject"]


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
    *,
    protocol_execution: bool = False,
) -> Dict[str, Any]:
    with TelemetrySpan("qa_sync_execution", attributes={"schema": schema}, collector=spans):
        is_full = hasattr(processor, "_resolve_question_route") and hasattr(processor, "_build_query_plan")
        if protocol_execution and all(callable(getattr(processor, name, None)) for name in (
            "route_question", "_build_query_plan", "_resolved_graph_engine",
        )):
            is_full = processor._resolved_graph_engine() == "langgraph"
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


class _ProtocolHolder:
    def __init__(self, factory):
        self._factory = factory
        self._lock = threading.Lock()
        self._instance = None
        self._admitting = True

    def get(self):
        with self._lock:
            if not self._admitting:
                raise RuntimeError("Protocol unavailable")
            if self._instance is None:
                self._instance = self._factory()
            return self._instance

    def take_for_shutdown(self):
        with self._lock:
            self._admitting = False
            instance, self._instance = self._instance, None
            return instance

    def start(self):
        with self._lock:
            self._admitting = True


def _protocol_enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes"}


def create_fastapi_app(
    processor: Any = None,
    *,
    cache: Optional[SemanticCache] = None,
    telemetry: Optional[TelemetryManager] = None,
    llm_client: Optional[AsyncLLMClient] = None,
    default_token_budget: Optional[int] = None,
    default_latency_budget_ms: Optional[float] = None,
    mcp_server_provider=None,
    a2a_store_provider=None,
    protocol_principal_resolver=None,
    checkpoint_db_path: str | Path | None = None,
) -> FastAPI:
    holder = _ProcessorHolder(processor)
    semantic_cache = cache or SemanticCache()
    telemetry_mgr = telemetry or TelemetryManager()
    owns_llm_client = llm_client is None
    token_budget_default = default_token_budget if default_token_budget is not None else _DEFAULT_TOKEN_BUDGET
    latency_budget_default = (
        default_latency_budget_ms if default_latency_budget_ms is not None else _DEFAULT_LATENCY_BUDGET_MS
    )
    mcp_holder = a2a_holder = session_holder = None
    checkpoint_enabled = _protocol_enabled("FINARAG_CHECKPOINT_ENABLED")
    approval_enabled = checkpoint_enabled and _protocol_enabled("FINARAG_HITL_ENABLED")

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        app.state.processor_holder = holder
        app.state.semantic_cache = semantic_cache
        app.state.telemetry = telemetry_mgr
        app.state.llm_client = llm_client or AsyncLLMClient()
        app.state.session_supplier = session_holder
        for binding in (mcp_holder, a2a_holder, session_holder):
            if binding is not None:
                binding.start()
        try:
            yield
        finally:
            try:
                if mcp_holder is not None:
                    server = await asyncio.to_thread(mcp_holder.take_for_shutdown)
                    if server is not None:
                        await asyncio.to_thread(server.close)
            finally:
                try:
                    if a2a_holder is not None:
                        store = await asyncio.to_thread(a2a_holder.take_for_shutdown)
                        if store is not None:
                            await store.aclose()
                finally:
                    if session_holder is not None:
                        await asyncio.to_thread(session_holder.take_for_shutdown)
                    if owns_llm_client and getattr(app.state, "llm_client", None) is not None:
                        await app.state.llm_client.aclose()
                    shutdown_cpu_executor()

    app = FastAPI(title="FinaRAG Async Service", version="1.0.0", lifespan=_lifespan)
    app.add_middleware(_TraceAndSecurityHeadersMiddleware)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, exc: RequestValidationError):
        if request.url.path.startswith("/sessions/") or (
            isinstance(exc.body, dict) and exc.body.get("thread_id") is not None
        ):
            return JSONResponse(status_code=422, content={
                "error": "invalid_session_request", "message": "Invalid session request", "trace_id": current_trace_id(),
            })
        return await request_validation_exception_handler(request, exc)

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
    async def qa_endpoint(req: QaRequest, request: Request, _auth: None = Depends(_verify_api_key)):
        with record_trajectory(
            entry_point="qa",
            request={"question": req.question, "schema": req.answer_schema, "thread_id": req.thread_id},
        ) as observer:
            started = time.perf_counter()
            trace_id = current_trace_id() or ""
            spans: List[Dict[str, Any]] = []
            schema = req.answer_schema

            if req.thread_id is not None:
                try:
                    principal, key = await session_request(request, write=True)
                    result = await asyncio.to_thread(run_session_turn, req, principal, key)
                except Exception as exc:
                    if observer is not None:
                        observer.finish(status="timeout" if isinstance(exc, TimeoutError) else "failed")
                    return session_error(exc)
                if observer is not None:
                    observer.finish(result)
                return JSONResponse(
                    status_code=202 if result["approval_status"] == "pending" else 200,
                    content=session_answer(result, req.question, started),
                )

            if req.use_cache:
                cached = semantic_cache.get(req.question, schema)
                if cached is not None:
                    if observer is not None:
                        capture_step(
                            "cache", operation="semantic_cache", step_key="cache", inputs={"schema": schema},
                            outputs={"cache_hit": True, "coverage": "absent"},
                        )
                        observer.finish(cached, status="cache_hit")
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
            except Exception as exc:
                if observer is not None:
                    observer.finish(status="timeout" if isinstance(exc, TimeoutError) else "failed")
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
            if observer is not None:
                observer.finish({**answer_dict, "budget_summary": budget_ctx.summary(), "degraded_level": degraded_level})
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
        if req.thread_id is not None:
            try:
                principal, key = await session_request(request, write=True)
                return await start_session_stream(req, request, principal, key)
            except Exception as exc:
                return session_error(exc)

        proc = holder.get()

        async def _sse_generator():
            try:
                async with aclosing(async_stream_qa_events(
                    proc,
                    req.question,
                    schema,
                    budget_context=budget_ctx,
                    is_disconnected=request.is_disconnected,
                    telemetry_manager=telemetry_mgr,
                    cache=semantic_cache if req.use_cache else None,
                )) as events:
                    async for event in events:
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

    async def protocol_auth(request: Request):
        from src.protocols.tool_registry import ToolPrincipal, valid_principal

        if not os.getenv("SERVICE_API_KEY", "").strip():
            raise HTTPException(status_code=401, detail="Unauthorized")
        try:
            await _verify_api_key(request)
        except TypeError:
            raise HTTPException(status_code=401, detail="Unauthorized") from None
        if protocol_principal_resolver is None:
            return ToolPrincipal("finarag-service", frozenset({"read"}))
        try:
            principal = protocol_principal_resolver(request)
            if inspect.isawaitable(principal):
                principal = await principal
        except Exception:
            raise HTTPException(status_code=403, detail="Access denied") from None
        if not valid_principal(principal):
            raise HTTPException(status_code=403, detail="Access denied")
        return principal

    def session_error(exc):
        if isinstance(exc, HTTPException):
            raise exc
        from src.memory.session_memory import SessionError
        from src.storage.base import StorageError, StorageSecurityError

        if isinstance(exc, SessionError):
            status_code, code, message = exc.status, exc.code, str(exc)
        elif isinstance(exc, (StorageError, StorageSecurityError, OSError)):
            status_code, code, message = 503, "session_unavailable", "Session storage unavailable"
        else:
            status_code, code, message = 500, "session_execution_failed", "Session execution failed"
        return JSONResponse(status_code=status_code, content={
            "error": code, "message": message, "trace_id": current_trace_id(),
        })

    async def session_request(request, *, write=False, reviewer=False):
        if not checkpoint_enabled:
            raise HTTPException(status_code=404, detail="Sessions unavailable")
        from src.memory.session_memory import SessionError, session_identifier

        principal = await protocol_auth(request)
        if "read" not in principal.roles or (reviewer and "reviewer" not in principal.roles):
            raise SessionError(403, "access_denied", "Access denied")
        key = session_identifier(request.headers.get("Idempotency-Key"), code="invalid_idempotency_key") if write else None
        return principal, key

    def get_session_service():
        from src.memory.session_memory import SessionError

        try:
            return session_holder.get()
        except SessionError:
            raise
        except Exception:
            raise SessionError(503, "session_unavailable", "Session storage unavailable") from None

    def run_session_turn(req, principal, key, on_event=None, on_admitted=None):
        memory, graph, proc = get_session_service()
        budget = BudgetContext(
            max_tokens=req.max_tokens if req.max_tokens is not None else token_budget_default,
            max_latency_ms=req.max_latency_ms if req.max_latency_ms is not None else latency_budget_default,
        )
        return memory.run_turn(
            proc, graph=graph, thread_id=req.thread_id, subject=principal.subject, key=key,
            question=req.question, schema=req.answer_schema, budget=budget, approval_enabled=approval_enabled,
            on_event=on_event, on_admitted=on_admitted,
        )

    def session_answer(result, question, started):
        answer = {**result, "trace_id": current_trace_id(),
                  "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        if result["approval_status"] != "pending":
            answer["question"] = question
        return answer

    async def start_session_stream(req, request, principal, key):
        loop = asyncio.get_running_loop()
        events = asyncio.Queue()
        admission = loop.create_future()
        disconnected = threading.Event()
        started = time.perf_counter()
        observer_ctx = None
        observer = None
        result = None
        operation_status = None
        if harness_enabled():
            observer_ctx = contextvars.copy_context()
            stack = ExitStack()
            observer = observer_ctx.run(
                stack.enter_context,
                record_trajectory(entry_point="qa_session_stream", request={
                    "question": req.question, "schema": req.answer_schema, "thread_id": req.thread_id,
                }),
            )

        def finish_capture():
            if observer is not None:
                observer.finish(result, status="disconnected" if delivery_cancelled else operation_status)
            if observer_ctx is not None:
                observer_ctx.run(stack.close)

        delivery_cancelled = False

        def on_event(event):
            if disconnected.is_set() or loop.is_closed():
                return
            safe = {name: event[name] for name in ("node", "stage", "ms", "degraded_level") if name in event}
            if safe.get("stage") == "done":
                safe["stage"] = "finalize"
            safe["trace_id"] = current_trace_id()
            try:
                loop.call_soon_threadsafe(events.put_nowait, safe)
            except RuntimeError:
                if not loop.is_closed():
                    raise

        def admit():
            if not admission.done():
                admission.set_result(None)

        def on_admitted():
            if not disconnected.is_set() and not loop.is_closed():
                loop.call_soon_threadsafe(admit)

        async def execute():
            nonlocal result, operation_status
            try:
                result = await asyncio.to_thread(run_session_turn, req, principal, key, on_event, on_admitted)
                stage = "approval_pending" if result["approval_status"] == "pending" else "done"
                await events.put({"stage": stage, **session_answer(result, req.question, started)})
            except Exception as exc:
                operation_status = "timeout" if isinstance(exc, TimeoutError) else "failed"
                try:
                    response = session_error(exc)
                except HTTPException as error:
                    if not admission.done():
                        admission.set_exception(error)
                    return
                if not admission.done():
                    admission.set_result(response)
                else:
                    await events.put({"stage": "error", "status": response.status_code, **json.loads(response.body)})
            finally:
                await events.put(None)

        worker = asyncio.create_task(execute(), context=observer_ctx.copy() if observer_ctx is not None else None)
        try:
            rejected = await admission
        except asyncio.CancelledError:
            delivery_cancelled = True
            disconnected.set()
            worker.add_done_callback(lambda _: finish_capture())
            raise
        except Exception:
            worker.add_done_callback(lambda _: finish_capture())
            raise
        if rejected is not None:
            await worker
            finish_capture()
            return rejected

        async def stream():
            nonlocal delivery_cancelled
            try:
                while (event := await events.get()) is not None:
                    if await request.is_disconnected():
                        delivery_cancelled = True
                        break
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except (asyncio.CancelledError, GeneratorExit):
                delivery_cancelled = True
                raise
            finally:
                disconnected.set()
                # An admitted operation finishes under the session lock even if delivery stops.
                worker.add_done_callback(lambda _: finish_capture())
                await asyncio.shield(worker)

        return StreamingResponse(
            stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if checkpoint_enabled:
        from src.graph.builder import build_finarag_graph
        from src.memory import ContextCompactor, SessionMemory, SqliteCheckpointSaver
        from src.memory.session_memory import SessionError

        def build_session_service():
            proc = holder.get()
            resolved_engine = getattr(proc, "_resolved_graph_engine", None)
            if (not callable(getattr(proc, "route_question", None)) or
                (resolved_engine is not None and resolved_engine() != "langgraph")):
                raise SessionError(503, "session_engine_unavailable", "Session graph engine unavailable")
            configured = os.getenv("FINARAG_CHECKPOINT_ROOT", "outputs/finarag_sessions").strip()
            if not configured:
                raise ValueError("Invalid checkpoint root")
            root = Path(configured).expanduser()
            if not root.is_absolute():
                root = Path(__file__).resolve().parents[2] / root
            if root != root.resolve():
                raise ValueError("Invalid checkpoint root")
            db_path = checkpoint_db_path if checkpoint_db_path is not None else root / "checkpoints.sqlite"
            saver = SqliteCheckpointSaver(db_path, allowed_base_dir=root)
            return SessionMemory(saver, ContextCompactor()), build_finarag_graph(proc, saver), proc

        session_holder = _ProtocolHolder(build_session_service)

        async def session_read(request, thread_id, *, history=False, before=None, limit=20):
            try:
                principal, _ = await session_request(request)
                def read():
                    memory, graph, _ = get_session_service()
                    if history:
                        return {"history": memory.history(thread_id, subject=principal.subject, before=before, limit=limit)}
                    return memory.describe(thread_id, subject=principal.subject, graph=graph)
                return await asyncio.to_thread(read)
            except Exception as exc:
                return session_error(exc)

        @app.get("/sessions/{thread_id}")
        async def session_status(thread_id: str, request: Request):
            return await session_read(request, thread_id)

        @app.get("/sessions/{thread_id}/history")
        async def session_history(
            thread_id: str, request: Request, before: str | None = None, limit: int = Query(20, ge=1, le=100),
        ):
            return await session_read(request, thread_id, history=True, before=before, limit=limit)

        @app.post("/sessions/{thread_id}/resume")
        async def session_resume(thread_id: str, req: ResumeRequest, request: Request):
            with record_trajectory(
                entry_point="resume", request={"thread_id": thread_id, "checkpoint_id": req.checkpoint_id},
            ) as observer:
                started = time.perf_counter()
                try:
                    principal, key = await session_request(request, write=True, reviewer=True)
                    def resume():
                        memory, graph, proc = get_session_service()
                        return memory.resume(
                            proc, graph=graph, thread_id=thread_id, subject=principal.subject,
                            key=key, **req.model_dump(),
                        )
                    result = await asyncio.to_thread(resume)
                    if observer is not None:
                        observer.finish(result)
                    return {**result, "trace_id": current_trace_id(),
                            "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
                except Exception as exc:
                    if observer is not None:
                        observer.finish(status="timeout" if isinstance(exc, TimeoutError) else "failed")
                    return session_error(exc)

    if _protocol_enabled("FINARAG_MCP_ENABLED"):
        from src.protocols.mcp_server import MCPServer, RESOURCES_ROOT, mount_mcp_routes
        from src.protocols.tool_registry import build_financial_tool_registry, default_policy

        def build_mcp_server():
            configured = os.getenv("FINARAG_PROTOCOL_CORPUS_ROOT")
            root = configured if configured is not None else Path(holder.get().documents_dir).parent
            registry = build_financial_tool_registry(root, policy=default_policy)
            try:
                return MCPServer(registry, resources_root=RESOURCES_ROOT)
            except Exception:
                registry.close()
                raise

        origins = [origin.strip() for origin in os.getenv("FINARAG_MCP_ALLOWED_ORIGINS", "").split(",")
                   if origin.strip()]
        mcp_holder = _ProtocolHolder(mcp_server_provider or build_mcp_server)
        mount_mcp_routes(app, mcp_holder.get, auth=protocol_auth, allowed_origins=origins)

    if _protocol_enabled("FINARAG_A2A_ENABLED"):
        from src.protocols.a2a_server import A2ATaskStore, mount_a2a_routes

        def run_a2a_question(question, schema):
            started = time.perf_counter()
            budget = BudgetContext(max_tokens=token_budget_default, max_latency_ms=latency_budget_default)
            spans = []
            answer = _execute_qa_sync(holder.get(), question, schema, budget, spans, protocol_execution=True)
            usage = token_usage_of(answer)
            if budget.spent_tokens == 0 and (usage["prompt_tokens"] or usage["completion_tokens"]):
                budget.record_usage(usage["prompt_tokens"], usage["completion_tokens"])
            telemetry_mgr.record_request(
                trace_id=current_trace_id() or "",
                question=question,
                schema=schema,
                answer_dict=answer,
                spans=spans,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
                cache_hit=False,
                degraded_level=budget.degradation_level,
            )
            return answer

        a2a_holder = _ProtocolHolder(a2a_store_provider or (lambda: A2ATaskStore(run_question=run_a2a_question)))
        mount_a2a_routes(
            app, a2a_holder.get, auth=protocol_auth,
            public_url=os.getenv("FINARAG_A2A_PUBLIC_URL", "http://127.0.0.1:8300"),
        )

    return app


app = create_fastapi_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="FinaRAG Async FastAPI Service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    workers = max(1, args.workers)
    if _protocol_enabled("FINARAG_A2A_ENABLED") and workers != 1:
        parser.error("A2A tasks require one service worker")
    uvicorn.run(
        "src.service.fastapi_app:app",
        host=args.host,
        port=args.port,
        workers=workers,
    )


if __name__ == "__main__":
    main()
