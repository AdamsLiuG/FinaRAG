"""Async LLM streaming client, incremental SSE bridge, and CPU process pool isolation."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, aclosing
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from src.graph.builder import stream_finarag_events
from src.harness.trajectory_store import capture_active, capture_step, harness_enabled, record_trajectory
from src.service.telemetry import TelemetrySpan, current_trace_id, inject_trace_headers

_CPU_POOL_LOCK = threading.Lock()
_CPU_POOL: Optional[ProcessPoolExecutor] = None


def _cpu_worker_init() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def get_cpu_executor(max_workers: int = 2) -> ProcessPoolExecutor:
    global _CPU_POOL
    if _CPU_POOL is not None:
        return _CPU_POOL
    with _CPU_POOL_LOCK:
        if _CPU_POOL is None:
            _CPU_POOL = ProcessPoolExecutor(
                max_workers=max(1, int(max_workers)),
                initializer=_cpu_worker_init,
            )
    return _CPU_POOL


def shutdown_cpu_executor() -> None:
    global _CPU_POOL
    with _CPU_POOL_LOCK:
        if _CPU_POOL is not None:
            _CPU_POOL.shutdown(wait=False, cancel_futures=True)
            _CPU_POOL = None


async def run_cpu_bound(fn: Callable[..., Any], *args: Any) -> Any:
    """Offload heavy CPU parsing into a bounded ProcessPoolExecutor without blocking event loop."""
    loop = asyncio.get_running_loop()
    pool = get_cpu_executor()
    return await loop.run_in_executor(pool, fn, *args)


class AsyncLLMClient:
    """Pooled async HTTPX client supporting real token-by-token OpenAI/vLLM SSE streaming."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_connections: int = 100,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        raw_base = (base_url or os.getenv("QWEN_BASE_URL") or os.getenv("LLM_BASE_URL") or "").strip().rstrip("/")
        if raw_base:
            parsed = urlparse(raw_base)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("Invalid LLM base_url scheme or host")
        self.base_url = raw_base
        resolved_key = api_key if api_key is not None else (os.getenv("QWEN_API_KEY") or os.getenv("LLM_API_KEY") or "")
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if resolved_key:
            headers["Authorization"] = f"Bearer {resolved_key}"

        self._client = httpx.AsyncClient(
            headers=headers,
            limits=httpx.Limits(
                max_connections=max(1, int(max_connections)),
                max_keepalive_connections=min(20, max(1, int(max_connections))),
            ),
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    def _chat_endpoint(self) -> str:
        if not self.base_url:
            raise ValueError("LLM base_url is required for streaming requests")
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    async def stream_chat_completion(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream token deltas from OpenAI-compatible /chat/completions and report TTFT & TPS."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        started = time.perf_counter()
        first_token_ts: Optional[float] = None
        token_chunks = 0
        final_usage: Dict[str, int] = {}

        async with self._client.stream(
            "POST",
            self._chat_endpoint(),
            headers=inject_trace_headers(),
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if isinstance(chunk.get("usage"), dict):
                    final_usage = {
                        "prompt_tokens": int(chunk["usage"].get("prompt_tokens") or 0),
                        "completion_tokens": int(chunk["usage"].get("completion_tokens") or 0),
                    }

                choices = chunk.get("choices") or []
                if choices:
                    delta_obj = (choices[0].get("delta") or {}) if isinstance(choices[0], dict) else {}
                    content = delta_obj.get("content") or ""
                    if content:
                        now = time.perf_counter()
                        if first_token_ts is None:
                            first_token_ts = now
                        token_chunks += 1
                        ttft_ms = round((first_token_ts - started) * 1000.0, 2)
                        gen_sec = max(1e-4, now - first_token_ts)
                        tps = round(token_chunks / gen_sec, 2) if token_chunks > 1 else 0.0
                        yield {
                            "stage": "token_delta",
                            "delta": content,
                            "ttft_ms": ttft_ms,
                            "tps": tps,
                        }

        if final_usage:
            yield {"stage": "llm_usage", "usage": final_usage}

    async def aclose(self) -> None:
        await self._client.aclose()


def _is_full_graph_processor(processor: Any) -> bool:
    return hasattr(processor, "_resolve_question_route") and hasattr(processor, "_build_query_plan")


async def async_stream_qa_events(
    processor: Any,
    question: str,
    schema: str,
    *,
    budget_context: Any = None,
    is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    telemetry_manager: Any = None,
    cache: Any = None,
    poll_interval: float = 0.05,
) -> AsyncIterator[Dict[str, Any]]:
    """Yield real-time QA stage events over SSE with disconnect cascade cancellation."""
    options = dict(
        budget_context=budget_context, is_disconnected=is_disconnected,
        telemetry_manager=telemetry_manager, cache=cache, poll_interval=poll_interval,
    )
    if capture_active() or not harness_enabled():
        async with aclosing(_stream_qa_events(processor, question, schema, **options)) as events:
            async for event in events:
                yield event
        return

    # The iterator can be closed by a different task. Its observer stays in
    # the producer context and is never installed in the consumer context.
    ctx = contextvars.copy_context()
    stack = ExitStack()
    observer = ctx.run(
        stack.enter_context,
        record_trajectory(entry_point="qa_stream", request={"question": question, "schema": schema}),
    )
    terminal = {}
    try:
        async with aclosing(_stream_qa_events(
            processor, question, schema, **options, observer_context=ctx, terminal=terminal,
        )) as events:
            async for event in events:
                yield event
    except (asyncio.CancelledError, GeneratorExit):
        terminal["status"] = "disconnected"
        raise
    except TimeoutError:
        terminal["status"] = "timeout"
        raise
    except Exception:
        terminal["status"] = "failed"
        raise
    finally:
        if observer is not None:
            observer.finish(terminal.get("answer"), status=terminal.get("status"))
        ctx.run(stack.close)


async def _stream_qa_events(
    processor, question, schema, *, budget_context, is_disconnected, telemetry_manager, cache, poll_interval,
    observer_context=None, terminal=None,
):
    started = time.perf_counter()
    trace_id = current_trace_id()
    spans: List[Dict[str, Any]] = []

    if cache is not None:
        cached = cache.get(question, schema)
        if cached is not None:
            if terminal is not None:
                observer_context.run(
                    capture_step, "cache", operation="semantic_cache", step_key="cache",
                    inputs={"schema": schema}, outputs={"cache_hit": True, "coverage": "absent"},
                )
                terminal.update(answer=cached, status="cache_hit")
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            yield {"stage": "route", "cache_hit": True, "trace_id": trace_id}
            yield {
                "stage": "done",
                "cache_hit": True,
                "final_answer": cached.get("final_answer"),
                "confidence": cached.get("confidence"),
                "references": cached.get("references") or [],
                "validation_flags": cached.get("validation_flags") or [],
                "degraded_level": 0,
                "trace_id": trace_id,
                "latency_ms": latency_ms,
            }
            if telemetry_manager is not None:
                await asyncio.to_thread(
                    telemetry_manager.record_request,
                    trace_id=trace_id or "",
                    question=question,
                    schema=schema,
                    answer_dict=cached,
                    spans=spans,
                    latency_ms=latency_ms,
                    cache_hit=True,
                    degraded_level=0,
                )
            return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    cancel_event = threading.Event()
    setattr(processor, "_stream_cancel_event", cancel_event)

    def _emit_from_worker(kind: str, payload: Any) -> None:
        if not loop.is_closed():
            loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    def _producer_sync() -> None:
        try:
            with TelemetrySpan("qa_stream_pipeline", attributes={"schema": schema}, collector=spans):
                if hasattr(processor, "stream_question_events"):
                    for ev in processor.stream_question_events(
                        question,
                        schema,
                        budget_context=budget_context,
                        cancel_event=cancel_event,
                    ):
                        if cancel_event.is_set():
                            return
                        _emit_from_worker("event", ev)
                elif _is_full_graph_processor(processor):
                    for ev in stream_finarag_events(
                        processor,
                        question,
                        schema,
                        budget_context=budget_context,
                    ):
                        if cancel_event.is_set():
                            return
                        _emit_from_worker("event", ev)
                else:
                    _emit_from_worker("event", {"stage": "route", "route_mode": "service"})
                    if cancel_event.is_set():
                        return
                    try:
                        answer = processor.process_question(question, schema, budget_context=budget_context)
                    except TypeError:
                        answer = processor.process_question(question, schema)
                    if cancel_event.is_set():
                        return
                    _emit_from_worker(
                        "event",
                        {
                            "node": "end",
                            "stage": "result",
                            "final_answer": (answer or {}).get("final_answer"),
                            "confidence": (answer or {}).get("confidence"),
                            "validation_flags": (answer or {}).get("validation_flags") or [],
                            "answer_dict": answer or {},
                        },
                    )
        except Exception as exc:
            _emit_from_worker("error", exc)
        finally:
            _emit_from_worker("eof", None)

    ctx = observer_context.copy() if observer_context is not None else contextvars.copy_context()
    producer_task = asyncio.create_task(asyncio.to_thread(ctx.run, _producer_sync))
    terminal_answer: Optional[Dict[str, Any]] = None

    try:
        while True:
            if is_disconnected is not None and await is_disconnected():
                if terminal is not None:
                    terminal["status"] = "disconnected"
                cancel_event.set()
                producer_task.cancel()
                return

            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=poll_interval)
            except asyncio.TimeoutError:
                continue

            if kind == "eof":
                break
            if kind == "error":
                raise payload

            event = dict(payload) if isinstance(payload, dict) else {"stage": "info", "payload": payload}
            if trace_id and "trace_id" not in event:
                event["trace_id"] = trace_id

            if event.get("stage") == "result":
                answer_dict = dict(event.pop("answer_dict", None) or {})
                terminal_answer = answer_dict
                if terminal is not None:
                    terminal["answer"] = answer_dict
                degraded_level = budget_context.degradation_level if budget_context is not None else int(answer_dict.get("degraded_level") or 0)
                done_event = {
                    "stage": "done",
                    "trace_id": trace_id,
                    "final_answer": answer_dict.get("final_answer"),
                    "confidence": answer_dict.get("confidence"),
                    "references": answer_dict.get("references") or [],
                    "validation_flags": answer_dict.get("validation_flags") or [],
                    "degraded_level": degraded_level,
                    "budget_summary": budget_context.summary() if budget_context is not None else answer_dict.get("budget_summary"),
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                }
                yield done_event
            else:
                yield event

        if terminal_answer is not None:
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            degraded_level = budget_context.degradation_level if budget_context is not None else int(terminal_answer.get("degraded_level") or 0)
            if cache is not None and degraded_level < 3:
                cache.set(question, schema, terminal_answer)
            if telemetry_manager is not None:
                await asyncio.to_thread(
                    telemetry_manager.record_request,
                    trace_id=trace_id or "",
                    question=question,
                    schema=schema,
                    answer_dict=terminal_answer,
                    spans=spans,
                    latency_ms=latency_ms,
                    cache_hit=False,
                    degraded_level=degraded_level,
                )
    finally:
        cancel_event.set()
        if not producer_task.done():
            producer_task.cancel()
        try:
            await producer_task
        except (asyncio.CancelledError, Exception):
            pass
