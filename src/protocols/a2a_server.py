"""A2A 0.3 task bindings and bounded, process-local task lifecycle."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import ipaddress
import math
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import Field

from .mcp_server import bounded_body, rpc_error, rpc_result, valid_rpc_request
from .tool_registry import (
    DOC_ID_PATTERN,
    MAX_ARGUMENT_BYTES,
    StrictObject,
    _bounded_page,
    _excerpt,
    canonical_json,
    decode_json,
    valid_principal,
)

_TaskToken = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")]
_TERMINAL = frozenset({"completed", "failed", "canceled"})
_ERRORS = {
    "invalid_arguments": (422, -32602, "Invalid task arguments"),
    "forbidden": (403, -32004, "Task access denied"),
    "not_found": (404, -32001, "Task not found"),
    "conflict": (409, -32002, "Task cannot be canceled"),
    "busy": (429, -32000, "Task executor busy"),
    "unavailable": (503, -32603, "A2A unavailable"),
}


class _TaskError(Exception):
    def __init__(self, code):
        self.code = code


class _TaskInput(StrictObject):
    question: str = Field(min_length=1, max_length=4000)
    answer_schema: str = Field(alias="schema", min_length=1, max_length=32)
    context_id: _TaskToken | None = None


class _TextPart(StrictObject):
    kind: Literal["text"]
    text: str = Field(min_length=1, max_length=4000)


class _UserMessage(StrictObject):
    kind: Literal["message"] = "message"
    role: Literal["user"]
    messageId: _TaskToken
    contextId: _TaskToken | None = None
    parts: list[_TextPart] = Field(min_length=1, max_length=8)


class _SendConfiguration(StrictObject):
    acceptedOutputModes: list[Literal["text/plain", "application/json"]] | None = Field(
        default=None, min_length=1, max_length=2)
    blocking: Literal[False] = False
    historyLength: Literal[0] = 0


class _SendInput(StrictObject):
    message: _UserMessage
    configuration: _SendConfiguration | None = None


class _TaskIdInput(StrictObject):
    id: _TaskToken


@dataclass
class _Task:
    id: str
    context_id: str
    owner: str
    status: str = "submitted"
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    terminal_at: float | None = None
    running: bool = True
    artifact: dict | None = None

    def transition(self, status, artifact=None):
        self.status = status
        self.updated_at = datetime.now(UTC).isoformat()
        if status in _TERMINAL:
            self.terminal_at = time.monotonic()
            self.artifact = artifact

    def snapshot(self):
        result = {"id": self.id, "context_id": self.context_id, "status": self.status,
                  "updated_at": self.updated_at}
        if self.artifact is not None:
            result["artifacts"] = [copy.deepcopy(self.artifact)]
        if self.status == "failed":
            result["error"] = {"code": "internal_error", "message": "Question execution failed"}
        return result


def _artifact(task_id, answer):
    if (not isinstance(answer, dict) or answer.get("error")
            or "processing_error" in (answer.get("validation_flags") or ())
            or "final_answer" not in answer):
        raise ValueError("Invalid QA result")
    value = answer["final_answer"]
    if isinstance(value, str):
        value = _excerpt(value, 8000)
    try:
        text = value if isinstance(value, str) else canonical_json(value)
    except (TypeError, ValueError, RecursionError):
        raise ValueError("Invalid answer") from None
    if len(text) > 8000:
        value = text = "[Answer omitted due to size]"
    confidence = answer.get("confidence")
    if isinstance(confidence, str):
        confidence = confidence[:64]
    elif type(confidence) not in (int, float) or not 0 <= confidence <= 1:
        confidence = None
    references = []
    if isinstance(answer.get("references"), (list, tuple)):
        for entry in answer["references"][:32]:
            if not isinstance(entry, dict):
                continue
            doc_id, page = entry.get("doc_id") or entry.get("pdf_sha1"), entry.get("page")
            if isinstance(doc_id, str) and re.fullmatch(DOC_ID_PATTERN, doc_id) and _bounded_page(page):
                references.append({"doc_id": doc_id, "page": page})
    return {
        "artifactId": task_id,
        "name": "Financial answer",
        "parts": [
            {"kind": "text", "text": text.encode("utf-8", "replace").decode("utf-8")},
            {"kind": "data", "data": {"answer": value, "confidence": confidence, "references": references}},
        ],
    }


class A2ATaskStore:
    """Run a bounded number of blocking QA calls and publish owner-scoped snapshots."""

    def __init__(self, *, run_question, max_active=4, max_retained=256, ttl_s=3600):
        if not 1 <= max_active <= max_retained or not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError("Invalid task executor configuration")
        self._run_question = run_question
        self._max_active = max_active
        self._max_retained = max_retained
        self._ttl_s = ttl_s
        self._executor = ThreadPoolExecutor(max_workers=max_active, thread_name_prefix="finarag-a2a")
        self._lock = threading.Lock()
        self._tasks: dict[str, _Task] = {}
        self._active = 0
        self._closed = False

    def _prune(self):
        now = time.monotonic()
        for task_id, task in list(self._tasks.items()):
            if not task.running and task.terminal_at is not None and now - task.terminal_at >= self._ttl_s:
                del self._tasks[task_id]

    async def submit(self, question, schema, *, principal, context_id=None):
        from src.service.fastapi_app import QaRequest

        if not valid_principal(principal):
            raise _TaskError("forbidden")
        try:
            args = _TaskInput(question=question, schema=schema, context_id=context_id)
            qa = QaRequest(question=args.question, schema=args.answer_schema)
        except ValueError:
            raise _TaskError("invalid_arguments") from None

        with self._lock:
            if self._closed:
                raise _TaskError("unavailable")
            self._prune()
            # Context IDs may only refer to an existing context for this owner.
            if context_id is not None and not any(
                    item.context_id == context_id and item.owner == principal.subject
                    for item in self._tasks.values()):
                raise _TaskError("not_found")
            if self._active >= self._max_active:
                raise _TaskError("busy")
            if len(self._tasks) >= self._max_retained:
                oldest = min((task for task in self._tasks.values() if not task.running),
                             key=lambda task: task.terminal_at, default=None)
                if oldest is None:
                    raise _TaskError("busy")
                del self._tasks[oldest.id]
            task_id = str(uuid.uuid4())
            task = _Task(task_id, context_id or str(uuid.uuid4()), principal.subject)
            self._tasks[task_id] = task
            self._active += 1
            context = contextvars.copy_context()
            try:
                self._executor.submit(context.run, self._work, task, qa.question, qa.answer_schema)
            except RuntimeError:
                self._active -= 1
                del self._tasks[task_id]
                raise _TaskError("unavailable") from None
            return task.snapshot()

    def _work(self, task, question, schema):
        try:
            with self._lock:
                if task.status != "submitted":
                    return
                task.transition("working")
            try:
                result = _artifact(task.id, self._run_question(question, schema))
                status = "completed"
            except Exception:
                status, result = "failed", None
            with self._lock:
                if task.status == "working":
                    task.transition(status, result)
        finally:
            with self._lock:
                task.running = False
                self._active -= 1

    def _owned(self, task_id, principal):
        if not valid_principal(principal):
            raise _TaskError("forbidden")
        self._prune()
        task = self._tasks.get(task_id) if isinstance(task_id, str) and len(task_id) <= 128 else None
        if task is None or task.owner != principal.subject:
            raise _TaskError("not_found")
        return task

    async def get(self, task_id, *, principal):
        with self._lock:
            return self._owned(task_id, principal).snapshot()

    async def cancel(self, task_id, *, principal):
        with self._lock:
            task = self._owned(task_id, principal)
            if task.status in {"completed", "failed"}:
                raise _TaskError("conflict")
            if task.status in {"submitted", "working"}:
                task.transition("canceled")
            return task.snapshot()

    async def aclose(self):
        with self._lock:
            self._closed = True
            for task in self._tasks.values():
                if task.status in {"submitted", "working"}:
                    task.transition("canceled")
        await asyncio.to_thread(self._executor.shutdown, wait=True)
        with self._lock:
            self._tasks.clear()


def _a2a_task(snapshot):
    status = {"state": snapshot["status"], "timestamp": snapshot["updated_at"]}
    if snapshot["status"] == "failed":
        status["message"] = {
            "kind": "message", "role": "agent", "messageId": snapshot["id"],
            "parts": [{"kind": "text", "text": snapshot["error"]["message"]}],
        }
    result = {"kind": "task", "id": snapshot["id"], "contextId": snapshot["context_id"], "status": status}
    if "artifacts" in snapshot:
        result["artifacts"] = snapshot["artifacts"]
    return result


def _agent_card(public_url, version):
    if len(public_url) > 2048 or any(char.isspace() or ord(char) < 32 or char == "\\" for char in public_url):
        raise ValueError("Invalid public URL")
    parts = urlsplit(public_url)
    port = parts.port
    if (not parts.hostname or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment or port == 0 or parts.scheme not in {"http", "https"}):
        raise ValueError("Invalid public URL")
    if parts.scheme == "http" and parts.hostname != "localhost":
        if not ipaddress.ip_address(parts.hostname).is_loopback:
            raise ValueError("Public URL requires HTTPS")
    return {
        "protocolVersion": "0.3.0",
        "name": "FinaRAG",
        "description": "Financial report question answering and evidence analysis",
        "version": version,
        "url": public_url.rstrip("/") + "/a2a",
        "preferredTransport": "JSONRPC",
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": False},
        "securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}},
        "security": [{"bearer": []}],
        "skills": [
            {"id": key, "name": name, "description": description, "tags": tags}
            for key, name, description, tags in (
                ("financial-qa", "Financial QA", "Answer questions about published reports", ["finance", "qa"]),
                ("report-navigation", "Report navigation", "Find report sections and pages", ["reports", "navigation"]),
                ("numeric-analysis", "Numeric analysis", "Compare and calculate financial values", ["finance", "numeric"]),
                ("evidence-verification", "Evidence verification", "Check claims against report evidence", ["evidence"]),
            )
        ],
    }


def mount_a2a_routes(app, store_provider, *, auth, public_url):
    try:
        card = _agent_card(public_url, app.version)
    except ValueError:
        card = None

    def available():
        if card is None:
            raise HTTPException(status_code=503, detail="A2A unavailable")

    async def store():
        available()
        try:
            return await asyncio.to_thread(store_provider)
        except Exception:
            raise HTTPException(status_code=503, detail="A2A unavailable") from None

    async def read_json(request):
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(status_code=415, detail="Expected application/json")
        return decode_json(await bounded_body(request, MAX_ARGUMENT_BYTES))

    def rest_error(exc):
        if isinstance(exc, _TaskError):
            http_status, _, message = _ERRORS[exc.code]
        else:
            http_status, message = 422, "Invalid task arguments"
        return HTTPException(status_code=http_status, detail=message)

    @app.get("/.well-known/agent.json")
    @app.get("/.well-known/agent-card.json")
    async def agent_card():
        available()
        return card

    @app.post("/a2a/tasks", status_code=202)
    async def submit(request: Request):
        principal = await auth(request)
        try:
            args = _TaskInput.model_validate(await read_json(request), strict=True)
            task = await (await store()).submit(args.question, args.answer_schema, principal=principal,
                                                context_id=args.context_id)
        except (ValueError, RecursionError, _TaskError) as exc:
            raise rest_error(exc) from None
        return Response(canonical_json(task), status_code=202, media_type="application/json",
                        headers={"Location": f"/a2a/tasks/{task['id']}"})

    @app.get("/a2a/tasks/{task_id}")
    async def get(task_id: str, request: Request):
        principal = await auth(request)
        try:
            task = await (await store()).get(task_id, principal=principal)
        except _TaskError as exc:
            raise rest_error(exc) from None
        return Response(canonical_json(task), media_type="application/json")

    @app.post("/a2a/tasks/{task_id}/cancel")
    async def cancel(task_id: str, request: Request):
        principal = await auth(request)
        try:
            task = await (await store()).cancel(task_id, principal=principal)
        except _TaskError as exc:
            raise rest_error(exc) from None
        return Response(canonical_json(task), media_type="application/json")

    @app.post("/a2a")
    async def rpc(request: Request):
        principal = await auth(request)
        try:
            message = await read_json(request)
        except (ValueError, RecursionError):
            return JSONResponse(rpc_error(-32700), status_code=400)
        if not valid_rpc_request(message) or "id" not in message:
            return JSONResponse(rpc_error(-32600))
        request_id, method = message["id"], message["method"]
        params = message.get("params")
        try:
            if method == "message/send":
                args = _SendInput.model_validate(params, strict=True)
                question = "\n".join(part.text for part in args.message.parts)
                task = await (await store()).submit(question, "string", principal=principal,
                                                    context_id=args.message.contextId)
            elif method in {"tasks/get", "tasks/cancel"}:
                args = _TaskIdInput.model_validate(params, strict=True)
                task_store = await store()
                action = task_store.get if method == "tasks/get" else task_store.cancel
                task = await action(args.id, principal=principal)
            else:
                return Response(canonical_json(rpc_error(-32601, request_id)), media_type="application/json")
            reply = rpc_result(request_id, _a2a_task(task))
        except ValueError:
            reply = rpc_error(-32602, request_id)
        except _TaskError as exc:
            _, code, error = _ERRORS[exc.code]
            reply = rpc_error(code, request_id, message=error)
        return Response(canonical_json(reply), media_type="application/json")
