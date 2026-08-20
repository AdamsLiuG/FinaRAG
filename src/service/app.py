"""D3 服务层：POST /qa、POST /qa/stream（SSE）、GET /healthz。

stdlib ThreadingHTTPServer 实现——离线环境装不上 fastapi（SSL 断连）的降级
路线；事件 schema 与 FastAPI 版设计对齐（docs/finarag_competitiveness_plan.md
§5.2），后续有网换框架时 handler 逻辑不变。

已知限制（如实声明）：
- /qa/stream 为伪流式：process_question 完成后按阶段顺序推送事件（真流式
  需要 pipeline 内部 per-stage 回调，属 v2）
- per-stage 计时 v1 只覆盖服务层可见的总耗时（process 内部埋点属 Agentic v2
  与 trace 体系配套批次）
- 预算控制 v1 为「超限标记」：SERVICE_TOKEN_BUDGET 超限时响应标
  budget_exceeded（逐级降级链——跳 HyDE/砍候选——需 process 配合，属 v2）
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from src.service.tracing import StageTimer, append_service_trace, token_usage_of

SERVICE_TOKEN_BUDGET = int(os.getenv("SERVICE_TOKEN_BUDGET", "8000"))
SSE_STAGES = ["route", "retrieve", "rerank", "ground", "generate", "validate", "done"]


class QaService:
    """QA 业务逻辑（与 HTTP 解耦，便于注入 mock processor 测试）。"""

    def __init__(self, processor=None):
        self._processor = processor
        self._processor_lock = threading.Lock()

    @property
    def processor(self):
        if self._processor is None:
            with self._processor_lock:
                if self._processor is None:
                    from src.questions_processing import QuestionsProcessor

                    self._processor = QuestionsProcessor()
        return self._processor

    def handle_qa(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /qa 主逻辑：校验 → process_question → trace → 响应组装。"""
        question = str(payload.get("question") or "").strip()
        if not question:
            raise ValueError("question is required")
        schema = str(payload.get("schema") or "string")
        if schema not in {"string", "number", "boolean", "json"}:
            raise ValueError(f"invalid schema: {schema}")

        trace: Dict[str, Any] = {"question": question, "schema": schema}
        with StageTimer(trace, "process_question"):
            answer_dict = self.processor.process_question(question, schema)

        usage = token_usage_of(answer_dict)
        total_tokens = usage["prompt_tokens"] + usage["completion_tokens"]
        budget_exceeded = total_tokens > SERVICE_TOKEN_BUDGET > 0

        trace.update({
            "final_answer": answer_dict.get("final_answer"),
            "confidence": answer_dict.get("confidence"),
            "validation_flags": answer_dict.get("validation_flags") or [],
            "token_usage": usage,
            "budget_exceeded": budget_exceeded,
        })
        append_service_trace(trace)

        return {
            "question": question,
            "final_answer": answer_dict.get("final_answer"),
            "confidence": answer_dict.get("confidence"),
            "references": answer_dict.get("references") or [],
            "validation_flags": answer_dict.get("validation_flags") or [],
            "token_usage": usage,
            "budget_exceeded": budget_exceeded,
            "trace": {
                "stages_ms": trace.get("stages_ms", {}),
                "token_usage": usage,
            },
        }

    def stream_events(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """POST /qa/stream 的 SSE 事件序列（伪流式：处理后按固定阶段序推送）。"""
        response = self.handle_qa(payload)
        answer_preview = str(response.get("final_answer") or "")[:80]
        return [
            {"stage": "route", "route_mode": "service"},
            {"stage": "retrieve", "n_references": len(response.get("references") or [])},
            {"stage": "rerank", "ms": response["trace"]["stages_ms"].get("process_question")},
            {"stage": "ground"},
            {"stage": "generate", "answer_preview": answer_preview},
            {
                "stage": "validate",
                "flags": response.get("validation_flags") or [],
                "confidence": response.get("confidence"),
            },
            {
                "stage": "done",
                "final_answer": response.get("final_answer"),
                "token_usage": response.get("token_usage"),
                "budget_exceeded": response.get("budget_exceeded"),
            },
        ]

    def healthz(self) -> Dict[str, Any]:
        """GET /healthz：知识层 Generation 校验（fail-closed 状态透出）+ 缓存统计。"""
        from src.adaptive_retrieval import adaptive_retrieval_enabled
        from src.retrieval import _INDEX_CACHE

        knowledge = {"valid": False, "detail": "unknown"}
        try:
            import os as _os
            from src.metric_atoms import MetricAtomStore

            MetricAtomStore.load()  # fail-closed 校验：源失步/缺失会 raise
            snapshot_path = _os.path.join(
                _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                "..", "config", "atom_snapshot.json",
            )
            with open(snapshot_path, encoding="utf-8") as f:
                snapshot = json.load(f)
            knowledge = {
                "valid": True,
                "schema_version": snapshot.get("schema_version"),
            }
        except Exception as exc:
            knowledge = {"valid": False, "detail": f"{type(exc).__name__}: {exc}"}

        return {
            "status": "ok" if knowledge["valid"] else "degraded",
            "knowledge_generation": knowledge,
            "index_cache": _INDEX_CACHE.stats(),
            "adaptive_enabled": adaptive_retrieval_enabled(),
        }


def _make_handler(service: QaService):
    class _QaRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # 静默访问日志（测试噪声控制）
            pass

        def _send_json(self, status: int, payload: Dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> Dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_POST(self):  # noqa: N802
            try:
                payload = self._read_body()
                if self.path == "/qa":
                    self._send_json(200, service.handle_qa(payload))
                    return
                if self.path == "/qa/stream":
                    events = service.stream_events(payload)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    for event in events:
                        data = json.dumps(event, ensure_ascii=False)
                        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    return
                self._send_json(404, {"error": f"not found: {self.path}"})
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON body"})
            except Exception as exc:
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self):  # noqa: N802
            if self.path == "/healthz":
                self._send_json(200, service.healthz())
                return
            self._send_json(404, {"error": f"not found: {self.path}"})

    return _QaRequestHandler


def make_server(host: str = "127.0.0.1", port: int = 8300, service: Optional[QaService] = None) -> ThreadingHTTPServer:
    """构建多线程 HTTP 服务（处理器单例共享，索引/模型常驻）。"""
    service = service or QaService()
    return ThreadingHTTPServer((host, port), _make_handler(service))


def main():
    parser = argparse.ArgumentParser(description="FinaRAG QA 服务（stdlib 实现）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()
    server = make_server(args.host, args.port)
    print(f"FinaRAG QA service listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
