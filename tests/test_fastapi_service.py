"""Async tests for P1-2 FastAPI service, true SSE streaming, budget degradation, cache, and security."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from src.prompts import PROMPT_TEMPLATE_VERSION, build_prefix_cached_messages
from src.questions_processing import QuestionsProcessor
from src.service.budget_controller import BudgetContext
from src.service.fastapi_app import create_fastapi_app
from src.service.semantic_cache import CacheVersions, SemanticCache
from src.service.streaming import AsyncLLMClient, async_stream_qa_events
from src.service.telemetry import TelemetryManager


class _ConcurrentFakeProcessor:
    def __init__(self, delay: float = 0.01):
        self.delay = delay
        self.lock = threading.Lock()
        self.calls = []

    def process_question(self, question: str, schema: str, budget_context=None):
        time.sleep(self.delay)
        with self.lock:
            self.calls.append((question, schema))
        return {
            "final_answer": f"ans:{question}",
            "confidence": "high",
            "references": [{"pdf_sha1": "doc1", "page_index": 5}],
            "validation_flags": [],
            "response_data": {"usage": {"prompt_tokens": 120, "completion_tokens": 30}},
        }


class _SlowStreamProcessor:
    def stream_question_events(self, question: str, schema: str, budget_context=None, cancel_event=None):
        yield {"stage": "route", "route_mode": "single_rag"}
        # Incremental 0.25s delay so client sees event 1 immediately before result finishes
        for _ in range(25):
            if cancel_event is not None and cancel_event.is_set():
                return
            time.sleep(0.01)
        yield {
            "stage": "result",
            "final_answer": f"stream:{question}",
            "confidence": "high",
            "validation_flags": [],
            "answer_dict": {
                "final_answer": f"stream:{question}",
                "confidence": "high",
                "references": [],
                "validation_flags": [],
            },
        }


class _FailingSecretProcessor:
    def process_question(self, question: str, schema: str, budget_context=None):
        raise RuntimeError("SECRET_DB_TOKEN=abc123xyz internal connection error")


class FastAPIServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.trace_dir = self.root / "traces"

    def tearDown(self):
        self.tmp_dir.cleanup()

    async def test_healthz_security_headers_auth_and_sanitized_500(self):
        cache = SemanticCache()
        telemetry = TelemetryManager(trace_dir=self.trace_dir)
        app = create_fastapi_app(
            processor=_ConcurrentFakeProcessor(delay=0.0),
            cache=cache,
            telemetry=telemetry,
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            health = await client.get("/healthz")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "ok")
            self.assertEqual(health.headers["X-Content-Type-Options"], "nosniff")
            self.assertEqual(health.headers["X-Frame-Options"], "DENY")
            self.assertIn("default-src 'none'", health.headers["Content-Security-Policy"])
            self.assertTrue(health.headers["X-Trace-Id"])

            # Input validation -> 422 on blank or invalid schema
            bad_blank = await client.post("/qa", json={"question": "   ", "schema": "string"})
            self.assertEqual(bad_blank.status_code, 422)
            bad_schema = await client.post("/qa", json={"question": "营收", "schema": "yaml"})
            self.assertEqual(bad_schema.status_code, 422)

            # API key guard when SERVICE_API_KEY is configured
            with patch.dict(os.environ, {"SERVICE_API_KEY": "prod-secret-token"}):
                unauth = await client.get("/metrics/cache")
                self.assertEqual(unauth.status_code, 401)
                wrong = await client.post(
                    "/qa",
                    headers={"Authorization": "Bearer wrong-token"},
                    json={"question": "营收"},
                )
                self.assertEqual(wrong.status_code, 401)
                ok = await client.get(
                    "/metrics/cache",
                    headers={"Authorization": "Bearer prod-secret-token"},
                )
                self.assertEqual(ok.status_code, 200)

        # Sanitized HTTP 500 never exposes secret exception strings
        err_app = create_fastapi_app(
            processor=_FailingSecretProcessor(),
            telemetry=TelemetryManager(trace_dir=self.trace_dir),
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=err_app), base_url="http://127.0.0.1") as client:
            res = await client.post("/qa", json={"question": "测试内部错误脱敏", "use_cache": False})
            self.assertEqual(res.status_code, 500)
            self.assertEqual(res.json()["error"], "Internal server error")
            self.assertNotIn("SECRET_DB_TOKEN", res.text)
            self.assertNotIn("abc123xyz", res.text)

    async def test_20_concurrent_requests_trace_isolation_and_zero_event_loop_blocking(self):
        cache = SemanticCache()
        telemetry = TelemetryManager(trace_dir=self.trace_dir)
        app = create_fastapi_app(
            processor=_ConcurrentFakeProcessor(delay=0.25),
            cache=cache,
            telemetry=telemetry,
        )

        lags: list[float] = []
        stop_monitor = asyncio.Event()

        async def _lag_monitor():
            while not stop_monitor.is_set():
                t0 = time.perf_counter()
                await asyncio.sleep(0.01)
                lags.append(max(0.0, (time.perf_counter() - t0 - 0.01) * 1000.0))

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
            # Pre-warm ASGI router & worker threadpool before measuring loop lag
            await asyncio.gather(
                *[client.post("/qa", json={"question": f"warm_{k}", "use_cache": False}) for k in range(4)]
            )
            monitor_task = asyncio.create_task(_lag_monitor())
            await asyncio.sleep(0)
            try:
                async def _send(i: int):
                    q = f"并发隔离测试问题_{i}"
                    res = await client.post("/qa", json={"question": q, "use_cache": False})
                    return i, q, res

                responses = await asyncio.gather(*[_send(i) for i in range(20)])
            finally:
                stop_monitor.set()
                await monitor_task

        trace_ids = set()
        for i, q, res in responses:
            self.assertEqual(res.status_code, 200)
            body = res.json()
            self.assertEqual(body["final_answer"], f"ans:{q}")
            self.assertEqual(body["trace_id"], res.headers["X-Trace-Id"])
            trace_ids.add(body["trace_id"])
        self.assertEqual(len(trace_ids), 20)
        sorted_lags = sorted(lags)
        p95_lag_ms = sorted_lags[int((len(sorted_lags) - 1) * 0.95)]
        self.assertLess(p95_lag_ms, 20.0)

    async def test_true_incremental_sse_and_disconnect_cascade_cancellation(self):
        proc = _SlowStreamProcessor()
        telemetry = TelemetryManager(trace_dir=self.trace_dir)
        cache = SemanticCache()
        app = create_fastapi_app(processor=proc, cache=cache, telemetry=telemetry)

        # Verify true incremental timing on async_stream_qa_events directly:
        # event 0 arrives immediately (< 0.1s) BEFORE the 0.25s tail stage finishes
        direct_events = []
        direct_times = []
        t0_direct = time.perf_counter()
        async for ev in async_stream_qa_events(proc, "增量计时测试", "string", poll_interval=0.01):
            direct_events.append(ev)
            direct_times.append(time.perf_counter() - t0_direct)
        self.assertEqual([e["stage"] for e in direct_events], ["route", "done"])
        self.assertLess(direct_times[0], 0.10)
        self.assertGreaterEqual(direct_times[-1], 0.20)

        # Verify FastAPI SSE wire endpoint /qa/stream
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
            async with client.stream("POST", "/qa/stream", json={"question": "流式营收测试", "use_cache": False}) as resp:
                self.assertEqual(resp.status_code, 200)
                events = []
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    events.append(json.loads(line[6:]))

        self.assertEqual([e["stage"] for e in events], ["route", "done"])
        self.assertEqual(events[-1]["final_answer"], "stream:流式营收测试")

        # Client disconnect cancellation stops worker cleanly in < 0.3s
        disconnected_flag = False

        async def _is_disconnected() -> bool:
            return disconnected_flag

        proc2 = _SlowStreamProcessor()
        t0 = time.perf_counter()
        seen = []
        async for ev in async_stream_qa_events(
            proc2,
            "断开取消测试",
            "string",
            is_disconnected=_is_disconnected,
            poll_interval=0.02,
        ):
            seen.append(ev)
            disconnected_flag = True
        cancel_elapsed = time.perf_counter() - t0
        self.assertEqual(len(seen), 1)
        self.assertTrue(proc2._stream_cancel_event.is_set())
        self.assertLess(cancel_elapsed, 0.2)

    async def test_async_llm_client_stream_ttft_tps_and_trace_header(self):
        captured_headers = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            sse_body = (
                'data: {"choices":[{"delta":{"content":"营业"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"收入增长"}}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":12}}\n\n'
                "data: [DONE]\n\n"
            ).encode("utf-8")
            return httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})

        client = AsyncLLMClient(
            base_url="http://127.0.0.1:9999/v1",
            api_key="test-local",
            transport=httpx.MockTransport(handler),
        )
        self.addAsyncCleanup(client.aclose)

        from src.service.telemetry import bind_trace_context, reset_trace_context

        trace_id, _, tokens = bind_trace_context(None)
        try:
            chunks = [
                item
                async for item in client.stream_chat_completion(
                    messages=[{"role": "user", "content": "营收?"}],
                    model="Qwen3.5-35B",
                )
            ]
        finally:
            reset_trace_context(tokens)

        deltas = [c for c in chunks if c["stage"] == "token_delta"]
        usages = [c for c in chunks if c["stage"] == "llm_usage"]
        self.assertEqual("".join(d["delta"] for d in deltas), "营业收入增长")
        self.assertGreaterEqual(deltas[0]["ttft_ms"], 0.0)
        self.assertEqual(usages[0]["usage"]["completion_tokens"], 12)
        self.assertIn(trace_id, captured_headers.get("traceparent", ""))

    def test_runtime_4_level_budget_degradation_and_langgraph_wiring(self):
        # Level progression unit verification
        ctx = BudgetContext(max_tokens=1000, max_latency_ms=10000.0)
        self.assertEqual(ctx.degradation_level, 0)
        self.assertTrue(ctx.allow_hyde)
        self.assertTrue(ctx.allow_adaptive_supplement)

        ctx.record_usage(prompt_tokens=600, completion_tokens=0)  # remaining 40% -> Level 1
        self.assertEqual(ctx.degradation_level, 1)
        self.assertFalse(ctx.allow_hyde)
        self.assertFalse(ctx.allow_adaptive_supplement)

        ctx.record_usage(prompt_tokens=200, completion_tokens=0)  # remaining 20% -> Level 2
        self.assertEqual(ctx.degradation_level, 2)
        self.assertEqual(ctx.effective_top_n(6), 3)
        trimmed = ctx.compress_results([{"text": "甲" * 1200, "chunk_type": "content"}], max_chars=200)
        self.assertLess(len(trimmed[0]["text"]), 250)

        ctx.record_usage(prompt_tokens=250, completion_tokens=0)  # remaining 0% -> Level 3
        self.assertEqual(ctx.degradation_level, 3)
        self.assertTrue(ctx.skip_second_stage_rerank)
        self.assertEqual(ctx.effective_top_n(6), 2)

        # LangGraph end-to-end Level 2 & Level 3 degradation verification
        docs_dir = self.root / "documents"
        docs_dir.mkdir(parents=True, exist_ok=True)
        subset_path = self.root / "manifest.csv"
        subset_path.write_text(
            "doc_id,company_name,company_aliases,security_code,doc_source_type,report_title,language,currency,report_year\n"
            "doc-1,浦发银行,浦发银行|600000,600000,annual_report,浦发银行2024年年报,zh,CNY,2024\n",
            encoding="utf-8",
        )
        doc_payload = {
            "metainfo": {"doc_id": "doc-1", "sha1_name": "doc-1", "company_name": "浦发银行", "report_year": 2024, "currency": "CNY"},
            "content": {"pages": [{"page": 19, "text": "2024年营业收入170748百万元"}]},
        }
        (docs_dir / "doc-1.json").write_text(json.dumps(doc_payload, ensure_ascii=False), encoding="utf-8")

        processor = QuestionsProcessor(documents_dir=docs_dir, subset_path=subset_path, parallel_requests=1, top_n_retrieval=6)

        class _FakeRet:
            def __init__(self):
                self.rerank_calls = 0

            def retrieve_by_company_name(self, **kwargs):
                return [
                    {
                        "page": 19 + i,
                        "text": "详细经营分析" * 300,
                        "pdf_sha1": "doc-1",
                        "combined_score": 0.9 - i * 0.05,
                        "distance": 0.9 - i * 0.05,
                        "ranking_score": 0.9 - i * 0.05,
                        "metadata": {"sha1_name": "doc-1", "company_name": "浦发银行", "report_year": 2024, "chunk_id": i, "node_type": "child"},
                        "retrieval_sources": ["vector"],
                    }
                    for i in range(6)
                ]

            def retrieve_candidates_by_company_name(self, **kwargs):
                return self.retrieve_by_company_name(**kwargs)

            def rerank_candidates(self, **kwargs):
                self.rerank_calls += 1
                return list(kwargs.get("candidate_results") or [])

        fake_ret = _FakeRet()
        processor._build_retriever = lambda: (fake_ret, "vector")
        captured_prompts = []
        processor.api_processor.get_answer_from_rag_context = lambda **kwargs: (
            captured_prompts.append(kwargs["rag_context"]),
            {
                "final_answer": "经营稳健",
                "relevant_pages": [19],
                "reasoning_summary": "依据第19页。",
                "step_by_step_analysis": "依据第19页。",
            },
        )[1]

        from src.graph.builder import run_finarag_graph

        # Level 2 caps retrieved passages passed to LLM to <= 3 and truncates long blocks
        lvl2 = BudgetContext(max_tokens=1000, spent_tokens=800)
        res2 = run_finarag_graph(processor, "浦发银行2024年经营策略如何？", "text", budget_context=lvl2)
        self.assertEqual(res2["degraded_level"], 2)
        self.assertLessEqual(captured_prompts[-1].count("第"), 4)

        # Level 3 skips rerank even if mode was hybrid_rerank
        processor._build_retriever = lambda: (fake_ret, "hybrid_rerank")
        lvl3 = BudgetContext(max_tokens=1000, spent_tokens=1000)
        res3 = run_finarag_graph(processor, "浦发银行2024年经营策略如何？", "text", budget_context=lvl3)
        self.assertEqual(res3["degraded_level"], 3)
        self.assertEqual(res3["final_reranking_backend"], "skipped_budget")

    def test_semantic_cache_version_invalidation_telemetry_and_prefix_prompt(self):
        v1 = CacheVersions(corpus_version="c1", atom_schema_version="a1", prompt_version=PROMPT_TEMPLATE_VERSION)
        v2 = CacheVersions(corpus_version="c2", atom_schema_version="a1", prompt_version=PROMPT_TEMPLATE_VERSION)
        cache = SemanticCache(max_entries=8, versions=v1)
        payload = {"final_answer": "1707.48亿元", "confidence": "high"}
        cache.set("浦发银行2024年营业收入是多少？", "number", payload)

        hit = cache.get("浦发银行2024年营业收入是多少？", "number")
        self.assertIsNotNone(hit)
        hit["final_answer"] = "MUTATED"
        self.assertEqual(cache.get("浦发银行2024年营业收入是多少？", "number")["final_answer"], "1707.48亿元")

        # Changing corpus version MUST invalidate cache entry
        cache.update_versions(v2)
        self.assertIsNone(cache.get("浦发银行2024年营业收入是多少？", "number"))

        # Telemetry + adaptive_trace signal routing + optional Langfuse export
        langfuse_calls = []

        def lf_handler(req: httpx.Request) -> httpx.Response:
            langfuse_calls.append(json.loads(req.content.decode("utf-8")))
            return httpx.Response(200, json={"status": "ok"})

        tm = TelemetryManager(
            trace_dir=self.trace_dir,
            langfuse_host="http://127.0.0.1:3000",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            http_transport=httpx.MockTransport(lf_handler),
        )
        tm.record_request(
            trace_id="a" * 32,
            question="扣非净利润是多少？",
            schema="number",
            answer_dict={
                "final_answer": "N/A",
                "confidence": "low",
                "validation_flags": ["metric_caliber_conflict_refused"],
                "response_data": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            },
            spans=[],
            latency_ms=42.5,
        )
        self.assertTrue((self.trace_dir / "service_trace.jsonl").exists())
        self.assertTrue((self.trace_dir / "adaptive_trace.jsonl").exists())
        self.assertEqual(len(langfuse_calls), 1)

        # Prefix caching message ordering check
        sys_msg, usr_msg = build_prefix_cached_messages(
            system_prompt="[STATIC_INSTRUCTIONS]",
            user_prompt_template="CTX:{context}|Q:{question}",
            rag_context="PAGE_19_DATA",
            question="REVENUE?",
            metric_caliber_hint="[METRIC_HINT]",
        )
        self.assertTrue(sys_msg.startswith("[STATIC_INSTRUCTIONS]"))
        self.assertIn("[METRIC_HINT]", sys_msg)
        self.assertLess(usr_msg.index("PAGE_19_DATA"), usr_msg.index("REVENUE?"))


if __name__ == "__main__":
    unittest.main()
