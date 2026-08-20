"""D3 服务层测试（stdlib 实现，起真实 ThreadingHTTPServer 于随机端口）+ D1 并发隔离。"""
import json
import threading
import unittest
import urllib.request
from unittest.mock import patch

from src.service.app import QaService, make_server
from src.service.tracing import StageTimer, token_usage_of


class _FakeProcessor:
    """记录调用与线程信息；按 question 回显（并发隔离断言用）。"""

    def __init__(self, usage=None):
        self.calls = []
        self.lock = threading.Lock()
        self.usage = usage or {"prompt_tokens": 100, "completion_tokens": 20}

    def process_question(self, question, schema, question_meta=None):
        with self.lock:
            self.calls.append((question, schema, threading.current_thread().ident))
        return {
            "final_answer": f"answer:{question}",
            "confidence": "high",
            "references": [{"pdf_sha1": "doc", "page_index": 1}],
            "validation_flags": [],
            "response_data": {"usage": self.usage},
        }


def _post(port, path, payload):
    import urllib.error

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as response:
        return response.status, response.read().decode("utf-8")


class _ServerFixture:
    def __init__(self, service):
        self.server = make_server(host="127.0.0.1", port=0, service=service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class TestQaServiceHTTP(unittest.TestCase):
    def setUp(self):
        self.fake = _FakeProcessor()
        self.fixture = _ServerFixture(QaService(processor=self.fake))
        self.addCleanup(self.fixture.close)

    def test_t1_qa_smoke(self):
        status, body = _post(self.fixture.port, "/qa", {"question": "归母净利润是多少", "schema": "string"})
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["final_answer"], "answer:归母净利润是多少")
        self.assertEqual(payload["confidence"], "high")
        self.assertFalse(payload["budget_exceeded"])
        self.assertIn("process_question", payload["trace"]["stages_ms"])
        self.assertEqual(payload["token_usage"]["prompt_tokens"], 100)

    def test_t2_sse_event_sequence(self):
        status, body = _post(self.fixture.port, "/qa/stream", {"question": "营收"})
        self.assertEqual(status, 200)
        events = [json.loads(line[6:]) for line in body.split("\n\n") if line.startswith("data: ")]
        self.assertEqual([e["stage"] for e in events],
                         ["route", "retrieve", "rerank", "ground", "generate", "validate", "done"])
        self.assertEqual(events[-1]["final_answer"], "answer:营收")
        self.assertFalse(events[-1]["budget_exceeded"])

    def test_t3_healthz(self):
        status, body = _get(self.fixture.port, "/healthz")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["knowledge_generation"]["valid"])
        self.assertEqual(payload["knowledge_generation"]["schema_version"], "1.2")
        self.assertIn("index_cache", payload)
        self.assertFalse(payload["adaptive_enabled"])

    def test_t4_healthz_degraded_when_knowledge_invalid(self):
        """知识层源失步 → fail-closed 透出 degraded（patch load 抛错）。"""
        with patch("src.metric_atoms.MetricAtomStore.load", side_effect=RuntimeError("源已变更")):
            service = QaService(processor=self.fake)
            health = service.healthz()
        self.assertEqual(health["status"], "degraded")
        self.assertFalse(health["knowledge_generation"]["valid"])
        self.assertIn("源已变更", health["knowledge_generation"]["detail"])

    def test_t5_concurrent_isolation(self):
        """并发隔离：两线程并发请求，各自响应 question 不串（D1 验收）。"""
        questions = ["问题甲", "问题乙"]
        results = {}

        def _ask(question):
            status, body = _post(self.fixture.port, "/qa", {"question": question})
            results[question] = json.loads(body)

        threads = [threading.Thread(target=_ask, args=(q,)) for q in questions]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for question in questions:
            self.assertEqual(results[question]["final_answer"], f"answer:{question}")

    def test_t6_budget_exceeded_flag(self):
        """预算超限标记（v1：标记不做逐级降级——降级链属 v2）。"""
        heavy = _FakeProcessor(usage={"prompt_tokens": 90000, "completion_tokens": 1000})
        fixture = _ServerFixture(QaService(processor=heavy))
        self.addCleanup(fixture.close)
        status, body = _post(fixture.port, "/qa", {"question": "超预算问题"})
        payload = json.loads(body)
        self.assertTrue(payload["budget_exceeded"])
        self.assertEqual(payload["token_usage"]["prompt_tokens"], 90000)

    def test_t7_input_validation(self):
        status, body = _post(self.fixture.port, "/qa", {"schema": "string"})
        self.assertEqual(status, 400)
        self.assertIn("question", json.loads(body)["error"])

        status, body = _post(self.fixture.port, "/qa", {"question": "q", "schema": "yaml"})
        self.assertEqual(status, 400)

    def test_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _get(self.fixture.port, "/nope")
        self.assertEqual(ctx.exception.code, 404)


class TestTracing(unittest.TestCase):
    def test_stage_timer_records_ms(self):
        trace = {}
        with StageTimer(trace, "stage_a"):
            pass
        self.assertIn("stage_a", trace["stages_ms"])
        self.assertIsInstance(trace["stages_ms"]["stage_a"], float)

    def test_token_usage_extraction(self):
        usage = token_usage_of({"response_data": {"usage": {"prompt_tokens": 5, "completion_tokens": 2}}})
        self.assertEqual(usage, {"prompt_tokens": 5, "completion_tokens": 2})
        self.assertEqual(token_usage_of({}), {"prompt_tokens": 0, "completion_tokens": 0})


class TestD1Concurrency(unittest.TestCase):
    def test_document_cache_concurrent_read_write(self):
        """D1：并发读写 _document_cache 无异常且结果一致（锁修复验收）。"""
        import tempfile
        from pathlib import Path

        from src.questions_processing import QuestionsProcessor

        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            (docs / "doc1.json").write_text(
                json.dumps({"metainfo": {"sha1_name": "doc1"}, "pages": []}), encoding="utf-8"
            )
            processor = object.__new__(QuestionsProcessor)
            processor.documents_dir = docs
            processor._document_cache = {}
            processor._document_cache_lock = threading.Lock()

            payloads = []
            errors = []

            def _load():
                try:
                    payloads.append(processor._load_document_payload("doc1"))
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=_load) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertFalse(errors)
            self.assertEqual(len(payloads), 8)
            self.assertTrue(all(p is payloads[0] for p in payloads))

    def test_retriever_cache_is_thread_local(self):
        """D1 实测修正：_retriever_cache 是 threading.local——retriever/reranker
        每线程一套，reranker.last_debug 无跨线程共享残留（v2 方案该评估不成立）。"""
        from src.questions_processing import QuestionsProcessor

        processor = object.__new__(QuestionsProcessor)
        processor._retriever_cache = threading.local()
        processor.retriever_cache_enabled = True
        processor._retriever_cache.bundle = "thread-a-bundle"

        seen = {}

        def _read():
            seen["bundle"] = getattr(processor._retriever_cache, "bundle", None)

        thread = threading.Thread(target=_read)
        thread.start()
        thread.join()
        self.assertIsNone(seen["bundle"])  # 线程 B 看不到线程 A 的 bundle
        self.assertEqual(processor._retriever_cache.bundle, "thread-a-bundle")


if __name__ == "__main__":
    unittest.main()
