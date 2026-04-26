import unittest

from pydantic import BaseModel

from eval.ragas_adapter import (
    OpenAIStreamingInstructorLLM,
    RagasRuntime,
    RagasRuntimeConfig,
    collect_ragas_contexts,
    prepare_ragas_runtime,
    score_with_ragas,
)


class FakeRagasRuntime:
    def score(self, *, question_text, answer, reference, contexts):
        self.last_call = {
            "question_text": question_text,
            "answer": answer,
            "reference": reference,
            "contexts": contexts,
        }
        return {
            "available": True,
            "reason": "ok",
            "error": None,
            "errors": [],
            "contexts_used": len(contexts),
            "answer_correctness": 0.9,
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "context_recall": 0.95,
            "context_precision": 0.85,
            "ragas_score": 0.8,
        }


class StrictScoreMetric:
    def __init__(self, allowed_keys, value):
        self.allowed_keys = set(allowed_keys)
        self.value = value
        self.last_kwargs = None

    def score(self, **kwargs):
        unexpected = set(kwargs) - self.allowed_keys
        if unexpected:
            raise TypeError(f"unexpected keyword argument(s): {sorted(unexpected)}")
        missing = self.allowed_keys - set(kwargs)
        if missing:
            raise TypeError(f"missing keyword argument(s): {sorted(missing)}")
        self.last_kwargs = kwargs
        return {"value": self.value}


class FakeResponseModel(BaseModel):
    ok: bool
    msg: str


class FakeStreamChunk:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"delta": type("Delta", (), {"content": content})()})()]


class FakeStream:
    def __init__(self, contents):
        self._contents = contents

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._contents):
            raise StopAsyncIteration
        item = self._contents[self._index]
        self._index += 1
        return FakeStreamChunk(item)


class FakeAsyncChatCompletions:
    def __init__(self, contents):
        self._contents = contents

    async def create(self, **kwargs):
        return FakeStream(self._contents)


class FakeAsyncClient:
    def __init__(self, contents):
        self.chat = type("Chat", (), {"completions": FakeAsyncChatCompletions(contents)})()


class RagasAdapterTests(unittest.TestCase):
    def test_collect_ragas_contexts_deduplicates_and_limits(self):
        pred_answer = {
            "citations": [
                {"evidence_snippet": " 营业收入 100 亿元 "},
                {"evidence_snippet": "营业收入 100 亿元"},
            ]
        }
        debug_detail = {
            "retrieval_results": [
                {"text": "归母净利润 10 亿元"},
                {"text": "归母净利润 10 亿元"},
                {"text": "经营现金流 8 亿元"},
            ]
        }

        contexts = collect_ragas_contexts(pred_answer, debug_detail=debug_detail, limit=2)

        self.assertEqual(contexts, ["营业收入 100 亿元", "归母净利润 10 亿元"])

    def test_score_with_ragas_uses_runtime_scores(self):
        runtime = FakeRagasRuntime()

        result = score_with_ragas(
            question_text="2024年营业收入是多少？",
            answer="100亿元",
            reference="100亿元",
            contexts=["营业收入 100 亿元"],
            runtime=runtime,
        )

        self.assertTrue(result["available"])
        self.assertEqual(result["ragas_score"], 0.8)
        self.assertEqual(result["answer_correctness"], 0.9)
        self.assertEqual(result["context_recall"], 0.95)
        self.assertEqual(result["context_precision"], 0.85)
        self.assertEqual(runtime.last_call["contexts"], ["营业收入 100 亿元"])

    def test_score_with_ragas_reports_runtime_unavailable_reason(self):
        result = score_with_ragas(
            question_text="2024年营业收入是多少？",
            answer="100亿元",
            reference="100亿元",
            contexts=["营业收入 100 亿元"],
            runtime=None,
            unavailable_reason="missing_ragas_llm_configuration",
            unavailable_error="ValueError: missing_ragas_llm_configuration",
        )

        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "missing_ragas_llm_configuration")
        self.assertEqual(result["error"], "ValueError: missing_ragas_llm_configuration")

    def test_prepare_ragas_runtime_respects_disabled_config(self):
        runtime, reason, error = prepare_ragas_runtime(RagasRuntimeConfig(enabled=False))

        self.assertIsNone(runtime)
        self.assertEqual(reason, "ragas_disabled")
        self.assertIsNone(error)

    def test_ragas_runtime_only_passes_supported_kwargs_to_context_metrics(self):
        runtime = object.__new__(RagasRuntime)
        runtime._metrics = {
            "answer_correctness": StrictScoreMetric(
                {"user_input", "response", "reference"},
                1.0,
            ),
            "faithfulness": StrictScoreMetric(
                {"user_input", "response", "retrieved_contexts"},
                0.8,
            ),
            "answer_relevancy": StrictScoreMetric(
                {"user_input", "response"},
                0.7,
            ),
            "context_recall": StrictScoreMetric(
                {"user_input", "retrieved_contexts", "reference"},
                0.6,
            ),
            "context_precision": StrictScoreMetric(
                {"user_input", "reference", "retrieved_contexts"},
                0.5,
            ),
        }

        result = runtime.score(
            question_text="2024年营业收入是多少？",
            answer="100亿元",
            reference="100亿元",
            contexts=["营业收入 100 亿元"],
        )

        self.assertTrue(result["available"])
        self.assertEqual(result["reason"], "ok")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["context_recall"], 0.6)
        self.assertEqual(result["context_precision"], 0.5)
        self.assertNotIn("response", runtime._metrics["context_recall"].last_kwargs)
        self.assertNotIn("response", runtime._metrics["context_precision"].last_kwargs)

    def test_streaming_instructor_llm_parses_streamed_json(self):
        llm = OpenAIStreamingInstructorLLM(
            client=FakeAsyncClient(['{"ok"', ': true, ', '"msg": "hello"}']),
            model="gpt-5.4",
        )

        result = llm.generate("ignored", FakeResponseModel)

        self.assertTrue(result.ok)
        self.assertEqual(result.msg, "hello")


if __name__ == "__main__":
    unittest.main()
