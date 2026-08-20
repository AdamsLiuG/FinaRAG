import unittest

from src.metric_atom_review import (
    ACCEPT_THRESHOLD,
    MAX_SCORE,
    REVIEW_DIMENSIONS,
    SameModelReviewError,
    aggregate_review,
    assert_reviewer_differs,
    build_review_prompt,
    parse_review_response,
    render_review_report,
    review_atom_with_llm,
)


class FakeProcessor:
    """假 api_processor：记录调用参数并返回预置响应，不发真实请求。"""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def send_message(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _scores(definition=2, caliber=2, alias=2, confusable=2):
    return {
        "definition_accuracy": definition,
        "caliber_completeness": caliber,
        "alias_coverage": alias,
        "confusable_annotation": confusable,
    }


class ParseReviewTests(unittest.TestCase):
    def test_full_score_accepts(self):
        result = parse_review_response('{"scores": %s, "issues": []}' % str(_scores()).replace("'", '"'))
        self.assertEqual(result["verdict"], "accept")
        self.assertEqual(sum(result["scores"].values()), MAX_SCORE)

    def test_any_zero_dimension_rejects(self):
        payload = str(_scores(confusable=0)).replace("'", '"')
        result = parse_review_response('{"scores": %s, "issues": ["\u672a\u6807\u51fa\u6613\u6df7\u6307\u6807"]}' % payload)
        self.assertEqual(result["verdict"], "reject")
        self.assertIn("未标出易混指标", result["issues"])

    def test_below_threshold_requires_revision(self):
        # 无 0 分但总分不足（1+1+1+2=5 < 6）
        payload = str(_scores(definition=1, caliber=1, alias=1)).replace("'", '"')
        result = parse_review_response('{"scores": %s, "issues": []}' % payload)
        self.assertLess(sum(result["scores"].values()), ACCEPT_THRESHOLD)
        self.assertEqual(result["verdict"], "revise")

    def test_missing_dimension_requires_revision(self):
        # 维度缺失视为质检未完成，不得放行
        result = parse_review_response('{"scores": {"definition_accuracy": 2}, "issues": []}')
        self.assertEqual(result["verdict"], "revise")
        self.assertTrue(any("缺少有效评分" in issue for issue in result["issues"]))

    def test_unparseable_response_is_fail_closed(self):
        # 质检失败不得当作质检通过
        for bad in ["模型拒绝回答", "", "{不是合法 JSON", None]:
            result = parse_review_response(bad)
            self.assertEqual(result["verdict"], "revise", msg=f"input={bad!r}")
        self.assertTrue(parse_review_response("模型拒绝回答")["parse_failed"])

    def test_score_out_of_range_is_clamped(self):
        payload = '{"scores": {"definition_accuracy": 5, "caliber_completeness": -1, ' \
                  '"alias_coverage": 2, "confusable_annotation": 2}, "issues": []}'
        result = parse_review_response(payload)
        self.assertEqual(result["scores"]["definition_accuracy"], 2)
        self.assertEqual(result["scores"]["caliber_completeness"], 0)
        self.assertEqual(result["verdict"], "reject")


class ReviewCallTests(unittest.TestCase):
    def test_review_atom_passes_model_and_payload(self):
        processor = FakeProcessor('{"scores": %s, "issues": []}' % str(_scores()).replace("'", '"'))
        review = review_atom_with_llm(
            processor,
            {
                "atom_id": "MET-001",
                "standard_name": "扣非归母净利润",
                "aliases": ["扣非归母净利润"],
                "definition": "扣除非经常性损益后归属于母公司股东的净利润",
                "caliber_notes": "与归母净利润区分",
                "unit": "元",
                "statement_location": "主要财务指标",
                "confusable_names": ["归母净利润"],
                "pitfalls": "口径不可混用",
            },
            model="reviewer-model",
        )
        self.assertEqual(review["verdict"], "accept")
        self.assertEqual(review["atom_id"], "MET-001")
        self.assertEqual(processor.calls[0]["model"], "reviewer-model")
        self.assertIn("扣非归母净利润", processor.calls[0]["human_content"])
        self.assertIn("归母净利润", processor.calls[0]["human_content"])

    def test_build_prompt_handles_empty_fields(self):
        prompt = build_review_prompt({"standard_name": "测试指标"})
        self.assertIn("测试指标", prompt)
        self.assertIn("（无）", prompt)


class AggregateAndReportTests(unittest.TestCase):
    def _reviews(self):
        return [
            {"standard_name": "指标A", "verdict": "accept", "scores": _scores(), "issues": []},
            {
                "standard_name": "指标B",
                "verdict": "reject",
                "scores": _scores(alias=0),
                "issues": ["别名含口径放宽词"],
            },
            {
                "standard_name": "指标C",
                "verdict": "revise",
                "scores": _scores(definition=1, caliber=1, alias=1),
                "issues": ["口径说明不完整"],
            },
        ]

    def test_aggregate_blocks_on_non_accept(self):
        summary = aggregate_review(self._reviews())
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["counts"], {"accept": 1, "revise": 1, "reject": 1})
        self.assertEqual(summary["blocking"], ["指标B", "指标C"])
        self.assertFalse(summary["gate_passed"])

    def test_gate_passes_only_when_all_accepted(self):
        summary = aggregate_review(
            [{"standard_name": "指标A", "verdict": "accept", "scores": _scores(), "issues": []}]
        )
        self.assertTrue(summary["gate_passed"])
        self.assertEqual(summary["accept_rate"], 1.0)

    def test_empty_reviews_do_not_pass_gate(self):
        # 零受检等于未质检，不得放行
        self.assertFalse(aggregate_review([])["gate_passed"])

    def test_render_report_contains_scores_and_issues(self):
        report = render_review_report(self._reviews())
        self.assertIn("# 指标原子质检报告", report)
        self.assertIn("门禁结论：未通过", report)
        self.assertIn("指标B", report)
        self.assertIn("别名含口径放宽词", report)
        self.assertIn(f"/{MAX_SCORE}", report)
        for dimension in REVIEW_DIMENSIONS:
            self.assertNotIn(dimension, report.split("## 扣分依据")[0].split("\n")[0])


class ProducerDoesNotSelfReviewTests(unittest.TestCase):
    def test_same_model_rejected(self):
        with self.assertRaises(SameModelReviewError):
            assert_reviewer_differs("qwen-max", "qwen-max")
        with self.assertRaises(SameModelReviewError):
            assert_reviewer_differs(" qwen-max ", "qwen-max")

    def test_different_model_allowed(self):
        assert_reviewer_differs("qwen-plus", "qwen-max")
        assert_reviewer_differs("", "qwen-max")  # 未声明提炼模型时不阻断


if __name__ == "__main__":
    unittest.main()
