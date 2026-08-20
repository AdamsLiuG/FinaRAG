"""F3 阶段归因 v1 测试：五分支短路判定 + 矩阵聚合。"""
import unittest

from eval.stage_attribution import (
    NON_ERROR_STAGES,
    STAGES,
    _reference_keys,
    attribute_case,
    attribution_matrix,
)


def _gold(
    value="123",
    should_refuse=False,
    refs=None,
    question_text="Q1",
    capability="metric_numeric",
):
    return {
        "question_text": question_text,
        "value": value,
        "should_refuse": should_refuse,
        "references": refs if refs is not None else [
            {"pdf_sha1": "doc_a", "page_index": 6},
        ],
        "capability": capability,
    }


def _pred(value="123", refs=None, question_text="Q1", error=None):
    return {
        "question_text": question_text,
        "value": value,
        "references": refs if refs is not None else [
            {"pdf_sha1": "doc_a", "page_index": 6},
        ],
        "error": error,
    }


class TestAttributeCase(unittest.TestCase):
    def test_t1_should_refuse_but_answered(self):
        """该拒不拒：should_refuse=true 且 pred 未拒答。"""
        pred = _pred(value="123")
        gold = _gold(should_refuse=True)
        self.assertEqual(attribute_case(pred, gold), "refusal_policy")

    def test_t2_correct_refusal(self):
        pred = _pred(value="N/A")
        gold = _gold(should_refuse=True)
        self.assertEqual(attribute_case(pred, gold), "correct_refusal")

    def test_t3_over_refusal(self):
        """不该拒而拒：非 refuse 题但 pred 为 N/A。"""
        pred = _pred(value="N/A")
        gold = _gold()
        self.assertEqual(attribute_case(pred, gold), "over_refusal")

    def test_t4_ungrounded_generation(self):
        """无证据作答：value 非 N/A 但 references 为空。"""
        pred = _pred(value="123", refs=[])
        gold = _gold()
        self.assertEqual(attribute_case(pred, gold), "ungrounded_generation")

    def test_t5_evidence_mislocalization(self):
        """证据错位：引用页与 gold 无交集且值错。"""
        pred = _pred(value="999", refs=[{"pdf_sha1": "doc_a", "page_index": 3}])
        gold = _gold()
        self.assertEqual(attribute_case(pred, gold), "evidence_mislocalization")

    def test_t6_generation_error(self):
        """证据对答案错：引用有交集但值不匹配。"""
        pred = _pred(value="999")
        gold = _gold()
        self.assertEqual(attribute_case(pred, gold), "generation_error")

    def test_t7_citation_error(self):
        """答案对引用错：值匹配但引用无交集。"""
        pred = _pred(value="123", refs=[{"pdf_sha1": "doc_a", "page_index": 3}])
        gold = _gold()
        self.assertEqual(attribute_case(pred, gold), "citation_error")

    def test_t8_gold_without_references(self):
        """gold 无引用且值不匹配 → unknown（缺口显式，不猜测检索/生成）。"""
        pred = _pred(value="999", refs=[{"pdf_sha1": "doc_a", "page_index": 6}])
        gold = _gold(refs=[])
        self.assertEqual(attribute_case(pred, gold), "unknown")

    def test_t8b_gold_without_references_value_match(self):
        """gold 无引用但值匹配 → correct（值层面可判）。"""
        pred = _pred(value="123")
        gold = _gold(refs=[])
        self.assertEqual(attribute_case(pred, gold), "correct")

    def test_error_message_classification(self):
        """异常样本：routing / parse / generation 关键词归类。"""
        self.assertEqual(
            attribute_case(_pred(error="No company name found for xxx"), None), "routing"
        )
        self.assertEqual(
            attribute_case(_pred(error="docling parse failed on pdf"), None), "parse"
        )
        self.assertEqual(
            attribute_case(_pred(error="timeout waiting for model"), None), "generation"
        )

    def test_no_gold_structural_only(self):
        """无 gold 对照：只判结构性问题（拒答/无引用），其余 unknown。"""
        self.assertEqual(attribute_case(_pred(value="N/A"), None), "over_refusal")
        self.assertEqual(attribute_case(_pred(value="1", refs=[]), None), "ungrounded_generation")
        self.assertEqual(attribute_case(_pred(value="1"), None), "unknown")

    def test_reference_keys_skips_malformed(self):
        entry = {"references": [
            {"pdf_sha1": "a", "page_index": 1},
            {"pdf_sha1": "", "page_index": 2},      # 缺 sha 跳过
            {"pdf_sha1": "b", "page_index": None},  # 缺页码跳过
            "not-a-dict",                            # 非法结构跳过
        ]}
        self.assertEqual(_reference_keys(entry), {("a", 1)})


class TestAttributionMatrix(unittest.TestCase):
    def test_matrix_aggregates_and_rates(self):
        preds = [
            _pred(value="123", question_text="Q1"),                                    # correct
            _pred(value="N/A", question_text="Q2"),                                    # over_refusal
            _pred(value="999", refs=[{"pdf_sha1": "x", "page_index": 1}], question_text="Q3"),  # misloc
            _pred(value="N/A", question_text="Q4"),                                    # 无 gold → unmatched
        ]
        golds = [
            _gold(question_text="Q1"),
            _gold(question_text="Q2"),
            _gold(question_text="Q3"),
        ]
        report = attribution_matrix(preds, golds)
        self.assertEqual(report["n"], 3)
        self.assertEqual(report["unmatched_predictions"], 1)
        self.assertEqual(report["stage_counts"]["correct"], 1)
        self.assertEqual(report["stage_counts"]["over_refusal"], 1)
        self.assertEqual(report["stage_counts"]["evidence_mislocalization"], 1)
        self.assertAlmostEqual(report["stage_rates"]["over_refusal"], 1 / 3, places=4)
        # 非错误阶段不进 examples
        self.assertNotIn("correct", report["examples"])
        self.assertIn("evidence_mislocalization", report["examples"])
        # capability 分桶
        self.assertEqual(report["by_capability"]["metric_numeric"]["correct"], 1)

    def test_stages_constant_complete(self):
        """STAGES 覆盖 v1 归因树全部错误分支。"""
        expected = {
            "routing", "parse", "refusal_policy", "over_refusal",
            "ungrounded_generation", "citation_error", "generation_error",
            "evidence_mislocalization",
        }
        self.assertEqual(set(STAGES), expected)
        self.assertEqual(NON_ERROR_STAGES, {"correct", "correct_refusal", "unknown"})


if __name__ == "__main__":
    unittest.main()
