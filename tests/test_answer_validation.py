import unittest

from src.answer_validation import validate_answer
from src.query_rewrite import QuestionRewriter


class AnswerValidationTests(unittest.TestCase):
    def test_currency_mismatch_forces_refusal(self):
        query_plan = QuestionRewriter().rewrite(
            "What was Alpha Corp revenue in 2023 in USD?",
            schema="number",
            company_name="Alpha Corp",
        )
        answer_dict = {
            "final_answer": 1200,
            "confidence": "high",
            "relevant_pages": [8],
            "references": [{"pdf_sha1": "sha1-alpha", "page_index": 8}],
            "citations": [{"page": 8, "chunk_type": "serialized_table"}],
        }
        retrieval_results = [
            {
                "page": 8,
                "distance": 0.9,
                "metadata": {
                    "currency": "EUR",
                    "report_year": 2023,
                    "topic_flags": [],
                },
            }
        ]

        validated = validate_answer(answer_dict, retrieval_results, query_plan)

        self.assertEqual(validated.answer["final_answer"], "N/A")
        self.assertEqual(validated.confidence, "low")
        self.assertIn("currency_mismatch", validated.validation_flags)

    def test_numeric_grounding_period_mismatch_forces_refusal(self):
        query_plan = QuestionRewriter().rewrite(
            "宁德时代2024Q3营收是多少？",
            schema="number",
            company_name="宁德时代",
        )
        answer_dict = {
            "final_answer": 1200,
            "confidence": "high",
            "relevant_pages": [8],
            "references": [{"pdf_sha1": "doc-alpha", "page_index": 8}],
            "citations": [{"page": 8, "chunk_type": "table_grounding"}],
            "table_grounding_result": {
                "table_id": "tbl-1",
                "page": 8,
                "period": "2024年报",
                "unit": "人民币百万元",
                "normalized_value": 1200,
            },
        }
        retrieval_results = [
            {
                "page": 8,
                "distance": 0.9,
                "metadata": {
                    "currency": "CNY",
                    "report_year": 2024,
                    "period": "2024年报",
                    "doc_source_type": "annual_report",
                },
            }
        ]

        validated = validate_answer(answer_dict, retrieval_results, query_plan)

        self.assertEqual(validated.answer["final_answer"], "N/A")
        self.assertEqual(validated.confidence, "low")
        self.assertIn("numeric_grounding_period_mismatch", validated.validation_flags)

    def test_chart_grounding_supports_numeric_answer_but_flags_low_confidence(self):
        query_plan = QuestionRewriter().rewrite(
            "浦发银行2024年营业收入是多少亿元？",
            schema="number",
            company_name="浦发银行",
        )
        answer_dict = {
            "final_answer": 135.8,
            "confidence": "medium",
            "relevant_pages": [35],
            "references": [{"pdf_sha1": "600000_2024", "page": 35}],
            "citations": [{"page": 35, "chunk_type": "chart_grounding"}],
            "chart_grounding_result": {
                "chart_id": "600000_2024_p35_pic2",
                "page": 35,
                "period": "2024",
                "unit": "亿元",
                "normalized_value": 13580000000.0,
                "confidence": 0.52,
            },
        }
        retrieval_results = [
            {
                "page": 35,
                "distance": 0.9,
                "metadata": {
                    "currency": "CNY",
                    "report_year": 2024,
                    "period": "2024",
                    "doc_source_type": "annual_report",
                },
            }
        ]

        validated = validate_answer(answer_dict, retrieval_results, query_plan)

        self.assertIn("chart_grounding_low_confidence", validated.validation_flags)
        self.assertNotIn("numeric_answer_without_structured_grounding", validated.validation_flags)
        self.assertNotIn("metric_evidence_mismatch", validated.validation_flags)
        self.assertEqual(validated.confidence, "low")

    def test_implicit_intent_requires_target_evidence(self):
        query_plan = QuestionRewriter().rewrite(
            "宁德时代2024年现金压力大不大？",
            schema="boolean",
            company_name="宁德时代",
        )
        answer_dict = {
            "final_answer": False,
            "confidence": "high",
            "relevant_pages": [18],
            "references": [{"pdf_sha1": "300750_2024", "page_index": 18}],
            "citations": [{"page": 18, "chunk_type": "text"}],
        }
        retrieval_results = [
            {
                "page": 18,
                "distance": 0.8,
                "text": "公司营业收入和净利润保持增长，管理层讨论了市场需求变化。",
                "metadata": {
                    "company_name": "宁德时代",
                    "report_year": 2024,
                    "topic_flags": [],
                    "chunk_type": "text",
                },
            }
        ]

        validated = validate_answer(answer_dict, retrieval_results, query_plan)

        self.assertIn("implicit_intent_evidence_mismatch", validated.validation_flags)
        self.assertEqual(validated.confidence, "medium")


class CaliberConflictRefusalTests(unittest.TestCase):
    """fail-closed 口径门禁：口径错位时强制拒答，而非返回一个错口径的数字。"""

    @staticmethod
    def _retrieval(text):
        return [
            {
                "page": 12,
                "distance": 0.7,
                "text": text,
                "metadata": {
                    "company_name": "宁德时代",
                    "report_year": 2024,
                    "topic_flags": [],
                    "chunk_type": "text",
                },
            }
        ]

    @staticmethod
    def _answer():
        return {
            "final_answer": 12345.0,
            "confidence": "high",
            "relevant_pages": [12],
            "references": [{"pdf_sha1": "300750_2024", "page_index": 12}],
            "citations": [{"page": 12, "chunk_type": "text"}],
        }

    def test_refuses_when_only_confusable_caliber_present(self):
        # 问扣非归母，证据只有归母净利润——不得拿归母数字当扣非答案
        query_plan = QuestionRewriter().rewrite(
            "宁德时代2024年扣非归母净利润是多少？",
            schema="number",
            company_name="宁德时代",
        )
        validated = validate_answer(
            self._answer(),
            self._retrieval("本年度归母净利润为 12,345 万元，同比增长。"),
            query_plan,
        )
        self.assertIn("metric_caliber_conflict_refused", validated.validation_flags)
        self.assertEqual(validated.answer["final_answer"], "N/A")
        self.assertEqual(validated.confidence, "low")

    def test_no_refusal_when_own_caliber_present(self):
        query_plan = QuestionRewriter().rewrite(
            "宁德时代2024年扣非归母净利润是多少？",
            schema="number",
            company_name="宁德时代",
        )
        validated = validate_answer(
            self._answer(),
            self._retrieval("扣除非经常性损益后的净利润为 12,345 万元，归母净利润为 14,000 万元。"),
            query_plan,
        )
        self.assertNotIn("metric_caliber_conflict_refused", validated.validation_flags)
        self.assertEqual(validated.answer["final_answer"], 12345.0)

    def test_no_refusal_for_non_numeric_schema(self):
        # 防误拒边界一：非数值类不拒答
        query_plan = QuestionRewriter().rewrite(
            "宁德时代2024年扣非归母净利润是否为正？",
            schema="boolean",
            company_name="宁德时代",
        )
        answer = self._answer()
        answer["final_answer"] = True
        validated = validate_answer(
            answer,
            self._retrieval("本年度归母净利润为 12,345 万元。"),
            query_plan,
        )
        self.assertNotIn("metric_caliber_conflict_refused", validated.validation_flags)
        self.assertEqual(validated.answer["final_answer"], True)

    def test_no_refusal_for_low_intent_confidence(self):
        # 防误拒边界二：意图置信度不足时不拒答
        query_plan = QuestionRewriter().rewrite(
            "宁德时代2024年扣非归母净利润是多少？",
            schema="number",
            company_name="宁德时代",
        )
        query_plan.intent_confidence = 0.5
        validated = validate_answer(
            self._answer(),
            self._retrieval("本年度归母净利润为 12,345 万元。"),
            query_plan,
        )
        self.assertNotIn("metric_caliber_conflict_refused", validated.validation_flags)
        self.assertEqual(validated.answer["final_answer"], 12345.0)


if __name__ == "__main__":
    unittest.main()
