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
        self.assertEqual(validated.confidence, "low")


if __name__ == "__main__":
    unittest.main()
