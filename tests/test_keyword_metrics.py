import unittest

from eval.dataset_schema import FinanceEvalQuestion, FinanceGoldAnswer
from eval.keyword_metrics import score_answer_keywords


class KeywordMetricsTests(unittest.TestCase):
    def test_metadata_aliases_count_toward_expected_keywords(self):
        question = FinanceEvalQuestion.model_validate(
            {
                "id": "q1",
                "text": "优博讯2024年年报中的研发投入金额是多少？",
                "kind": "number",
                "company_name": "优博讯",
                "stock_code": "300531",
                "report_year": 2024,
                "report_type": "annual_report",
                "period": "本期",
                "metric_name": "研发投入",
                "currency": "CNY",
                "unit": "元",
            }
        )
        gold_answer = FinanceGoldAnswer(
            question_id="q1",
            question_text=question.question_text,
            kind="number",
            value="356,778,998.12",
            company_name="优博讯",
            stock_code="300531",
            report_year=2024,
            report_type="annual_report",
            period="本期",
            metric_name="研发投入",
            currency="CNY",
            unit="元",
        )
        pred_answer = {
            "value": "356,778,998.12",
            "citations": [
                {
                    "evidence_snippet": "研发投入合计 356,778,998.12 元",
                    "company_name": "优博讯",
                    "stock_code": "300531",
                    "report_year": 2024,
                    "doc_source_type": "annual_report",
                    "currency": "CNY",
                    "unit": "人民币元",
                }
            ],
        }
        debug_detail = {
            "retrieval_results": [
                {
                    "text": "研发投入合计 356,778,998.12 元",
                    "metadata": {
                        "company_name": "优博讯",
                        "stock_code": "300531",
                        "report_year": 2024,
                        "doc_source_type": "annual_report",
                        "currency": "CNY",
                        "period": "本期",
                        "unit_hint": "元",
                    },
                }
            ]
        }

        result = score_answer_keywords(pred_answer, gold_answer, question, debug_detail=debug_detail)

        self.assertEqual(result["keyword_score"], 1.0)
        self.assertEqual(result["missing_keywords"], [])


if __name__ == "__main__":
    unittest.main()
