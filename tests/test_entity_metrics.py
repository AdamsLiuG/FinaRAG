from pathlib import Path
import unittest

from eval.dataset_schema import load_gold_answer_set, load_question_set
from eval.entity_metrics import score_finance_entities


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "finance_eval_v1"


class EntityMetricsTests(unittest.TestCase):
    def test_entity_score_matches_financial_metadata(self):
        question = load_question_set(DATA_DIR / "questions.sample.json").questions[1]
        gold_answer = load_gold_answer_set(DATA_DIR / "answers_gold.sample.json").answers[1]
        pred_answer = {
            "value": "356778998.12",
            "references": [{"pdf_sha1": "300531_2024_20250429", "page_index": 114}],
            "citations": [
                {
                    "page": 115,
                    "source": "300531_2024_20250429",
                    "company_name": "优博讯",
                    "stock_code": "300531",
                    "security_code": "300531",
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
                    "page": 115,
                    "metadata": {
                        "sha1_name": "300531_2024_20250429",
                        "company_name": "优博讯",
                        "security_code": "300531",
                        "stock_code": "300531",
                        "report_year": 2024,
                        "doc_source_type": "annual_report",
                        "currency": "CNY",
                        "unit_hint": "元",
                        "period": "本期",
                    },
                }
            ]
        }

        report = score_finance_entities(pred_answer, gold_answer, question, debug_detail=debug_detail)

        self.assertFalse(report["skipped"])
        self.assertEqual(report["entity_score"], 1.0)
        self.assertGreaterEqual(report["matched_field_count"], 6)

    def test_entity_score_skips_refusal_case(self):
        gold_answer = load_gold_answer_set(DATA_DIR / "answers_gold.sample.json").answers[2]

        report = score_finance_entities({"value": "N/A"}, gold_answer, None)

        self.assertTrue(report["skipped"])
        self.assertIsNone(report["entity_score"])


if __name__ == "__main__":
    unittest.main()
