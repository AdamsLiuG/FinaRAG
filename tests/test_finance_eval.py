from pathlib import Path
import unittest

from eval.finance_eval import evaluate_finance_answers
from eval.ragas_adapter import RagasRuntimeConfig


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "finance_eval_v1"


class FinanceEvalTests(unittest.TestCase):
    def test_finance_eval_smoke_test(self):
        report = evaluate_finance_answers(
            questions_file=DATA_DIR / "questions.sample.json",
            gold_answers_file=DATA_DIR / "answers_gold.sample.json",
            pred_answers_file=DATA_DIR / "pred_answers.sample.json",
            debug_file=DATA_DIR / "pred_answers_debug.sample.json",
            include_cases=True,
            ragas_config=RagasRuntimeConfig(enabled=False),
        )

        self.assertEqual(report["summary"]["matched_predictions"], 3)
        self.assertEqual(report["summary"]["unmatched_predictions"], [])
        self.assertGreater(report["summary"]["mean_final_quality_score"], 0.95)
        self.assertEqual(report["aggregate_metrics"]["reference_exact_match"], 1.0)
        self.assertEqual(report["ranked_retrieval_metrics"]["hit_at_10"], 1.0)
        self.assertEqual(report["summary"]["ragas_available_cases"], 0)
        self.assertEqual(report["ragas"]["runtime_reason"], "ragas_disabled")
        self.assertEqual(len(report["cases"]), 3)


if __name__ == "__main__":
    unittest.main()
