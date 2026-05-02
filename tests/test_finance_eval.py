from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

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

    def test_ragas_resume_reuses_successful_cases_and_retries_failed_cases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            resume_report = Path(tmp_dir) / "existing_ragas.report.json"
            resume_report.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "question_id": "fin-eval-0001",
                                "question_text": "国网信通2024年年报中的法定代表人是谁？",
                                "ragas": {
                                    "available": True,
                                    "reason": "ok",
                                    "error": None,
                                    "errors": [],
                                    "contexts_used": 1,
                                    "answer_correctness": 0.91,
                                    "faithfulness": 0.92,
                                    "answer_relevancy": 0.93,
                                    "context_recall": 0.94,
                                    "context_precision": 0.95,
                                    "ragas_score": 0.93,
                                },
                            },
                            {
                                "question_id": "fin-eval-0002",
                                "question_text": "优博讯2024年年报中的研发投入金额是多少？",
                                "ragas": {
                                    "available": False,
                                    "reason": "metric_scoring_failed",
                                    "error": "old API failure",
                                    "errors": ["old API failure"],
                                    "contexts_used": 1,
                                    "answer_correctness": None,
                                    "faithfulness": None,
                                    "answer_relevancy": None,
                                    "context_recall": None,
                                    "context_precision": None,
                                    "ragas_score": None,
                                },
                            },
                            {
                                "question_id": "fin-eval-0003",
                                "question_text": "国网信通2024年年报中是否披露了火星业务收入？",
                                "ragas": {
                                    "available": False,
                                    "reason": "no_contexts",
                                    "error": None,
                                    "errors": [],
                                    "contexts_used": 0,
                                    "answer_correctness": None,
                                    "faithfulness": None,
                                    "answer_relevancy": None,
                                    "context_recall": None,
                                    "context_precision": None,
                                    "ragas_score": None,
                                },
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            calls = []

            def fake_score_with_ragas(**kwargs):
                calls.append(kwargs["question_text"])
                return {
                    "available": True,
                    "reason": "ok",
                    "error": None,
                    "errors": [],
                    "contexts_used": len(kwargs["contexts"]),
                    "answer_correctness": 0.81,
                    "faithfulness": 0.82,
                    "answer_relevancy": 0.83,
                    "context_recall": 0.84,
                    "context_precision": 0.85,
                    "ragas_score": 0.83,
                }

            with patch("eval.finance_eval.score_with_ragas", side_effect=fake_score_with_ragas):
                report = evaluate_finance_answers(
                    questions_file=DATA_DIR / "questions.sample.json",
                    gold_answers_file=DATA_DIR / "answers_gold.sample.json",
                    pred_answers_file=DATA_DIR / "pred_answers.sample.json",
                    debug_file=DATA_DIR / "pred_answers_debug.sample.json",
                    include_cases=True,
                    ragas_config=RagasRuntimeConfig(enabled=False),
                    ragas_resume_report=resume_report,
                )

        self.assertEqual(calls, ["优博讯2024年年报中的研发投入金额是多少？"])
        cases_by_id = {case["question_id"]: case for case in report["cases"]}
        self.assertEqual(cases_by_id["fin-eval-0001"]["ragas"]["ragas_score"], 0.93)
        self.assertEqual(cases_by_id["fin-eval-0002"]["ragas"]["ragas_score"], 0.83)
        self.assertEqual(cases_by_id["fin-eval-0003"]["ragas"]["reason"], "no_contexts")
        self.assertEqual(report["summary"]["ragas_available_cases"], 2)
        self.assertEqual(report["ragas"]["resume"]["reused_cases"], 2)
        self.assertEqual(report["ragas"]["resume"]["retried_cases"], 1)

    def test_ragas_progress_log_and_checkpoint_report_are_written_per_case(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            progress_log = Path(tmp_dir) / "ragas.progress.jsonl"
            checkpoint_report = Path(tmp_dir) / "ragas.partial.report.json"

            def fake_score_with_ragas(**kwargs):
                return {
                    "available": True,
                    "reason": "ok",
                    "error": None,
                    "errors": [],
                    "contexts_used": len(kwargs["contexts"]),
                    "answer_correctness": 0.71,
                    "faithfulness": 0.72,
                    "answer_relevancy": 0.73,
                    "context_recall": 0.74,
                    "context_precision": 0.75,
                    "ragas_score": 0.73,
                }

            with patch("eval.finance_eval.score_with_ragas", side_effect=fake_score_with_ragas):
                report = evaluate_finance_answers(
                    questions_file=DATA_DIR / "questions.sample.json",
                    gold_answers_file=DATA_DIR / "answers_gold.sample.json",
                    pred_answers_file=DATA_DIR / "pred_answers.sample.json",
                    debug_file=DATA_DIR / "pred_answers_debug.sample.json",
                    include_cases=True,
                    ragas_config=RagasRuntimeConfig(enabled=False),
                    ragas_progress_log=progress_log,
                    ragas_checkpoint_report=checkpoint_report,
                    ragas_checkpoint_interval=1,
                )

            self.assertTrue(progress_log.exists())
            progress_events = [
                json.loads(line)
                for line in progress_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            case_events = [event for event in progress_events if event["event"] == "case_completed"]
            self.assertEqual(len(case_events), 3)
            self.assertEqual(case_events[0]["question_id"], "fin-eval-0001")
            self.assertEqual(case_events[-1]["matched_cases"], 3)
            self.assertEqual(progress_events[-1]["event"], "finished")

            checkpoint_payload = json.loads(checkpoint_report.read_text(encoding="utf-8"))
            self.assertEqual(len(checkpoint_payload["cases"]), 3)
            self.assertTrue(checkpoint_payload["ragas"]["checkpoint"]["complete"])
            self.assertEqual(checkpoint_payload["summary"]["matched_predictions"], 3)

            resume_calls = []

            def fail_if_called(**kwargs):
                resume_calls.append(kwargs["question_text"])
                raise AssertionError("checkpointed RAGAS result should have been reused")

            with patch("eval.finance_eval.score_with_ragas", side_effect=fail_if_called):
                resumed = evaluate_finance_answers(
                    questions_file=DATA_DIR / "questions.sample.json",
                    gold_answers_file=DATA_DIR / "answers_gold.sample.json",
                    pred_answers_file=DATA_DIR / "pred_answers.sample.json",
                    debug_file=DATA_DIR / "pred_answers_debug.sample.json",
                    include_cases=True,
                    ragas_config=RagasRuntimeConfig(enabled=False),
                    ragas_resume_report=checkpoint_report,
                )

            self.assertEqual(resume_calls, [])
            self.assertEqual(resumed["ragas"]["resume"]["reused_cases"], 3)
            self.assertEqual(report["summary"]["ragas_available_cases"], 3)


if __name__ == "__main__":
    unittest.main()
