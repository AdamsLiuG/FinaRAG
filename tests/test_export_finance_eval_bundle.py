from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from eval.export_finance_eval_bundle import export_finance_eval_bundle
from eval.finance_eval import evaluate_finance_answers
from eval.ragas_adapter import RagasRuntimeConfig


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "finance_eval_v1"


class ExportFinanceEvalBundleTests(unittest.TestCase):
    def test_exporter_adds_question_ids_and_keeps_debug_usable(self):
        questions_payload = json.loads((DATA_DIR / "questions.sample.json").read_text(encoding="utf-8"))
        raw_answers_payload = json.loads((DATA_DIR / "pred_answers.sample.json").read_text(encoding="utf-8"))
        raw_debug_payload = json.loads((DATA_DIR / "pred_answers_debug.sample.json").read_text(encoding="utf-8"))

        for answer in raw_answers_payload["answers"]:
            answer.pop("question_id", None)

        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            questions_file = tmpdir_path / "questions.json"
            raw_answers_file = tmpdir_path / "answers.json"
            raw_debug_file = tmpdir_path / "answers_debug.json"
            pred_out = tmpdir_path / "pred_answers.json"
            debug_out = tmpdir_path / "pred_answers_debug.json"

            questions_file.write_text(json.dumps(questions_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            raw_answers_file.write_text(json.dumps(raw_answers_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            raw_debug_file.write_text(json.dumps(raw_debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            report = export_finance_eval_bundle(
                questions_file=questions_file,
                answers_file=raw_answers_file,
                pred_answers_out=pred_out,
                debug_out=debug_out,
                debug_file=raw_debug_file,
            )

            self.assertEqual(report["matched_questions"], 3)
            exported_answers = json.loads(pred_out.read_text(encoding="utf-8"))["answers"]
            self.assertEqual(exported_answers[0]["question_id"], "fin-eval-0001")

            eval_report = evaluate_finance_answers(
                questions_file=questions_file,
                gold_answers_file=DATA_DIR / "answers_gold.sample.json",
                pred_answers_file=pred_out,
                debug_file=debug_out,
                include_cases=True,
                ragas_config=RagasRuntimeConfig(enabled=False),
            )
            self.assertEqual(eval_report["summary"]["matched_predictions"], 3)
            self.assertEqual(eval_report["aggregate_metrics"]["reference_exact_match"], 1.0)

    def test_exporter_fills_missing_predictions_with_na(self):
        questions_payload = json.loads((DATA_DIR / "questions.sample.json").read_text(encoding="utf-8"))
        raw_answers_payload = json.loads((DATA_DIR / "pred_answers.sample.json").read_text(encoding="utf-8"))

        raw_answers_payload["answers"] = raw_answers_payload["answers"][:2]
        for answer in raw_answers_payload["answers"]:
            answer.pop("question_id", None)

        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            questions_file = tmpdir_path / "questions.json"
            raw_answers_file = tmpdir_path / "answers.json"
            pred_out = tmpdir_path / "pred_answers.json"
            debug_out = tmpdir_path / "pred_answers_debug.json"

            questions_file.write_text(json.dumps(questions_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            raw_answers_file.write_text(json.dumps(raw_answers_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            report = export_finance_eval_bundle(
                questions_file=questions_file,
                answers_file=raw_answers_file,
                pred_answers_out=pred_out,
                debug_out=debug_out,
                debug_file=None,
            )

            exported = json.loads(pred_out.read_text(encoding="utf-8"))
            self.assertEqual(len(exported["answers"]), 3)
            self.assertEqual(exported["answers"][2]["question_id"], "fin-eval-0003")
            self.assertEqual(exported["answers"][2]["value"], "N/A")
            self.assertEqual(report["missing_questions"], ["fin-eval-0003"])


if __name__ == "__main__":
    unittest.main()
