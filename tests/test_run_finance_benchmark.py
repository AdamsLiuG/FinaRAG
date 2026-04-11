from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from eval.ragas_adapter import RagasRuntimeConfig
from eval.run_finance_benchmark import run_finance_benchmark


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "finance_eval_v1"


class RunFinanceBenchmarkTests(unittest.TestCase):
    def test_run_finance_benchmark_with_existing_answers(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            report = run_finance_benchmark(
                pipeline_dataset_dir=ROOT / "data" / "top10_industries_2024_20each",
                questions_file=DATA_DIR / "questions.sample.json",
                gold_answers_file=DATA_DIR / "answers_gold.sample.json",
                answers_file=DATA_DIR / "pred_answers.sample.json",
                debug_file=DATA_DIR / "pred_answers_debug.sample.json",
                pred_answers_out=tmpdir_path / "pred_answers.exported.json",
                eval_debug_out=tmpdir_path / "pred_answers.exported_debug.json",
                report_out=tmpdir_path / "finance_eval.report.json",
                include_cases=True,
                ragas_config=RagasRuntimeConfig(enabled=False),
            )

            self.assertEqual(report["summary"]["matched_predictions"], 3)
            self.assertEqual(report["summary"]["unmatched_predictions"], [])
            self.assertTrue((tmpdir_path / "pred_answers.exported.json").exists())
            self.assertTrue((tmpdir_path / "pred_answers.exported_debug.json").exists())
            self.assertTrue((tmpdir_path / "finance_eval.report.json").exists())

            evaluation_report = json.loads((tmpdir_path / "finance_eval.report.json").read_text(encoding="utf-8"))
            self.assertEqual(evaluation_report["aggregate_metrics"]["reference_exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()
