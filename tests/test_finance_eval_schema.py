from pathlib import Path
import unittest

from eval.dataset_schema import export_json_schemas, load_gold_answer_set, load_manifest, load_question_set, validate_dataset_alignment


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "finance_eval_v1"


class FinanceEvalSchemaTests(unittest.TestCase):
    def test_sample_dataset_passes_alignment_validation(self):
        question_set = load_question_set(DATA_DIR / "questions.sample.json")
        answer_set = load_gold_answer_set(DATA_DIR / "answers_gold.sample.json")

        report = validate_dataset_alignment(question_set, answer_set)

        self.assertTrue(report["valid"])
        self.assertEqual(report["question_count"], 3)
        self.assertEqual(report["answer_count"], 3)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["warnings"], [])

    def test_sample_manifest_loads(self):
        manifest = load_manifest(DATA_DIR / "dataset_manifest.sample.json")

        self.assertEqual(manifest.dataset_name, "finance_eval_v1_sample")
        self.assertEqual(manifest.question_count, 3)
        self.assertEqual(manifest.answer_count, 3)

    def test_json_schema_export_contains_core_models(self):
        schemas = export_json_schemas()

        self.assertIn("finance_eval_question", schemas)
        self.assertIn("finance_gold_answer", schemas)
        self.assertIn("finance_eval_manifest", schemas)


if __name__ == "__main__":
    unittest.main()
