from pathlib import Path
import unittest

from eval.dataset_schema import load_gold_answer_set, load_manifest, load_question_set, validate_dataset_alignment


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "finance_eval_benchmark_v2"


class FinanceEvalBenchmarkV2Tests(unittest.TestCase):
    def _assert_dataset(self, dataset_dir: Path, total: int, dev_total: int, test_total: int):
        question_set = load_question_set(dataset_dir / "questions.json")
        answer_set = load_gold_answer_set(dataset_dir / "answers_gold.json")
        manifest = load_manifest(dataset_dir / "dataset_manifest.json")
        report = validate_dataset_alignment(question_set, answer_set)

        self.assertTrue(report["valid"])
        self.assertEqual(len(question_set.questions), total)
        self.assertEqual(len(answer_set.answers), total)
        self.assertEqual(manifest.question_count, total)
        self.assertEqual(manifest.answer_count, total)

        dev_questions = load_question_set(dataset_dir / "splits" / "dev" / "questions.json")
        test_questions = load_question_set(dataset_dir / "splits" / "test" / "questions.json")
        self.assertEqual(len(dev_questions.questions), dev_total)
        self.assertEqual(len(test_questions.questions), test_total)

    def test_core120_dataset_is_aligned(self):
        self._assert_dataset(DATA_DIR / "core120", total=120, dev_total=60, test_total=60)

    def test_core200_dataset_is_aligned(self):
        self._assert_dataset(DATA_DIR / "core200", total=200, dev_total=100, test_total=100)

    def test_core120_contains_new_scenarios(self):
        question_set = load_question_set(DATA_DIR / "core120" / "questions.json")
        scenarios = {question.metadata.get("scenario_family") for question in question_set.questions}

        self.assertIn("numeric_unit_wanyuan", scenarios)
        self.assertIn("numeric_unit_yiyuan", scenarios)
        self.assertIn("numeric_threshold_boolean", scenarios)
        self.assertIn("metadata_count_aggregation", scenarios)
        self.assertIn("compare_boolean_variant", scenarios)
        self.assertIn("refusal_teacher_verified", scenarios)

    def test_core200_has_expected_refusal_and_membership_coverage(self):
        question_set = load_question_set(DATA_DIR / "core200" / "questions.json")
        refusal_count = sum(1 for question in question_set.questions if question.should_refuse)
        scenarios = [question.metadata.get("scenario_family") for question in question_set.questions]

        self.assertEqual(refusal_count, 50)
        self.assertIn("numeric_unit_baiwan", scenarios)
        self.assertIn("section_boolean_variant", scenarios)
        self.assertIn("metadata_membership_positive", scenarios)
        self.assertIn("metadata_membership_negative", scenarios)

    def test_gen120_dataset_requires_generation_for_each_capability(self):
        dataset_dir = DATA_DIR / "gen120"
        question_set = load_question_set(dataset_dir / "questions.json")
        answer_set = load_gold_answer_set(dataset_dir / "answers_gold.json")
        manifest = load_manifest(dataset_dir / "dataset_manifest.json")
        report = validate_dataset_alignment(question_set, answer_set)

        expected_capabilities = {
            "generation_summary",
            "generation_risk_synthesis",
            "generation_strategy_synthesis",
            "generation_table_text_reasoning",
            "generation_cross_doc_compare",
            "generation_evidence_based_classification",
        }
        capability_counts = {}
        for question in question_set.questions:
            capability_counts[question.capability] = capability_counts.get(question.capability, 0) + 1
            self.assertEqual(question.kind, "text")
            self.assertFalse(question.should_refuse)
            self.assertEqual(question.metadata.get("requires_generation"), True)
            self.assertEqual(question.metadata.get("rules_role"), "evidence_only")

        answers_by_id = {answer.question_id: answer for answer in answer_set.answers}
        cross_doc_questions = [
            question
            for question in question_set.questions
            if question.capability == "generation_cross_doc_compare"
        ]
        for question in cross_doc_questions:
            answer = answers_by_id[question.question_id]
            referenced_doc_ids = {reference.pdf_sha1 for reference in answer.references}
            self.assertTrue(set(question.doc_ids).issubset(referenced_doc_ids))

        self.assertTrue(report["valid"])
        self.assertEqual(len(question_set.questions), 120)
        self.assertEqual(len(answer_set.answers), 120)
        self.assertEqual(manifest.question_count, 120)
        self.assertEqual(manifest.answer_count, 120)
        self.assertEqual(set(capability_counts), expected_capabilities)
        self.assertTrue(all(count == 20 for count in capability_counts.values()))


if __name__ == "__main__":
    unittest.main()
