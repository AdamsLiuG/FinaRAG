from pathlib import Path
import unittest

from eval.dataset_schema import load_gold_answer_set, load_manifest, load_question_set, validate_dataset_alignment


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "finance_eval_benchmark_v1"


class FinanceEvalBenchmarkTests(unittest.TestCase):
    def test_core60_dataset_is_aligned(self):
        question_set = load_question_set(DATA_DIR / "questions.json")
        answer_set = load_gold_answer_set(DATA_DIR / "answers_gold.json")
        manifest = load_manifest(DATA_DIR / "dataset_manifest.json")

        report = validate_dataset_alignment(question_set, answer_set)

        self.assertTrue(report["valid"])
        self.assertEqual(len(question_set.questions), 60)
        self.assertEqual(len(answer_set.answers), 60)
        self.assertEqual(manifest.question_count, 60)
        self.assertEqual(manifest.answer_count, 60)

    def test_dev_and_test_splits_are_balanced(self):
        dev_questions = load_question_set(DATA_DIR / "splits" / "dev" / "questions.json")
        test_questions = load_question_set(DATA_DIR / "splits" / "test" / "questions.json")
        dev_answers = load_gold_answer_set(DATA_DIR / "splits" / "dev" / "answers_gold.json")
        test_answers = load_gold_answer_set(DATA_DIR / "splits" / "test" / "answers_gold.json")

        self.assertEqual(len(dev_questions.questions), 30)
        self.assertEqual(len(test_questions.questions), 30)
        self.assertEqual(len(dev_answers.answers), 30)
        self.assertEqual(len(test_answers.answers), 30)

        dev_ids = {question.question_id for question in dev_questions.questions}
        test_ids = {question.question_id for question in test_questions.questions}
        self.assertEqual(len(dev_ids & test_ids), 0)
        self.assertEqual(len(dev_ids | test_ids), 60)


if __name__ == "__main__":
    unittest.main()
