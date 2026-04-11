import unittest

from eval.semantic_metrics import numeric_similarity_score, score_semantic_similarity, token_f1_score


class SemanticMetricsTests(unittest.TestCase):
    def test_numeric_similarity_ignores_formatting_noise(self):
        score = numeric_similarity_score("356,778,998.12元", "356778998.12")

        self.assertEqual(score, 1.0)

    def test_semantic_similarity_uses_numeric_backend_for_number_questions(self):
        report = score_semantic_similarity("356778998.12", "356,778,998.12", kind="number")

        self.assertEqual(report["backend"], "numeric")
        self.assertEqual(report["semantic_score"], 1.0)

    def test_token_f1_handles_chinese_text(self):
        score = token_f1_score("法定代表人 王奔", "王奔")

        self.assertGreaterEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
