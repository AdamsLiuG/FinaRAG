import unittest

from eval.metrics import compare_ranked_retrieval


class RankedRetrievalMetricsTests(unittest.TestCase):
    def test_compare_ranked_retrieval_deduplicates_pages_and_computes_macro_scores(self):
        pred_answers = [
            {"question_text": "q1", "kind": "boolean"},
            {"question_text": "q2", "kind": "number"},
            {"question_text": "q3", "kind": "name"},
        ]
        ref_answers = [
            {
                "question_text": "q1",
                "references": [
                    {"page_index": 0},
                    {"page_index": 2},
                ],
            },
            {
                "question_text": "q2",
                "references": [
                    {"page_index": 3},
                ],
            },
            {
                "question_text": "q3",
                "references": [],
            },
        ]
        debug_payload = {
            "questions": [
                {"question_text": "q1"},
                {"question_text": "q2"},
                {"question_text": "q3"},
            ],
            "answer_details": [
                {
                    "retrieval_results": [
                        {"page": 3},
                        {"page": 3},
                        {"page": 8},
                        {"page": 1},
                    ]
                },
                {
                    "retrieval_pages": [2, 4],
                },
                {
                    "retrieval_results": [
                        {"page": 9},
                    ]
                },
            ],
        }

        report = compare_ranked_retrieval(
            pred_answers,
            ref_answers,
            debug_payload=debug_payload,
            recall_k=10,
            precision_k=3,
        )

        self.assertEqual(report["evaluation_level"], "page")
        self.assertEqual(report["eligible_questions"], 2)
        self.assertEqual(report["questions_without_reference_pages"], 1)
        self.assertEqual(report["questions_with_fewer_than_recall_k_ranked_pages"], 2)
        self.assertEqual(report["questions_with_fewer_than_precision_k_ranked_pages"], 1)
        self.assertEqual(report["macro_recall_at_10"], 1.0)
        self.assertEqual(report["macro_precision_at_3"], 0.5)
        self.assertEqual(report["hit_at_10"], 1.0)

        first_question = report["question_details"][0]
        self.assertEqual(first_question["ranked_pages"], [3, 8, 1])
        self.assertEqual(first_question["recall_hits"], [1, 3])
        self.assertEqual(first_question["precision_hits"], [1, 3])


if __name__ == "__main__":
    unittest.main()
