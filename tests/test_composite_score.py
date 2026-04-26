import unittest

from eval.composite_score import compute_finance_case_score, get_finance_scoring_profile


class CompositeScoreTests(unittest.TestCase):
    def test_ragas_metrics_are_assigned_to_retrieval_and_citation_layers(self):
        result = compute_finance_case_score(
            semantic_result={"semantic_score": 1.0},
            entity_result={"entity_score": 1.0},
            keyword_result={"keyword_score": 1.0},
            ragas_result={
                "ragas_score": 0.0,
                "context_recall": 0.0,
                "context_precision": 0.0,
                "faithfulness": 0.0,
            },
            retrieval_result={
                "retrieval_score": 1.0,
                "doc_hit": 1.0,
                "page_hit": 1.0,
                "page_recall": 1.0,
                "table_hit": None,
            },
            citation_result={
                "citation_score": 1.0,
                "citation_page_hit": 1.0,
                "citation_precision": 1.0,
                "citation_coverage": 1.0,
            },
        )

        self.assertEqual(result["answer_score"], 1.0)
        self.assertLess(result["retrieval_score"], 1.0)
        self.assertLess(result["citation_score"], 1.0)
        self.assertEqual(result["ragas_score"], 0.0)

    def test_scoring_profile_names_atomic_answer_layers(self):
        profile = get_finance_scoring_profile()

        self.assertEqual(profile["profile_name"], "finance_atomic_answer_rag_three_layer_v3")
        self.assertIn("layer_1_type_aware_answer_value", profile["layers"])
        self.assertIn("layer_2_evidence_retrieval_quality", profile["layers"])
        self.assertIn("layer_3_citation_grounding", profile["layers"])
        self.assertNotIn("ragas_score", profile["layers"]["layer_1_type_aware_answer_value"])


if __name__ == "__main__":
    unittest.main()
