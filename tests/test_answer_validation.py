import unittest

from src.answer_validation import validate_answer
from src.query_rewrite import QuestionRewriter


class AnswerValidationTests(unittest.TestCase):
    def test_currency_mismatch_forces_refusal(self):
        query_plan = QuestionRewriter().rewrite(
            "What was Alpha Corp revenue in 2023 in USD?",
            schema="number",
            company_name="Alpha Corp",
        )
        answer_dict = {
            "final_answer": 1200,
            "confidence": "high",
            "relevant_pages": [8],
            "references": [{"pdf_sha1": "sha1-alpha", "page_index": 8}],
            "citations": [{"page": 8, "chunk_type": "serialized_table"}],
        }
        retrieval_results = [
            {
                "page": 8,
                "distance": 0.9,
                "metadata": {
                    "currency": "EUR",
                    "report_year": 2023,
                    "topic_flags": [],
                },
            }
        ]

        validated = validate_answer(answer_dict, retrieval_results, query_plan)

        self.assertEqual(validated.answer["final_answer"], "N/A")
        self.assertEqual(validated.confidence, "low")
        self.assertIn("currency_mismatch", validated.validation_flags)


if __name__ == "__main__":
    unittest.main()
