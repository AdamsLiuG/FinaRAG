import unittest

from eval.export_finance_eval_bundle import _normalize_reference
from src.questions_processing import QuestionsProcessor


class ReferenceValidationTests(unittest.TestCase):
    def test_validate_page_references_drops_hallucinated_pages_and_backfills_top_hits(self):
        processor = QuestionsProcessor()
        retrieval_results = [
            {"page": 3, "text": "Revenue was 100.", "distance": 0.9},
            {"page": 5, "text": "Operating margin was 10%.", "distance": 0.8},
        ]

        validated = processor._validate_page_references([9, 3], retrieval_results, min_pages=2, max_pages=8)
        self.assertEqual(validated, [3, 5])

    def test_internal_page_references_convert_only_at_submission_boundary(self):
        processor = QuestionsProcessor()
        processed_questions = [
            {
                "question_text": "甲公司营业收入是多少？",
                "kind": "number",
                "value": 100,
                "references": [{"pdf_sha1": "doc-alpha", "page": 12}],
                "citations": [{"source": "doc-alpha", "page": 12}],
                "confidence": "high",
                "answer_details": {"$ref": "#/answer_details/0"},
            }
        ]

        submission_answers = processor._post_process_submission_answers(processed_questions)

        self.assertEqual(submission_answers[0]["references"], [{"pdf_sha1": "doc-alpha", "page_index": 11}])

    def test_export_reference_normalizes_internal_page_before_legacy_page_index(self):
        self.assertEqual(
            _normalize_reference({"pdf_sha1": "doc-alpha", "page": 12, "page_index": 99}),
            {"pdf_sha1": "doc-alpha", "page_index": 11},
        )
        self.assertEqual(
            _normalize_reference({"pdf_sha1": "doc-alpha", "page_index": 11}),
            {"pdf_sha1": "doc-alpha", "page_index": 11},
        )


if __name__ == "__main__":
    unittest.main()
