import unittest

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


if __name__ == "__main__":
    unittest.main()
