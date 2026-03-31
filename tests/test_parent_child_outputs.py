import unittest

from src.citation_formatter import build_citations
from src.questions_processing import QuestionsProcessor


class ParentChildOutputTests(unittest.TestCase):
    def test_citations_and_retrieval_debug_include_parent_child_fields(self):
        retrieval_result = {
            "page": 4,
            "text": "Parent operating margin block",
            "distance": 0.88,
            "chunk_id": 12,
            "chunk_type": "content",
            "result_scope": "parent",
            "matched_child_chunk_ids": [3, 4],
            "retrieval_sources": ["vector"],
            "metadata": {
                "chunk_id": 12,
                "chunk_type": "content",
                "node_type": "parent",
                "parent_chunk_id": None,
                "section_title": "Margins",
                "report_section": "Margins",
                "company_name": "Alpha Corp",
                "currency": "USD",
                "report_year": 2023,
                "report_type": "annual",
                "major_industry": "Industrials",
                "topic_flags": [],
                "parent_block_id": "page4_block0",
                "sha1_name": "alpha-sha",
            },
        }

        citations = build_citations([retrieval_result], [4])
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["node_type"], "parent")
        self.assertEqual(citations[0]["matched_child_chunk_ids"], [3, 4])

        processor = QuestionsProcessor()
        serialized = processor._serialize_retrieval_result(retrieval_result)
        self.assertEqual(serialized["node_type"], "parent")
        self.assertEqual(serialized["matched_child_chunk_ids"], [3, 4])
        self.assertEqual(serialized["result_scope"], "parent")


if __name__ == "__main__":
    unittest.main()
