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
            "matched_tags": ["国产替代"],
            "matched_queries": ["毛利率是多少", "综合毛利率是多少"],
            "query_hit_count": 2,
            "retrieval_sources": ["vector"],
            "metadata": {
                "chunk_id": 12,
                "chunk_type": "content",
                "node_type": "parent",
                "parent_chunk_id": None,
                "section_title": "Margins",
                "section_name": "管理层讨论与分析",
                "report_section": "Margins",
                "company_name": "Alpha Corp",
                "stock_code": "600000",
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
        self.assertEqual(serialized["section_name"], "管理层讨论与分析")
        self.assertEqual(serialized["matched_tags"], ["国产替代"])
        self.assertEqual(serialized["matched_queries"], ["毛利率是多少", "综合毛利率是多少"])
        self.assertEqual(serialized["query_hit_count"], 2)
        self.assertEqual(serialized["final_score"], 0.88)


if __name__ == "__main__":
    unittest.main()

