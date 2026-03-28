import unittest

from src.retrieval_filters import RetrievalFilters, apply_retrieval_filters


class RetrievalFilterTests(unittest.TestCase):
    def test_required_topic_flags_filter_and_boolean_bonus(self):
        results = [
            {
                "distance": 0.3,
                "metadata": {
                    "chunk_type": "content",
                    "topic_flags": ["mentions_recent_mergers_and_acquisitions"],
                },
            },
            {
                "distance": 0.9,
                "metadata": {
                    "chunk_type": "content",
                    "topic_flags": [],
                },
            },
        ]

        filtered = apply_retrieval_filters(
            results,
            RetrievalFilters(
                required_topic_flags=["mentions_recent_mergers_and_acquisitions"],
                question_kind="boolean",
            ),
        )

        self.assertEqual(len(filtered), 1)
        self.assertGreater(filtered[0]["ranking_score"], filtered[0]["distance"])


if __name__ == "__main__":
    unittest.main()
