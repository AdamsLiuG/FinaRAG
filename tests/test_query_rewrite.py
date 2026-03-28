import unittest

from src.query_rewrite import QuestionRewriter


class QueryRewriteTests(unittest.TestCase):
    def test_rewrite_extracts_financial_topic_flags_and_filters(self):
        rewriter = QuestionRewriter()

        plan = rewriter.rewrite(
            "Did the company mention any mergers or acquisitions in 2023 and report it in USD?",
            schema="boolean",
        )

        self.assertEqual(plan.filters.year, 2023)
        self.assertEqual(plan.filters.currency, "USD")
        self.assertIn("mentions_recent_mergers_and_acquisitions", plan.topic_flags)
        self.assertEqual(plan.expected_answer_type, "boolean")
        self.assertGreaterEqual(len(plan.search_queries), 2)


if __name__ == "__main__":
    unittest.main()
