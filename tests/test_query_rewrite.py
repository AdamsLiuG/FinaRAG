import unittest

from src.query_rewrite import QuestionRewriter


class QueryRewriteTests(unittest.TestCase):
    def test_rewrite_extracts_chinese_financial_filters_and_expansions(self):
        rewriter = QuestionRewriter()

        plan = rewriter.rewrite(
            "300750在2024年报里的归母净利润是多少人民币？",
            schema="number",
        )

        self.assertEqual(plan.filters.year, 2024)
        self.assertEqual(plan.filters.currency, "CNY")
        self.assertEqual(plan.filters.doc_source_type, "annual_report")
        self.assertEqual(plan.filters.security_code, "300750")
        self.assertIn("归属于母公司股东的净利润", " ".join(plan.search_queries))
        self.assertEqual(plan.expected_answer_type, "numeric")
        self.assertGreaterEqual(len(plan.search_queries), 2)

    def test_rewrite_extracts_metadata_filters_for_sections_and_tags(self):
        rewriter = QuestionRewriter()

        plan = rewriter.rewrite(
            "科创板半导体行业公司在管理层讨论与分析章节里关于国产替代的表述是什么？",
            schema="name",
        )

        self.assertEqual(plan.filters.board, "科创板")
        self.assertEqual(plan.filters.section_name, "管理层讨论与分析")
        self.assertEqual(plan.filters.industry_l1, "科创板半导体")
        self.assertIn("国产替代", plan.filters.strategy_tags)
        self.assertIn("管理层讨论与分析", " ".join(plan.search_queries))


if __name__ == "__main__":
    unittest.main()
