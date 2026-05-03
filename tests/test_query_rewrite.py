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

    def test_rewrite_does_not_treat_st_prefix_as_security_code(self):
        rewriter = QuestionRewriter()

        plan = rewriter.rewrite(
            "ST曙光2024年年报中的法定代表人是谁？",
            schema="name",
        )

        self.assertIsNone(plan.filters.security_code)
        self.assertEqual(plan.route_hints["security_codes"], [])

    def test_rewrite_normalizes_explicit_exchange_security_code(self):
        rewriter = QuestionRewriter()

        plan = rewriter.rewrite(
            "SH600000在2024年年报里的营业收入是多少？",
            schema="number",
        )

        self.assertEqual(plan.filters.security_code, "600000")
        self.assertEqual(plan.route_hints["security_codes"], ["600000"])

        bj_plan = rewriter.rewrite("BJ430047在2024年报里的营业收入是多少？", schema="number")
        self.assertEqual(bj_plan.filters.security_code, "430047")

    def test_rewrite_cleans_industry_prefix_and_long_industry_name(self):
        rewriter = QuestionRewriter()

        auto_plan = rewriter.rewrite(
            "在汽车行业的科创板公司中，2024年年报里提到“国产替代”的公司共有几家？",
            schema="number",
        )
        power_plan = rewriter.rewrite(
            "在电力、热力生产和供应业行业的公司中，2024年年报里提到“绿色转型”的公司共有几家？",
            schema="number",
        )

        self.assertEqual(auto_plan.filters.industry_l1, "汽车")
        self.assertEqual(power_plan.filters.industry_l1, "电力、热力生产和供应业")


if __name__ == "__main__":
    unittest.main()
