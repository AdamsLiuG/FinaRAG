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

    def test_rewrite_builds_metric_aggregation_intent_plan(self):
        rewriter = QuestionRewriter()

        plan = rewriter.rewrite(
            "2024年新能源行业营收最高的前10家公司有哪些？",
            schema="names",
        )

        self.assertEqual(plan.task_type, "metric_aggregation")
        self.assertEqual(plan.route, "metric_fact_store")
        self.assertEqual(plan.operation, "top_k")
        self.assertEqual(plan.metric, "营业收入")
        self.assertEqual(plan.filters.year, 2024)
        self.assertEqual(plan.filters.industry_l1, "新能源")
        self.assertEqual(plan.limit, 10)
        self.assertEqual(plan.sort_order, "desc")
        self.assertEqual(plan.expected_answer_type, "entity_list_with_values")
        self.assertIn("主营业务收入", plan.evidence_targets)
        self.assertEqual(plan.fallback_route, "table_grounding_then_rag")
        self.assertGreaterEqual(plan.intent_confidence, 0.85)
        self.assertEqual(plan.route_hints["task_type"], "metric_aggregation")
        self.assertEqual(plan.route_hints["operation"], "top_k")
        self.assertEqual(plan.route_hints["metric"], "营业收入")

    def test_rewrite_turns_implicit_liquidity_question_into_evidence_queries(self):
        rewriter = QuestionRewriter()

        plan = rewriter.rewrite(
            "宁德时代2024年现金压力大不大？",
            schema="boolean",
            company_name="宁德时代",
        )

        self.assertEqual(plan.task_type, "implicit_intent_qa")
        self.assertEqual(plan.route, "metric_fact_store_plus_rag")
        self.assertEqual(plan.operation, "assess")
        self.assertEqual(plan.intent, "liquidity_pressure")
        self.assertIn("经营活动现金流量净额", plan.evidence_targets)
        self.assertIn("货币资金", plan.evidence_targets)
        self.assertIn("短期借款", " ".join(plan.search_queries))
        self.assertIn("偿债能力", " ".join(plan.search_queries))
        self.assertEqual(plan.route_hints["implicit_intent"], "liquidity_pressure")


if __name__ == "__main__":
    unittest.main()
