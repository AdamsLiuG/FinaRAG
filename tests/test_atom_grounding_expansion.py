"""G-4 grounding 扩容定向测试：11 个新增原子的应命中/应否决边界。

词表 provenance: inferred——本测试是这些新知识的唯一机器验收：
正向（行首精确命中得高分）+ 负向（口径错位/门槛不符一律否决）。
MET-004（extracted）的行为由黄金基线等价测试覆盖，此处不重复。
"""

import unittest

from src.table_grounding import _atom_row_score, _reject_unit_by_atom


def score(question, row, col="2024年"):
    return _atom_row_score(question, row, col)


class ExactHitTests(unittest.TestCase):
    """正向：年报标准行首精确命中。"""

    def test_metric_001_koufei(self):
        self.assertEqual(score("宁德时代2024年扣非归母净利润是多少？", "扣除非经常性损益后的净利润"), 8.0)

    def test_metric_002_guimu(self):
        self.assertEqual(score("宁德时代2024年归母净利润是多少？", "归属于母公司股东的净利润"), 8.0)

    def test_metric_003_operating_cashflow(self):
        self.assertEqual(score("宁德时代2024年经营活动现金流量净额是多少？", "经营活动产生的现金流量净额"), 8.0)

    def test_metric_005_cash(self):
        self.assertEqual(score("宁德时代2024年货币资金是多少？", "货币资金"), 8.0)

    def test_metric_006_short_loan(self):
        self.assertEqual(score("宁德时代2024年短期借款是多少？", "短期借款"), 8.0)

    def test_metric_007_total_assets(self):
        self.assertEqual(score("宁德时代2024年总资产是多少？", "资产总计"), 8.0)

    def test_metric_008_total_liabilities(self):
        self.assertEqual(score("宁德时代2024年总负债是多少？", "负债合计"), 8.0)

    def test_metric_009_rd_investment(self):
        self.assertEqual(score("宁德时代2024年研发投入是多少？", "研发投入金额"), 8.0)

    def test_metric_010_gross_margin(self):
        self.assertEqual(score("宁德时代2024年毛利率是多少？", "综合毛利率"), 8.0)

    def test_metric_011_debt_ratio(self):
        self.assertEqual(score("宁德时代2024年资产负债率是多少？", "资产负债率"), 8.0)

    def test_metric_012_rd_expense(self):
        self.assertEqual(score("宁德时代2024年研发费用是多少？", "研发费用"), 8.0)

    def test_secondary_headers_score_below_exact(self):
        self.assertEqual(score("宁德时代2024年归母净利润是多少？", "归母净利润"), 7.5)


class CaliberGuardTests(unittest.TestCase):
    """负向：口径错位行必须否决——这是扩容的主要安全收益。"""

    def test_koufei_question_rejects_plain_guimu_row(self):
        # 问扣非，行是归母（未扣非口径）——最典型的口径错位
        self.assertIsNone(score("宁德时代2024年扣非归母净利润是多少？", "归属于母公司股东的净利润"))

    def test_guimu_question_rejects_koufei_row(self):
        self.assertIsNone(score("宁德时代2024年归母净利润是多少？", "扣除非经常性损益后的净利润"))

    def test_guimu_question_rejects_equity_row(self):
        # 「归属于母公司股东的所有者权益」含「归属于母公司股东」——必须靠 bad_row 拦
        self.assertIsNone(score("宁德时代2024年归母净利润是多少？", "归属于母公司股东的所有者权益"))

    def test_cashflow_question_rejects_investing_row(self):
        self.assertIsNone(score("宁德时代2024年经营活动现金流量净额是多少？", "投资活动产生的现金流量净额"))

    def test_cashflow_question_rejects_inflow_subtotal(self):
        self.assertIsNone(score("宁德时代2024年经营活动现金流量净额是多少？", "经营活动现金流入小计"))

    def test_rd_investment_rejects_capitalized_row(self):
        self.assertIsNone(score("宁德时代2024年研发投入是多少？", "资本化研发投入金额"))

    def test_rd_investment_rejects_ratio_row(self):
        self.assertIsNone(score("宁德时代2024年研发投入是多少？", "研发投入占营业收入比例"))

    def test_rd_expense_question_rejects_investment_row(self):
        # 问研发费用（费用化口径），行是研发投入（含资本化）——口径错位
        self.assertIsNone(score("宁德时代2024年研发费用是多少？", "研发投入金额"))

    def test_rd_investment_question_rejects_expense_row_via_term_gate(self):
        # 问研发投入，行是研发费用——match_terms 门槛拦截
        self.assertIsNone(score("宁德时代2024年研发投入是多少？", "研发费用"))

    def test_total_assets_rejects_current_assets_subtotal(self):
        self.assertIsNone(score("宁德时代2024年总资产是多少？", "流动资产合计"))

    def test_total_liabilities_rejects_equity_total(self):
        self.assertIsNone(score("宁德时代2024年总负债是多少？", "负债和所有者权益总计"))

    def test_cash_rejects_cash_equivalents_row(self):
        self.assertIsNone(score("宁德时代2024年货币资金是多少？", "现金及现金等价物余额"))


class UnitGuardTests(unittest.TestCase):
    def test_percent_metric_with_percent_column_not_killed(self):
        # percent 类指标 + % 列：通用守卫不得误杀（守卫只对 currency 生效）
        self.assertEqual(
            _atom_row_score("宁德时代2024年资产负债率是多少？", "资产负债率", "2024年(%)"), 8.0
        )

    def test_currency_metric_with_percent_column_rejected(self):
        # 货币类指标 + % 列（问题本身不含 %）→ 通用守卫否决
        self.assertIsNone(_atom_row_score("宁德时代2024年归母净利润是多少？", "归属于母公司股东的净利润", "同比(%)"))

    def test_reject_unit_by_atom_covers_all_metrics(self):
        # 单位守卫从营收专属扩展到全部 12 个原子（Stage C 的结构性收益）
        self.assertTrue(_reject_unit_by_atom("宁德时代2024年货币资金是多少？", "占比(%)"))
        self.assertTrue(_reject_unit_by_atom("宁德时代2024年总资产是多少？", "%"))
        self.assertFalse(_reject_unit_by_atom("宁德时代2024年资产负债率是多少？", "%"))


if __name__ == "__main__":
    unittest.main()
