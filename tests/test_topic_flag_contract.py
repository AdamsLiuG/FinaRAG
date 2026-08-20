import csv
import tempfile
import unittest
from pathlib import Path

from src.query_rewrite import _TOPIC_PATTERNS
from src.text_splitter import TextSplitter
from src.topic_flag_contract import (
    assert_topic_patterns_valid,
    FlagContract,
    coverage_report,
    filter_report_topic_flags,
    get_default_contract,
    load_flag_contract,
    reset_default_contract,
    validate_topic_patterns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TopicFlagContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = get_default_contract()

    def test_contract_covers_all_topic_patterns(self):
        # 回归锁：新增查询侧词表 key 而忘记登记契约时失败
        self.assertEqual(validate_topic_patterns(_TOPIC_PATTERNS, contract=self.contract), [])

    def test_contract_matches_dataset_columns(self):
        # 契约命名空间必须等于数据集表头并集；数据集新增 flag 字段时失败
        dataset_flags = set()
        for subset in sorted(REPO_ROOT.glob("data/*/subset.csv")):
            with subset.open(encoding="utf-8", newline="") as fh:
                header = next(csv.reader(fh), [])
            dataset_flags.update(
                col.strip() for col in header if col.strip().startswith(("has_", "mentions_"))
            )
        if not dataset_flags:
            self.skipTest("no dataset subset.csv available")
        self.assertEqual(self.contract.flags, dataset_flags)

    def test_reserved_field_not_treated_as_flag(self):
        # D-1 回归：内部技术字段不得被当作 topic flag 收集
        report_meta = {
            "has_pdfcrawl_metadata": "true",
            "has_table_context": "true",
            "company_name": "示例公司",
        }
        self.assertEqual(filter_report_topic_flags(report_meta, contract=self.contract), [])

    def test_legit_flag_still_collected(self):
        report_meta = {
            "has_leadership_changes": "true",
            "has_layoffs": "True",
            "has_esg_initiatives": "false",
            "has_pdfcrawl_metadata": "true",
        }
        self.assertEqual(
            filter_report_topic_flags(report_meta, contract=self.contract),
            ["has_layoffs", "has_leadership_changes"],
        )

    def test_text_splitter_uses_contract(self):
        # D-1 端到端：打标侧走契约过滤
        splitter = TextSplitter()
        flags = splitter._report_topic_flags(
            {"has_pdfcrawl_metadata": "true", "has_guidance_updates": "true"}
        )
        self.assertEqual(flags, ["has_guidance_updates"])

    def test_unknown_flag_rejected(self):
        violations = validate_topic_patterns({"has_typo_flag": ["x"]}, contract=self.contract)
        self.assertEqual(len(violations), 1)
        self.assertIn("not declared", violations[0])

    def test_reserved_key_in_patterns_rejected(self):
        violations = validate_topic_patterns({"has_table_context": ["x"]}, contract=self.contract)
        self.assertEqual(len(violations), 1)
        self.assertIn("reserved", violations[0])

    def test_assert_raises_on_violation(self):
        with self.assertRaises(ValueError):
            assert_topic_patterns_valid({"nonsense": ["x"]}, contract=self.contract)

    def test_coverage_report_quantifies_gap(self):
        # G-4c 后全量覆盖：17/17；缺口量化能力用小构造契约验证（保留 D-2 的测试意图）
        report = coverage_report(_TOPIC_PATTERNS, contract=self.contract)
        self.assertEqual(report["total_flags"], 17)
        self.assertEqual(report["covered_flags"], 17)
        self.assertEqual(report["coverage_rate"], 1.0)
        self.assertEqual(report["missing"], [])
        # query_side: true 的 flag 必须都有词表，否则是声明与实现不符
        self.assertEqual(report["declared_query_side_without_patterns"], [])
        # 小构造契约：缺口仍然可度量（防止 coverage_report 退化为永远 100%）
        partial = FlagContract(flags={"a", "b", "c"}, query_side_flags={"a"}, reserved_non_flags=set())
        gap_report = coverage_report({"a": ["x"]}, contract=partial)
        self.assertEqual(gap_report["covered_flags"], 1)
        self.assertEqual(gap_report["missing"], ["b", "c"])

    def test_overlap_between_flag_and_reserved_rejected_at_load(self):
        # 同一名字既声明为 flag 又列为内部保留字段 → 契约自相矛盾，加载期报错
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as fh:
            fh.write(
                "version: '1.0'\n"
                "flags:\n"
                "  - {name: has_leadership_changes, query_side: true}\n"
                "reserved_non_flags: [has_leadership_changes]\n"
            )
            temp_path = Path(fh.name)
        try:
            with self.assertRaises(ValueError):
                load_flag_contract(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_empty_flags_rejected_at_load(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as fh:
            fh.write("version: '1.0'\nflags: []\n")
            temp_path = Path(fh.name)
        try:
            with self.assertRaises(ValueError):
                load_flag_contract(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)


class ContractSingletonTests(unittest.TestCase):
    def tearDown(self):
        reset_default_contract()

    def test_reset_default_contract(self):
        custom = load_flag_contract()
        reset_default_contract(custom)
        self.assertIs(get_default_contract(), custom)


class ActivatedFlagRoutingTests(unittest.TestCase):
    """G-4c：11 类激活词表的路由命中——每类至少一个中文与一个英文问法。"""

    CASES = [
        ("has_layoffs", "公司是否有裁员计划？", "did the company announce layoffs"),
        ("has_executive_compensation", "高管薪酬的披露情况如何？", "executive compensation disclosure"),
        ("has_rnd_investment_numbers", "研发投入金额是多少？", "rnd investment this year"),
        ("has_new_product_launches", "今年有哪些新产品发布？", "new product launches"),
        ("has_capital_expenditures", "资本开支计划如何？", "capex plan"),
        ("has_financial_performance_indicators", "主要财务指标表现如何？", "financial performance"),
        ("has_capital_structure_changes", "资本结构有什么变动？", "capital structure changes"),
        ("mentions_new_risk_factors", "年报提到哪些新增风险？", "risk factors"),
        ("has_strategic_restructuring", "公司是否涉及资产重组？", "strategic restructuring"),
        ("has_supply_chain_disruptions", "供应链是否受到影响？", "supply chain disruption"),
        ("has_esg_initiatives", "碳中和目标进展如何？", "esg initiatives"),
    ]

    def test_each_activated_flag_routable(self):
        from src.query_rewrite import _TOPIC_PATTERNS

        for flag, zh_query, en_query in self.CASES:
            for query in (zh_query, en_query):
                matched = any(p in query.lower() for p in _TOPIC_PATTERNS[flag])
                self.assertTrue(matched, msg=f"{flag}: {query!r} 未命中词表")

    def test_unrelated_query_triggers_no_flag(self):
        from src.query_rewrite import QuestionRewriter

        rewriter = QuestionRewriter()
        plan = rewriter.rewrite("宁德时代2024年归母净利润是多少？", schema="number", company_name="宁德时代")
        self.assertEqual(plan.topic_flags, [])

    def test_metric_query_with_flag_coexists(self):
        # 口径查询与话题过滤正交：研发投入问题同时挂 MET-009 原子与研发 flag
        from src.query_rewrite import QuestionRewriter

        plan = QuestionRewriter().rewrite(
            "宁德时代2024年研发投入是多少？", schema="number", company_name="宁德时代"
        )
        self.assertIn("has_rnd_investment_numbers", plan.topic_flags)
        self.assertEqual(plan.metric_atom["atom_id"], "MET-009")


if __name__ == "__main__":
    unittest.main()
