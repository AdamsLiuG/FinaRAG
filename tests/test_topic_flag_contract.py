import csv
import tempfile
import unittest
from pathlib import Path

from src.query_rewrite import _TOPIC_PATTERNS
from src.text_splitter import TextSplitter
from src.topic_flag_contract import (
    assert_topic_patterns_valid,
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
        # D-2 量化：数据集 17 个 flag，查询侧只覆盖 6 个
        report = coverage_report(_TOPIC_PATTERNS, contract=self.contract)
        self.assertEqual(report["total_flags"], 17)
        self.assertEqual(report["covered_flags"], 6)
        self.assertIn("has_layoffs", report["missing"])
        self.assertIn("has_esg_initiatives", report["missing"])
        # query_side: true 的 flag 必须都有词表，否则是声明与实现不符
        self.assertEqual(report["declared_query_side_without_patterns"], [])

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


if __name__ == "__main__":
    unittest.main()
