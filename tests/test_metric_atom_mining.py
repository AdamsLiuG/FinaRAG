import unittest

from src.metric_atom_mining import (
    render_draft_yaml,
    validate_atom_draft,
    validate_atom_drafts,
)
from src.metric_atoms import MetricAtomStore


class AtomMiningValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = MetricAtomStore.load()

    def test_validate_accepts_valid_draft(self):
        draft = {
            "standard_name": "存货周转率",
            "aliases": ["存货周转率", "库存周转率"],
            "definition": "营业成本与平均存货余额之比",
            "caliber_notes": "分母为平均存货",
            "unit": "次",
            "statement_location": "主要财务指标",
        }
        self.assertEqual(validate_atom_draft(draft, self.store), [])

    def test_validate_rejects_bare_caliber_terms(self):
        # D1 同源原则：裸"收入/利润/费用/现金流"这类口径放宽词禁止作为 alias
        draft = {
            "standard_name": "利息收入",
            "aliases": ["利息收入", "收入"],
            "definition": "银行存款与债券投资产生的利息",
        }
        errors = validate_atom_draft(draft, self.store)
        self.assertTrue(any("口径放宽词" in e for e in errors))

    def test_validate_rejects_overlap_with_existing_atom(self):
        draft = {
            "standard_name": "归母净利润",
            "aliases": ["归母净利润"],
            "definition": "重复指标",
        }
        errors = validate_atom_draft(draft, self.store)
        self.assertTrue(any("口径重叠" in e for e in errors))

    def test_validate_rejects_missing_fields(self):
        errors = validate_atom_draft({"aliases": ["某指标"]}, self.store)
        self.assertTrue(any("standard_name" in e for e in errors))
        self.assertTrue(any("definition" in e for e in errors))

    def test_validate_drafts_detects_duplicates(self):
        drafts = [
            {"standard_name": "流动比率", "aliases": ["流动比率"], "definition": "流动资产/流动负债"},
            {"standard_name": "流动比率", "aliases": ["流动比率"], "definition": "重复"},
        ]
        results = validate_atom_drafts(drafts, self.store)
        self.assertTrue(any("重复" in e for e in results["流动比率"]))

    def test_render_draft_yaml_contains_accepted_only(self):
        drafts = [
            {"standard_name": "速动比率", "aliases": ["速动比率"], "definition": "(流动资产-存货)/流动负债",
             "unit": "倍", "statement_location": "主要财务指标"},
            {"standard_name": "归母净利润", "aliases": ["归母净利润"], "definition": "与现有原子重叠"},
        ]
        accepted = ["速动比率"]
        rendered = render_draft_yaml(drafts, accepted=accepted)
        self.assertIn("速动比率", rendered)
        self.assertNotIn("DRAFT-002", rendered)


if __name__ == "__main__":
    unittest.main()
