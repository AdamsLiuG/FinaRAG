import unittest

from src.metric_atoms import MetricAtomStore, get_default_store, reset_default_store


class MetricAtomStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = get_default_store()

    def test_load_default_atoms(self):
        self.assertGreaterEqual(len(self.store), 12)
        ids = [atom.atom_id for atom in self.store.all_atoms()]
        self.assertEqual(len(ids), len(set(ids)))
        for atom in self.store.all_atoms():
            self.assertTrue(atom.standard_name)
            self.assertTrue(atom.aliases)
            self.assertTrue(atom.definition)
            self.assertTrue(atom.evidence_targets)

    def test_longest_match_wins(self):
        # D3 回归：扣非归母净利润 ⊃ 归母净利润（子串包含），长词必须优先命中，不依赖配置顺序
        atom = self.store.find_metric("某公司扣非归母净利润是多少")
        self.assertIsNotNone(atom)
        self.assertEqual(atom.standard_name, "扣非归母净利润")

    def test_alias_matching(self):
        self.assertEqual(self.store.find_metric("归母").standard_name, "归母净利润")
        self.assertEqual(
            self.store.find_metric("归属于母公司股东的净利润同比").standard_name,
            "归母净利润",
        )
        # 无"量"字变体兼容（承接原 _TERM_EXPANSIONS 的扩展行为）
        self.assertEqual(
            self.store.find_metric("经营活动现金流净额").standard_name,
            "经营活动现金流量净额",
        )

    def test_caliber_separation(self):
        # D2 回归：研发费用（费用化）与研发投入（费用化+资本化）是不同口径，互不误触
        self.assertEqual(self.store.find_metric("某公司研发费用").standard_name, "研发费用")
        self.assertEqual(self.store.find_metric("某公司研发投入").standard_name, "研发投入")

    def test_confusable_references_valid(self):
        for atom in self.store.all_atoms():
            for ref in atom.confusable_with:
                self.assertIsNotNone(self.store.get(ref), f"{atom.atom_id} -> {ref}")

    def test_no_caliber_pollution_in_aliases(self):
        # D1 回归（数据层）：aliases 禁止口径放宽词——裸"收入"会把营业收入的扩展命中到其他业务收入口径
        for atom in self.store.all_atoms():
            self.assertNotIn(
                "收入",
                [a for a in atom.aliases if a not in ("营业收入", "主营业务收入")],
            )

    def test_find_metric_miss(self):
        self.assertIsNone(self.store.find_metric("公司的法定代表人是谁"))

    def test_match_terms_include_standard_name(self):
        # 研发投入的 aliases 不含 standard_name 时，match_terms 仍须覆盖它
        atom = self.store.get("K1-MET-009")
        self.assertIn("研发投入", atom.match_terms)


class DefaultStoreTests(unittest.TestCase):
    def tearDown(self):
        reset_default_store()

    def test_reset_default_store(self):
        custom = MetricAtomStore.load()
        reset_default_store(custom)
        self.assertIs(get_default_store(), custom)


if __name__ == "__main__":
    unittest.main()
