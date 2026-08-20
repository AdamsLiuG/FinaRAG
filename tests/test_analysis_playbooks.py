import unittest

from src.analysis_playbooks import (
    AnalysisPlaybook,
    PlaybookStore,
    build_playbook_hint,
    build_playbook_payload,
    get_default_playbook_store,
    reset_default_playbook_store,
)
from src.metric_atoms import get_default_store


class PlaybookStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = get_default_playbook_store()

    def test_load_default_playbooks(self):
        self.assertGreaterEqual(len(self.store), 5)
        ids = [playbook.playbook_id for playbook in self.store.all_playbooks()]
        intents = [playbook.intent for playbook in self.store.all_playbooks()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(intents), len(set(intents)))
        for playbook in self.store.all_playbooks():
            self.assertTrue(playbook.question_forms)
            self.assertTrue(playbook.judgment)

    def test_required_metrics_reference_valid_atoms(self):
        # 结构性约束：剧本必须引用合法指标原子，分析方法不得脱离事实基础
        atom_store = get_default_store()
        for playbook in self.store.all_playbooks():
            for atom_id in playbook.required_metrics:
                self.assertIsNotNone(atom_store.get(atom_id), f"{playbook.playbook_id} -> {atom_id}")

    def test_invalid_metric_reference_fails_at_load(self):
        broken = AnalysisPlaybook(
            playbook_id="PLB-999",
            intent="broken_intent",
            question_forms=["测试问法"],
            required_metrics=["MET-999"],
            judgment="测试",
        )
        with self.assertRaises(ValueError):
            PlaybookStore([broken])

    def test_playbook_without_any_evidence_source_fails(self):
        empty = AnalysisPlaybook(
            playbook_id="PLB-998",
            intent="empty_intent",
            question_forms=["空剧本"],
            required_metrics=[],
            evidence_keywords=[],
        )
        with self.assertRaises(ValueError):
            PlaybookStore([empty])

    def test_find_playbook_by_question_form(self):
        playbook = self.store.find_playbook("宁德时代2024年现金压力大不大")
        self.assertIsNotNone(playbook)
        self.assertEqual(playbook.intent, "liquidity_pressure")
        self.assertIsNone(self.store.find_playbook("法定代表人是谁"))

    def test_dual_granularity_expansion(self):
        # 检索粒度取标准名（精确简短），校验粒度取完整 evidence_targets（宽松覆盖别名）
        playbook = self.store.get_by_intent("liquidity_pressure")
        search_targets = self.store.search_targets(playbook)
        validation_targets = self.store.validation_targets(playbook)
        self.assertEqual(
            search_targets,
            ["经营活动现金流量净额", "货币资金", "短期借款", "资产负债率", "偿债能力"],
        )
        self.assertLess(len(search_targets), len(validation_targets))
        self.assertIn("现金流量表", validation_targets)
        self.assertIn("一年内到期的非流动负债", validation_targets)

    def test_metric_coverage(self):
        playbook = self.store.get_by_intent("liquidity_pressure")
        full_blob = "货币资金 短期借款 经营活动现金流量净额 资产负债率"
        hits, total = self.store.metric_coverage(playbook, full_blob)
        self.assertEqual((hits, total), (4, 4))
        partial_hits, partial_total = self.store.metric_coverage(playbook, "货币资金充裕")
        self.assertEqual((partial_hits, partial_total), (1, 4))
        none_hits, _ = self.store.metric_coverage(playbook, "管理层讨论了市场需求")
        self.assertEqual(none_hits, 0)

    def test_build_playbook_hint(self):
        playbook = self.store.get_by_intent("liquidity_pressure")
        payload = build_playbook_payload(playbook, self.store)
        hint = build_playbook_hint(payload)
        self.assertIn("【分析判断依据】", hint)
        self.assertIn("货币资金", hint)
        self.assertIn("判断标准", hint)
        self.assertIn("显式说明缺失项", hint)
        self.assertIsNone(build_playbook_hint({}))


class DefaultPlaybookStoreTests(unittest.TestCase):
    def tearDown(self):
        reset_default_playbook_store()

    def test_reset_default_store(self):
        custom = PlaybookStore.load()
        reset_default_playbook_store(custom)
        self.assertIs(get_default_playbook_store(), custom)


class DividendKeywordSingleSourceTests(unittest.TestCase):
    """G-4 单一真源：分红表述只住 PLB-001，代码侧消费必须与剧本一致（防再漂移）。"""

    def test_questions_processing_consumes_playbook_keywords(self):
        from src.questions_processing import _dividend_mention_terms

        playbook = get_default_playbook_store().get("PLB-001")
        self.assertIsNotNone(playbook)
        self.assertEqual(set(_dividend_mention_terms()), set(playbook.cash_dividend_mentions))

    def test_playbook_covers_legacy_code_vocab(self):
        # 收编证据：原 questions_processing._DIVIDEND_MENTION_TERMS 的 9 个词全部在剧本里
        legacy = {"现金分红", "现金红利", "现金股利", "派发股息", "派发现金红利",
                  "每10股派发", "每 10 股派发", "每10股派息", "每 10 股派息"}
        mentions = set(get_default_playbook_store().get("PLB-001").cash_dividend_mentions)
        self.assertTrue(legacy <= mentions)
        # 语义边界：预案话题词不得混入事实表述词表
        self.assertNotIn("利润分配预案", mentions)
        self.assertNotIn("股份回购", mentions)


if __name__ == "__main__":
    unittest.main()
