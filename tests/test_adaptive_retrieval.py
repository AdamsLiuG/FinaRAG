"""Agentic 自适应检索 v1 测试：缺口判定（知识驱动）+ 合并 + try 语义 + 开关/预算。"""
import os
import unittest
from types import SimpleNamespace

from src.adaptive_retrieval import (
    ADAPTIVE_MIN_TOP_SCORE,
    RetrievalGap,
    adaptive_max_rounds,
    adaptive_retrieval_enabled,
    detect_gap,
    merge_supplement_results,
)
from src.metric_atoms import get_default_store


def _plan(metric_atom=None, playbook=None, task_type="single_metric_lookup"):
    return SimpleNamespace(
        metric_atom=metric_atom,
        playbook=playbook,
        task_type=task_type,
    )


def _met001_payload():
    store = get_default_store()
    atom = store.get("MET-001")
    return {
        "atom_id": "MET-001",
        "standard_name": atom.standard_name,
        "confusable_names": ["归母净利润"],  # MET-002 标准名
    }


def _results(texts):
    return [
        {"page": i + 1, "text": text, "pdf_sha1": "doc_a", "combined_score": 0.5}
        for i, text in enumerate(texts)
    ]


class TestDetectGap(unittest.TestCase):
    def test_t1_evidence_gap_triggered_by_atom(self):
        """扣非原子命中但证据全是归母口径 → evidence_gap；补检查询含扣非目标、
        不含易混口径标准名（归母净利润——防强化错口径）。"""
        plan = _plan(metric_atom=_met001_payload())
        results = _results(["归母净利润为 100 亿元", "归属于母公司股东的净利润"])
        gap = detect_gap(plan, results, top_score=0.8)
        self.assertIsNotNone(gap)
        self.assertEqual(gap.trigger, "evidence_gap")
        self.assertTrue(gap.supplement_queries)
        # 归母净利润（易混口径标准名）不得作为独立补检查询；
        # 注意「扣非归母净利润」包含其子串——必须用成员判断而非子串判断
        self.assertNotIn("归母净利润", gap.supplement_queries)
        self.assertIn("扣除非经常性损益后的净利润", gap.supplement_queries)

    def test_t2_no_gap_zero_cost(self):
        """证据已覆盖 evidence_targets + 置信度足够 → None（零补检成本）。"""
        plan = _plan(metric_atom=_met001_payload())
        results = _results(["扣非归母净利润为 5 亿元，扣除非经常性损益后的净利润"])
        self.assertIsNone(detect_gap(plan, results, top_score=0.8))

    def test_t3_playbook_incomplete(self):
        """剧本必需指标（货币资金）证据未覆盖 → 逐指标补检。"""
        playbook_payload = {
            "playbook_id": "PLB-002",
            "required_metrics": ["MET-005"],
        }
        plan = _plan(playbook=playbook_payload)
        results = _results(["经营活动现金流分析"])
        gap = detect_gap(plan, results, top_score=0.9)
        self.assertIsNotNone(gap)
        self.assertEqual(gap.trigger, "playbook_incomplete")
        self.assertIn("货币资金", gap.supplement_queries)

    def test_t4_low_confidence(self):
        """top_score 低于阈值且无原子 → low_confidence（无定向查询）。"""
        plan = _plan()
        results = _results(["无关文本"])
        gap = detect_gap(plan, results, top_score=0.05)
        self.assertIsNotNone(gap)
        self.assertEqual(gap.trigger, "low_confidence")
        self.assertEqual(gap.supplement_queries, [])

    def test_low_confidence_skipped_without_results(self):
        """检索结果为空时 low_confidence 不触发（空结果由上游路径处理）。"""
        plan = _plan()
        self.assertIsNone(detect_gap(plan, [], top_score=0.05))

    def test_evidence_gap_takes_priority_over_playbook(self):
        """优先级：evidence_gap > playbook_incomplete。"""
        plan = _plan(
            metric_atom=_met001_payload(),
            playbook={"playbook_id": "PLB-002", "required_metrics": ["MET-005"]},
        )
        results = _results(["只有归母净利润的内容"])
        gap = detect_gap(plan, results, top_score=0.9)
        self.assertEqual(gap.trigger, "evidence_gap")


class TestTrySemantics(unittest.TestCase):
    def test_t6_gap_closed_after_supplement(self):
        """try 语义（模块级）：补检结果合并后缺口闭合——再判 detect_gap 为 None。"""
        plan = _plan(metric_atom=_met001_payload())
        base = _results(["归母净利润为 100 亿元"])
        gap = detect_gap(plan, base, top_score=0.8)
        self.assertIsNotNone(gap)
        # 补检索回扣非证据（不同文档页，避免被 (sha, page) 去重）
        supplement = [{
            "page": 7, "pdf_sha1": "doc_a",
            "text": "扣非归母净利润为 5 亿元，扣除非经常性损益后的净利润 4.8 亿",
            "combined_score": 0.6,
        }]
        merged = merge_supplement_results(base, supplement)
        self.assertIsNone(detect_gap(plan, merged, top_score=0.8))


class TestMergeSupplementResults(unittest.TestCase):
    def test_t5_merge_dedupes_and_marks_adaptive(self):
        base = [{"pdf_sha1": "doc_a", "page": 1, "retrieval_sources": ["vector"]}]
        supplement = [
            {"pdf_sha1": "doc_a", "page": 1, "retrieval_sources": ["bm25"]},  # 重复 → 丢弃
            {"pdf_sha1": "doc_a", "page": 7, "retrieval_sources": ["vector"]},  # 新 → 标记
        ]
        merged = merge_supplement_results(base, supplement)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["retrieval_sources"], ["vector"])
        self.assertIn("adaptive", merged[1]["retrieval_sources"])
        self.assertEqual(merged[1]["page"], 7)


class TestSwitchAndBudget(unittest.TestCase):
    def test_t9_disabled_by_default(self):
        """默认关闭：未设 env → False（回归零影响）。"""
        with os.popen("true"):  # 不动真实 env
            pass
        self.assertFalse(adaptive_retrieval_enabled())

    def test_t9b_enabled_by_env(self):
        env = dict(os.environ)
        env["ADAPTIVE_RETRIEVAL_ENABLED"] = "1"
        os.environ.clear()
        os.environ.update(env)
        try:
            self.assertTrue(adaptive_retrieval_enabled())
        finally:
            os.environ.clear()
            os.environ.update({k: v for k, v in env.items() if k != "ADAPTIVE_RETRIEVAL_ENABLED"})

    def test_t8_budget_single_round_default(self):
        """预算上限默认 1（防循环）。"""
        self.assertEqual(adaptive_max_rounds(), 1)

    def test_min_top_score_default(self):
        self.assertEqual(ADAPTIVE_MIN_TOP_SCORE, 0.15)


if __name__ == "__main__":
    unittest.main()
