"""④ trace 反哺知识层测试：统计 / 三条建议规则 / 脏行防御 / 报告渲染。"""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_adaptive_trace import (
    alias_expansion_suggestions,
    load_trace,
    per_atom_stats,
    render_report,
)
from src.metric_atoms import get_default_store


def _entry(atom_id="MET-001", queries=None, new_hits=0, gap_closed=False,
           trigger="evidence_gap", strategy="dense_recall", outcome=None):
    return {
        "question": "q", "company_name": "c",
        "atom_id": atom_id, "trigger": trigger, "strategy": strategy,
        "queries": queries or [], "new_hits": new_hits,
        "rounds": 1, "gap_closed": gap_closed, "outcome": outcome,
    }


class TestLoadTrace(unittest.TestCase):
    def test_a6_dirty_lines_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            path.write_text(
                json.dumps(_entry()) + "\n"
                + "{invalid json}\n"
                + "\n"
                + json.dumps(_entry()),
                encoding="utf-8",
            )
            entries, skipped = load_trace(path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(skipped, 1)

    def test_a7_missing_file(self):
        entries, skipped = load_trace(Path("/nonexistent/trace.jsonl"))
        self.assertEqual((entries, skipped), ([], 0))


class TestPerAtomStats(unittest.TestCase):
    def test_a1_stats_computation(self):
        entries = [
            _entry(queries=["扣非归母净利润"], new_hits=2, gap_closed=True,
                   outcome="supplement_succeeded"),
            _entry(queries=["扣非归母净利润"], new_hits=0, gap_closed=False,
                   outcome="refused_maintained"),
            _entry(atom_id="MET-005", queries=["货币资金"], new_hits=1, gap_closed=True),
        ]
        rows = per_atom_stats(entries)
        self.assertEqual(len(rows), 2)
        met001 = next(r for r in rows if r["atom_id"] == "MET-001")
        self.assertEqual(met001["triggers"], 2)
        self.assertAlmostEqual(met001["gap_closed_rate"], 0.5, places=4)
        self.assertEqual(met001["outcome_dist"]["supplement_succeeded"], 1)
        self.assertEqual(met001["outcome_dist"]["refused_maintained"], 1)


class TestSuggestions(unittest.TestCase):
    def setUp(self):
        self.store = get_default_store()

    def test_a2_high_hit_rate_expands_alias(self):
        """R1：命中率 100% 且样本 3、不在 aliases → 扩 aliases 建议。"""
        entries = [
            _entry(queries=["扣除非经常性损益后的净利润"], new_hits=3) for _ in range(3)
        ]
        suggestions = alias_expansion_suggestions(entries, self.store)
        r1 = [s for s in suggestions if s["rule"] == "R1_expand_alias"
              and s["atom_id"] == "MET-001"]
        self.assertTrue(r1)
        self.assertEqual(r1[0]["term"], "扣除非经常性损益后的净利润")

    def test_a3_zero_hit_target_review(self):
        """R2：evidence_target 补检零命中 → 复核表述建议。"""
        entries = [
            _entry(queries=["合并利润表"], new_hits=0) for _ in range(3)
        ]
        suggestions = alias_expansion_suggestions(entries, self.store)
        r2 = [s for s in suggestions if s["rule"] == "R2_review_target"
              and s["atom_id"] == "MET-001"]
        self.assertTrue(r2)
        self.assertEqual(r2[0]["term"], "合并利润表")

    def test_a4_insufficient_samples_no_suggestion(self):
        """样本不足（<3）不出建议。"""
        entries = [_entry(queries=["扣除非经常性损益后的净利润"], new_hits=3)] * 2
        suggestions = alias_expansion_suggestions(entries, self.store)
        self.assertFalse([s for s in suggestions if s["atom_id"] == "MET-001"])

    def test_a5_already_alias_no_r1(self):
        """高命中词已是 aliases 成员 → 无需扩（不出 R1）。"""
        entries = [
            _entry(queries=["扣非净利润"], new_hits=3) for _ in range(3)
        ]  # 扣非净利润 是 MET-001 已有 alias
        suggestions = alias_expansion_suggestions(entries, self.store)
        self.assertFalse([s for s in suggestions if s["rule"] == "R1_expand_alias"
                          and s["term"] == "扣非净利润"])

    def test_a8_zero_trigger_atom_absent(self):
        """R3：零触发原子不出现在任何建议（无操作）。"""
        entries = [_entry(queries=["扣非归母净利润"], new_hits=1)] * 3
        suggestions = alias_expansion_suggestions(entries, self.store)
        self.assertFalse([s for s in suggestions if s["atom_id"] == "MET-005"])


class TestRenderReport(unittest.TestCase):
    def test_render_includes_stats_and_suggestions(self):
        rows = [{
            "atom_id": "MET-001", "triggers": 3, "trigger_dist": {"evidence_gap": 3},
            "strategy_dist": {"dense_recall": 3}, "rounds_total": 3,
            "gap_closed_rate": 1.0, "outcome_dist": {"supplement_succeeded": 3},
        }]
        suggestions = [{
            "rule": "R1_expand_alias", "atom_id": "MET-001",
            "term": "扣除非经常性损益后的净利润", "evidence": "补检命中率 100%（3/3）",
        }]
        report = render_report(rows, suggestions, skipped=1)
        self.assertIn("MET-001", report)
        self.assertIn("R1_expand_alias", report)
        self.assertIn("脏行跳过：1", report)
        self.assertIn("入库流程", report)   # Builder 不自批提示

    def test_render_empty(self):
        report = render_report([], [], skipped=0)
        self.assertIn("（无——样本不足或知识已覆盖）", report)


if __name__ == "__main__":
    unittest.main()
