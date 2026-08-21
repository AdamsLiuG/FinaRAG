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
    detect_post_validate_gap,
    merge_supplement_results,
)
from src.metric_atoms import get_default_store


def _plan(metric_atom=None, playbook=None, task_type="single_metric_lookup"):
    return SimpleNamespace(
        metric_atom=metric_atom,
        playbook=playbook,
        task_type=task_type,
        filters=None,
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


class TestPostValidateGap(unittest.TestCase):
    """① 校验后触发（try-then-closed 的感知层）：flags → RetrievalGap。"""

    def test_p1_caliber_refused_triggers(self):
        """口径门禁拒答 → 定向补本口径（含扣非 targets、不含归母易混名）。"""
        plan = _plan(metric_atom=_met001_payload())
        results = _results(["归母净利润为 100 亿元"])
        gap = detect_post_validate_gap(["metric_caliber_conflict_refused"], plan, results)
        self.assertIsNotNone(gap)
        self.assertEqual(gap.trigger, "evidence_gap")
        self.assertNotIn("归母净利润", gap.supplement_queries)
        self.assertIn("扣除非经常性损益后的净利润", gap.supplement_queries)

    def test_p2_evidence_mismatch_triggers(self):
        plan = _plan(metric_atom=_met001_payload())
        gap = detect_post_validate_gap(["metric_evidence_mismatch"], plan, _results(["无关文本"]))
        self.assertIsNotNone(gap)
        self.assertEqual(gap.trigger, "evidence_gap")

    def test_p3_playbook_incomplete_triggers(self):
        plan = _plan(playbook={"playbook_id": "PLB-002", "required_metrics": ["MET-005"]})
        gap = detect_post_validate_gap(
            ["implicit_intent_evidence_incomplete"], plan, _results(["现金流分析"])
        )
        self.assertIsNotNone(gap)
        self.assertEqual(gap.trigger, "playbook_incomplete")
        self.assertIn("货币资金", gap.supplement_queries)

    def test_p4_unrelated_flags_no_trigger(self):
        plan = _plan(metric_atom=_met001_payload())
        self.assertIsNone(
            detect_post_validate_gap(
                ["numeric_grounding_weak", "missing_relevant_pages"], plan, _results(["x"])
            )
        )
        self.assertIsNone(detect_post_validate_gap([], plan, _results(["x"])))
        self.assertIsNone(detect_post_validate_gap(None, plan, _results(["x"])))

    def test_p5_try_semantics_gap_closes_after_supplement(self):
        """try 语义（判定链与主流程一致）：拒答 flags 触发 → 补检含本口径证据
        → 二次校验的 conflict 解除（新 flags 为空）→ gap_closed 判定基础成立。

        注意：闭环终止的判定输入是**二次校验的新 flags**（证据含本口径后
        answer_validation 不再产生 conflict flag），而非旧 flags。"""
        plan = _plan(metric_atom=_met001_payload())
        base = _results(["归母净利润为 100 亿元"])
        gap = detect_post_validate_gap(["metric_caliber_conflict_refused"], plan, base)
        self.assertIsNotNone(gap)
        supplement = [{
            "page": 7, "pdf_sha1": "doc_a",
            "text": "扣非归母净利润为 5 亿元，扣除非经常性损益后的净利润 4.8 亿",
            "combined_score": 0.6,
        }]
        merged = merge_supplement_results(base, supplement)
        # 模拟二次校验：证据已含本口径 → conflict 解除 → 新 flags 为空
        new_flags = [] if _targets_all_covered(merged, plan) else ["metric_caliber_conflict_refused"]
        self.assertEqual(new_flags, [])  # 前提确认：补检证据确实覆盖本口径
        # gap_closed 判定（主流程同构）：新 flags 无可补救源 → 闭环终止
        self.assertIsNone(detect_post_validate_gap(new_flags, plan, merged))

    def test_p6_then_closed_semantics_gap_persists(self):
        """then-closed 语义：补检仍只有易混口径证据 → 再判仍返回 gap
        （refused_maintained 的判定基础）。"""
        plan = _plan(metric_atom=_met001_payload())
        base = _results(["归母净利润为 100 亿元"])
        supplement = _results(["归属于母公司股东的净利润继续增长"])  # 还是易混口径
        merged = merge_supplement_results(base, supplement)
        gap = detect_post_validate_gap(["metric_caliber_conflict_refused"], plan, merged)
        self.assertIsNotNone(gap)  # 缺口未闭合 → 维持拒答

    def test_p7_no_atom_no_gap(self):
        """flag 有触发源但原子未命中 → 不硬造 gap（不可补救的 flags 跳过）。"""
        plan = _plan()  # 无 metric_atom
        self.assertIsNone(
            detect_post_validate_gap(["metric_caliber_conflict_refused"], plan, _results(["x"]))
        )

    def test_p8_first_remediable_flag_wins(self):
        """多 flags 时首个可补救源生效（映射表序）。"""
        plan = _plan(
            metric_atom=_met001_payload(),
            playbook={"playbook_id": "PLB-002", "required_metrics": ["MET-005"]},
        )
        gap = detect_post_validate_gap(
            ["numeric_grounding_weak", "metric_evidence_mismatch", "implicit_intent_evidence_incomplete"],
            plan, _results(["无关文本"]),
        )
        self.assertEqual(gap.trigger, "evidence_gap")


def _targets_all_covered(results, plan):
    """P5 辅助：MET-001 的任一目标词是否已出现在结果文本中。"""
    from src.adaptive_retrieval import _result_texts

    corpus = _result_texts(results)
    return any(t in corpus for t in ["扣非归母净利润", "扣除非经常性损益后的净利润", "合并利润表"])


class _LoopFakeRetriever:
    """② 循环测试用 fake：retrieve 按 round 返回可控结果；rerank 原样返回。"""

    def __init__(self, supplement_pages_by_round):
        self.supplement_pages_by_round = supplement_pages_by_round
        self.retrieve_calls = 0

    def retrieve_candidates_by_company_name(self, company_name, query, top_n=28,
                                            parent_retrieval_mode="child", filters=None,
                                            candidate_doc_ids=None, **kwargs):
        self.retrieve_calls += 1
        idx = min(self.retrieve_calls - 1, len(self.supplement_pages_by_round) - 1)
        return list(self.supplement_pages_by_round[idx])

    def rerank_candidates(self, query, candidate_results, documents_batch_size=2,
                          top_n=6, llm_weight=0.7, task_type=None):
        return list(candidate_results)[:top_n]


def _bare_processor():
    from src.questions_processing import QuestionsProcessor

    processor = object.__new__(QuestionsProcessor)
    processor.final_reranking_batch_size = 4
    processor.retrieval_debug_top_n = 10
    processor.llm_reranking_sample_size = 28
    processor.parent_retrieval_mode = "child"
    return processor


def _pages(items):
    """items: [(page, text)] → retrieval 结果形态。"""
    return [
        {"page": p, "text": t, "pdf_sha1": "doc_a",
         "retrieval_sources": [], "matched_child_chunk_ids": [], "matched_tags": [],
         "combined_score": 0.5}
        for p, t in items
    ]


class TestConvergentLoop(unittest.TestCase):
    """② 收敛驱动的多轮：三收敛条件 + 查询幂等 + 预算。"""

    def setUp(self):
        self.plan = _plan(metric_atom=_met001_payload())
        self.base = _pages([(1, "归母净利润为 100 亿元")])
        self.processor = _bare_processor()

    def _run(self, fake_retriever, state=None):
        return self.processor._run_adaptive_supplement(
            retriever=fake_retriever,
            company_name="公司",
            question="扣非归母净利润是多少",
            rewrite_result=self.plan,
            retrieval_results=self.base,
            state=state,
        )

    def test_c1_gap_closes_stops(self):
        """收敛条件 1：补检覆盖 targets → 第二轮 detect_gap None → 停。"""
        fake = _LoopFakeRetriever([
            _pages([(7, "扣非归母净利润为 5 亿元，扣除非经常性损益后的净利润 4.8 亿")]),
        ])
        state = {"rounds_used": 0, "traces": []}
        result = self._run(fake, state)
        self.assertIsNotNone(result)
        self.assertEqual(state["rounds_used"], 1)      # 一轮即闭合
        texts = "\n".join(r.get("text") or "" for r in result)
        self.assertIn("扣非归母净利润", texts)

    def test_c2_no_new_evidence_stops(self):
        """收敛条件 2：补检全重复（同 sha/page）→ 池长不变 → 停且不计轮。"""
        fake = _LoopFakeRetriever([
            _pages([(1, "归母净利润（重复内容）")]),   # 同 page=1 → 去重后无新增
        ])
        state = {"rounds_used": 0, "traces": []}
        result = self._run(fake, state)
        self.assertIsNone(result)                        # 无有效轮
        self.assertEqual(state["rounds_used"], 0)

    def test_c3_query_idempotent_stops(self):
        """查询幂等停：gap.queries 两轮相同（patch detect_gap 恒返同 gap）。"""
        from unittest.mock import patch

        from src.adaptive_retrieval import RetrievalGap

        fixed_gap = RetrievalGap(
            trigger="evidence_gap",
            supplement_queries=["扣非归母净利润"],
            rationale="fixed",
        )
        fake = _LoopFakeRetriever([
            _pages([(7, "扣非归母净利润内容"), (8, "更多新证据"), (9, "再新证据")]),
        ])
        state = {"rounds_used": 0, "traces": []}
        with patch("src.adaptive_retrieval.detect_gap", return_value=fixed_gap):
            # questions_processing 内 import 的是模块属性——patch 模块级即可
            result = self._run(fake, state)
        # 第二轮 queries 与第一轮相同 → 幂等停：恰好 1 轮
        self.assertEqual(state["rounds_used"], 1)

    def test_c4_budget_cap(self):
        """收敛条件 3：预算耗尽——跨触发源接力（queries 逐轮变化，不触发
        幂等停）+ 每轮新证据，恰 max_rounds 轮停。

        注：同一触发源的持续 gap 其 queries 固定，会被幂等停先拦——预算
        上限真正约束的是「不同触发源接力」的场景（本测试以变化 gap 模拟）。"""
        from unittest.mock import patch

        from src.adaptive_retrieval import RetrievalGap

        gap_sequence = [
            RetrievalGap(trigger="evidence_gap", supplement_queries=["扣非归母净利润"], rationale="r1"),
            RetrievalGap(trigger="evidence_gap", supplement_queries=["扣除非经常性损益后的净利润"], rationale="r2"),
            RetrievalGap(trigger="evidence_gap", supplement_queries=["合并利润表"], rationale="r3"),
        ]
        pages_sequence = [
            _pages([(10 + i, f"第 {i} 轮新证据")]) for i in range(3)
        ]
        fake = _LoopFakeRetriever(pages_sequence)
        state = {"rounds_used": 0, "traces": []}
        with patch("src.adaptive_retrieval.adaptive_max_rounds", return_value=2), \
                patch("src.adaptive_retrieval.detect_gap", side_effect=gap_sequence):
            self._run(fake, state)
        self.assertEqual(state["rounds_used"], 2)   # 第 3 轮被预算拦
