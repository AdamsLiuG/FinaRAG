"""A1 融合量纲修复测试：池内归一化融合（combined_score 名义权重=实际权重）。

背景：rerank 层 combined = llm_weight * relevance + vector_weight * distance，
distance 量纲随检索层 fusion_method 漂移（rrf ≈ Σ1/(60+rank) <0.1；average 为
归一后均值）——修复前 rrf 模式下向量路实际权重 ~2%（名义 30%）。
"""
import unittest

from src.reranking import (
    LLMReranker,
    _fuse_scores,
    _minmax_normalize,
)
from src.retrieval import HybridRetriever, _normalize_scores


def _make_doc(page: int, distance: float, text: str = "doc") -> dict:
    return {
        "page": page,
        "distance": distance,
        "text": f"{text} page {page}",
        "pdf_sha1": "sha",
    }


class TestFuseScores(unittest.TestCase):
    def test_t1_rrf_scale_distortion_fixed(self):
        """rrf 量纲：修复前 0.3×(0.03-0.01) 的向量差异被抹平，rel 高者胜不保。"""
        fused = _fuse_scores([0.9, 0.1], [0.03, 0.02], 0.7)
        self.assertGreater(fused[0], fused[1])

    def test_t2_bm25_scale_distortion_fixed(self):
        """BM25 量纲：修复前 0.3×25 绝对量级淹没 rel 信号（w=0.7 下 BM25 分高者胜）；
        修复后名义权重生效——LLM 主导（w=0.7）时 rel=0.9 者胜。"""
        fused = _fuse_scores([0.5, 0.9], [25.0, 24.9], 0.7)
        self.assertGreater(fused[1], fused[0])

    def test_t3_constant_pool_full_signal(self):
        """常数池：两路均 >0 的常数 → 全 1.0（该路有信号语义，与检索层对齐）。"""
        fused = _fuse_scores([0.5, 0.5], [0.1, 0.1], 0.7)
        self.assertEqual(fused, [1.0, 1.0])

    def test_t4_single_element_pool(self):
        fused = _fuse_scores([0.7], [999.0], 0.7)
        self.assertEqual(fused, [1.0])

    def test_t5_weight_endpoints(self):
        pure_rel = _fuse_scores([0.2, 0.8], [0.9, 0.1], 1.0)
        pure_dist = _fuse_scores([0.2, 0.8], [0.9, 0.1], 0.0)
        self.assertEqual(pure_rel, [0.0, 1.0])
        self.assertEqual(pure_dist, [1.0, 0.0])

    def test_empty_inputs(self):
        self.assertEqual(_fuse_scores([], [], 0.7), [])
        self.assertEqual(_minmax_normalize([]), [])


class TestLLMRerankerBatchFusion(unittest.TestCase):
    def test_t6_batch_fusion_overrides_per_doc(self):
        """批量融合覆盖逐文档临时值：combined 为池内归一后融合（非逐文档直加）。"""
        reranker = LLMReranker.__new__(LLMReranker)

        def fake_rank_multiple(query, texts):
            scores = [0.9, 0.1, 0.5]
            return {"block_rankings": [{"relevance_score": s} for s in scores]}

        reranker.get_rank_for_multiple_blocks = fake_rank_multiple
        reranker.get_rank_for_single_block = None  # 不应被调用
        docs = [
            _make_doc(1, 0.03),
            _make_doc(2, 0.02),
            _make_doc(3, 0.025),
        ]
        results = reranker.rerank_documents("query", docs, documents_batch_size=4, llm_weight=0.7)
        self.assertEqual(len(results), 3)
        # rel 归一 [1.0, 0.0, 0.5]；dist 归一 [1.0, 0.0, 0.5]
        # combined = 0.7*rel_n + 0.3*dist_n
        by_page = {r["page"]: r for r in results}
        self.assertAlmostEqual(by_page[1]["combined_score"], 0.7 * 1.0 + 0.3 * 1.0, places=4)
        self.assertAlmostEqual(by_page[2]["combined_score"], 0.0, places=4)
        self.assertAlmostEqual(by_page[3]["combined_score"], 0.7 * 0.5 + 0.3 * 0.5, places=4)
        # 排序：page1 > page3 > page2
        self.assertEqual([r["page"] for r in results], [1, 3, 2])


class TestRetrievalFusionRegression(unittest.TestCase):
    """检索层现状回归：每路归一化（_normalize_scores）已存在，本修复不动检索层。"""

    @staticmethod
    def _bare_retriever(fusion_method: str) -> HybridRetriever:
        retriever = object.__new__(HybridRetriever)
        retriever.fusion_method = fusion_method
        retriever.rrf_k = 60
        return retriever

    def test_t7_average_mode_normalizes_per_backend(self):
        """average 模式：BM25 无界分数先经每路 min-max，两路同权参与平均。"""
        retriever = self._bare_retriever("average")
        vector_result = [{
            "page": 1, "distance": 0.9, "pdf_sha1": "a",
            "retrieval_sources": ["vector"], "matched_child_chunk_ids": [], "matched_tags": [],
        }]
        bm25_result = [{
            "page": 1, "distance": 30.0, "pdf_sha1": "a",
            "retrieval_sources": ["bm25"], "matched_child_chunk_ids": [], "matched_tags": [],
        }]
        merged = retriever._merge_retrieval_results(
            {"vector": vector_result, "bm25": bm25_result}, top_n=5
        )
        self.assertEqual(len(merged), 1)
        item = merged[0]
        self.assertAlmostEqual(item["vector_score"], 1.0, places=4)
        self.assertAlmostEqual(item["bm25_score"], 1.0, places=4)
        # average = (1 + 1) / 2
        self.assertAlmostEqual(item["average_score"], 1.0, places=4)

    def test_t8_rrf_mode_unchanged(self):
        """rrf 模式：rank-based 累加逻辑保持原样（本修复不触碰）。"""
        retriever = self._bare_retriever("rrf")
        vector_result = [{
            "page": 1, "distance": 0.9, "pdf_sha1": "a",
            "retrieval_sources": ["vector"], "matched_child_chunk_ids": [], "matched_tags": [],
        }]
        bm25_result = [{
            "page": 1, "distance": 30.0, "pdf_sha1": "a",
            "retrieval_sources": ["bm25"], "matched_child_chunk_ids": [], "matched_tags": [],
        }]
        merged = retriever._merge_retrieval_results(
            {"vector": vector_result, "bm25": bm25_result}, top_n=5
        )
        item = merged[0]
        # 两路 rank 均为 1：rrf = 2 × 1/(60+1)
        self.assertAlmostEqual(item["distance"], 2 / 61, places=6)
        # rrf 模式下 distance 即 rrf_score，不受本次 rerank 层修复影响
        self.assertAlmostEqual(item["rrf_score"], 2 / 61, places=6)

    def test_normalize_constant_pool_semantics_aligned(self):
        """两文件同名归一函数语义对齐：常数池 >0 → 1.0（该路有信号）。
        融合改造刻意复用既有 _minmax_normalize 而非新造语义。"""
        self.assertEqual(_normalize_scores([5.0, 5.0]), [1.0, 1.0])
        self.assertEqual(_minmax_normalize([5.0, 5.0]), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()


class TestTaskTypeWeights(unittest.TestCase):
    """A4 题型加权：task_type → llm_weight（config/retrieval_fusion_weights.yaml）。"""

    def setUp(self):
        from src import reranking
        self._reranking = reranking
        reranking.reset_fusion_weights_cache()
        self.addCleanup(reranking.reset_fusion_weights_cache)

    def test_t9_registered_task_types(self):
        resolve = self._reranking.resolve_llm_weight
        self.assertAlmostEqual(resolve("definition_qa", 0.7), 0.85)
        self.assertAlmostEqual(resolve("single_metric_lookup", 0.7), 0.50)
        self.assertAlmostEqual(resolve("evidence_qa", 0.7), 0.70)

    def test_t10_unregistered_falls_back_to_default(self):
        resolve = self._reranking.resolve_llm_weight
        self.assertAlmostEqual(resolve("novel_type", 0.7), 0.7)
        self.assertAlmostEqual(resolve(None, 0.7), 0.7)
        self.assertAlmostEqual(resolve("", 0.7), 0.7)

    def test_t11_out_of_range_weight_raises(self):
        """权重越界（∉(0,1) 开区间）→ ValueError：配置错误显式暴露。"""
        import tempfile
        import os
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("task_type_weights:\n  definition_qa: {llm_weight: 1.5}\n")
            tmp_path = tmp.name
        try:
            with patch.object(self._reranking, "_FUSION_WEIGHTS_PATH", tmp_path):
                with self.assertRaises(ValueError):
                    self._reranking.resolve_llm_weight("definition_qa", 0.7)
        finally:
            os.unlink(tmp_path)

    def test_t12_missing_config_falls_back(self):
        """yaml 缺失 → 空 dict 回退 default（部署容错，不阻断启动）。"""
        from unittest.mock import patch

        with patch.object(self._reranking, "_FUSION_WEIGHTS_PATH", "/nonexistent/path.yaml"):
            resolve = self._reranking.resolve_llm_weight
            self.assertAlmostEqual(resolve("definition_qa", 0.7), 0.7)

    def test_t13_rerank_candidates_resolves_weight(self):
        """集成：rerank_candidates 按 task_type 解析权重并传给 reranker。"""
        from src.retrieval import HybridRetriever

        captured = {}

        class _FakeReranker:
            def rerank_documents(self, query, documents, documents_batch_size, llm_weight):
                captured["llm_weight"] = llm_weight
                return list(documents)

        retriever = object.__new__(HybridRetriever)
        retriever.reranker = _FakeReranker()
        retriever.reranking_strategy = "single"
        retriever.colbert_top_n = 10
        retriever.final_reranking_backend = "llm_prompt"
        candidates = [_make_doc(1, 0.5)]
        retriever.rerank_candidates(
            query="q", candidate_results=candidates, llm_weight=0.7,
            task_type="definition_qa",
        )
        self.assertAlmostEqual(captured["llm_weight"], 0.85)
        retriever.rerank_candidates(
            query="q", candidate_results=candidates, llm_weight=0.7,
            task_type="single_metric_lookup",
        )
        self.assertAlmostEqual(captured["llm_weight"], 0.50)
        retriever.rerank_candidates(
            query="q", candidate_results=candidates, llm_weight=0.7,
        )
        self.assertAlmostEqual(captured["llm_weight"], 0.7)
