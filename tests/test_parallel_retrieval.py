"""A2 并行检索 + 索引常驻缓存测试。

层 1（本组）：_IndexCache 命中/mtime 失效/LRU 驱逐/统计。
层 2/3（四路并行、多查询并行）：见 TestParallelBackends / TestMultiQueryParallel。
"""
import pickle
import tempfile
import time
import unittest
from pathlib import Path

from src.retrieval import _IndexCache, _load_index, reset_index_cache


class TestIndexCache(unittest.TestCase):
    def setUp(self):
        reset_index_cache()
        self.addCleanup(reset_index_cache)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _write_index(self, name: str, payload) -> Path:
        path = self.dir / name
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path

    def test_t4_cache_hit_skips_disk(self):
        """同路径两次加载：第二次命中缓存（不重新反序列化）。"""
        path = self._write_index("a.pkl", {"marker": 1})
        first = _load_index(path)
        second = _load_index(path)
        self.assertIs(first, second)  # 同一对象，未重新加载
        stats = _load_index.__globals__["_INDEX_CACHE"].stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)

    def test_t5_mtime_invalidation(self):
        """mtime 变化（索引重建）自动失效重载。"""
        path = self._write_index("a.pkl", {"version": 1})
        first = _load_index(path)
        # 重写文件（内容变化）并确保 mtime 前进
        time.sleep(0.01)
        with open(path, "wb") as f:
            pickle.dump({"version": 2}, f)
        second = _load_index(path)
        self.assertIsNot(first, second)
        self.assertEqual(second["version"], 2)

    def test_t6_lru_eviction_by_index_count(self):
        """容量驱逐按「索引文件」计数：超限逐出最久未用索引。"""
        cache = _IndexCache(max_slots=2)
        p1 = self._write_index("i1.pkl", {"n": 1})
        p2 = self._write_index("i2.pkl", {"n": 2})
        p3 = self._write_index("i3.pkl", {"n": 3})
        cache.get(p1)
        cache.get(p2)
        cache.get(p1)          # p1 变最近使用
        obj3 = cache.get(p3)   # 超限 → 逐出 p2（最久未用）
        self.assertEqual(obj3["n"], 3)
        stats = cache.stats()
        self.assertEqual(stats["slots"], 2)
        self.assertEqual(stats["evictions"], 1)
        # p2 已被逐出：再取 p2 应 miss 重载
        obj2 = cache.get(p2)
        self.assertEqual(obj2["n"], 2)
        self.assertEqual(cache.stats()["misses"], 4)

    def test_stats_and_clear(self):
        cache = _IndexCache(max_slots=4)
        self.assertEqual(cache.stats(), {"slots": 0, "max": 4, "hits": 0, "misses": 0, "evictions": 0})
        cache.get(self._write_index("x.pkl", {"n": 0}))
        cache.clear()
        self.assertEqual(cache.stats()["slots"], 0)


if __name__ == "__main__":
    unittest.main()


class _FakeBackendRetriever:
    """单路 fake：按页返回固定结果；fail_backend=True 时抛错。"""

    def __init__(self, pages, fail_backend=False):
        self.pages = pages
        self.fail_backend = fail_backend
        self.calls = 0

    def retrieve_by_company_name(self, **kwargs):
        self.calls += 1
        if self.fail_backend:
            raise RuntimeError("simulated backend failure")
        return [
            {"page": p, "distance": 0.5 + p * 0.01, "pdf_sha1": "doc",
             "retrieval_sources": [], "matched_child_chunk_ids": [], "matched_tags": []}
            for p in self.pages
        ]


def _bare_hybrid_retriever(parallel: bool):
    from src.retrieval import HybridRetriever
    retriever = object.__new__(HybridRetriever)
    retriever.fusion_method = "rrf"
    retriever.rrf_k = 60
    retriever._parallel_backends_enabled = parallel
    retriever.vector_retriever = _FakeBackendRetriever([1, 2])
    retriever.bm25_retriever = _FakeBackendRetriever([2, 3])
    retriever.sparse_retriever = _FakeBackendRetriever([1, 3])
    retriever.tag_retriever = _FakeBackendRetriever([2])
    return retriever


class TestParallelBackends(unittest.TestCase):
    def test_t1_parallel_equals_serial(self):
        """并行与串行输出全等（含顺序）——确定性聚合。"""
        serial = _bare_hybrid_retriever(parallel=False)
        parallel = _bare_hybrid_retriever(parallel=True)
        kwargs = dict(company_name="公司", query="营收", top_n=5,
                      parent_retrieval_mode="child", filters=None, candidate_doc_ids=None)
        out_serial = serial.retrieve_candidates_by_company_name(**kwargs)
        out_parallel = parallel.retrieve_candidates_by_company_name(**kwargs)
        self.assertEqual(out_serial, out_parallel)

    def test_t2_single_backend_failure_degrades(self):
        """单路异常：并行路径降级为空 + 其余三路正常（串行路径保持异常传播）。"""
        retriever = _bare_hybrid_retriever(parallel=True)
        retriever.bm25_retriever = _FakeBackendRetriever([], fail_backend=True)
        results = retriever.retrieve_candidates_by_company_name(
            company_name="公司", query="营收", top_n=5,
        )
        # vector/sparse/tag 三路仍贡献结果（page 1/2/3）
        pages = {item["page"] for item in results}
        self.assertTrue({1, 2, 3} <= pages)
        # 降级路径：bm25 无贡献但整体不失败
        self.assertEqual(retriever.bm25_retriever.calls, 1)

    def test_t2b_serial_path_propagates_failure(self):
        """串行回退路径保持旧语义：一路抛错整体失败。"""
        retriever = _bare_hybrid_retriever(parallel=False)
        retriever.bm25_retriever = _FakeBackendRetriever([], fail_backend=True)
        with self.assertRaises(RuntimeError):
            retriever.retrieve_candidates_by_company_name(company_name="公司", query="营收")

    def test_t3_vector_only_scope(self):
        """backend_scope=vector_only：仅 vector 路被调用。"""
        retriever = _bare_hybrid_retriever(parallel=True)
        retriever.retrieve_candidates_by_company_name(
            company_name="公司", query="营收", top_n=5, backend_scope="vector_only",
        )
        self.assertEqual(retriever.vector_retriever.calls, 1)
        self.assertEqual(retriever.bm25_retriever.calls, 0)
        self.assertEqual(retriever.sparse_retriever.calls, 0)
        self.assertEqual(retriever.tag_retriever.calls, 0)


class TestMultiQueryParallel(unittest.TestCase):
    def _make_processor(self, retriever):
        from src.questions_processing import QuestionsProcessor
        processor = object.__new__(QuestionsProcessor)
        processor.llm_reranking_sample_size = 28
        processor.retrieval_debug_top_n = 5
        processor.parent_retrieval_mode = "child"
        return processor

    def test_t7_multi_query_parallel_equals_serial(self):
        """多查询并行与逐条串行输出全等（顺序与配对不变）。"""
        queries = ["归母净利润", "归属于母公司股东的净利润", "扣非后归母净利润"]

        serial_retriever = _bare_hybrid_retriever(parallel=False)
        serial_processor = self._make_processor(serial_retriever)
        serial_runs = serial_processor._retrieve_multi_query_candidate_runs(
            retriever=serial_retriever, company_name="公司",
            search_queries=queries, filters=None,
        )

        parallel_retriever = _bare_hybrid_retriever(parallel=True)
        parallel_processor = self._make_processor(parallel_retriever)
        parallel_runs = parallel_processor._retrieve_multi_query_candidate_runs(
            retriever=parallel_retriever, company_name="公司",
            search_queries=queries, filters=None,
        )
        self.assertEqual(serial_runs, parallel_runs)
        self.assertEqual([q for q, _ in parallel_runs], queries)

    def test_t7b_multi_query_empty_and_blank_handling(self):
        """空/空白查询过滤：不产生空 run。"""
        retriever = _bare_hybrid_retriever(parallel=True)
        processor = self._make_processor(retriever)
        runs = processor._retrieve_multi_query_candidate_runs(
            retriever=retriever, company_name="公司",
            search_queries=["", "   ", "净利润"], filters=None,
        )
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0][0], "净利润")


if __name__ == "__main__":
    unittest.main()
