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
