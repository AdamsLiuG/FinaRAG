"""Tests for P1-1 SqliteDocumentRepository, QdrantHybridStore (embedded & HTTP), and UnifiedHybridRetriever."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np

from src.document_store import clear_document_store_cache, get_document_store
from src.retrieval_filters import RetrievalFilters
from src.storage.base import HybridSearchFilter, StorageSecurityError
from src.storage.migrate_to_qdrant import migrate_corpus
from src.storage.qdrant_store import QdrantHybridStore, UnifiedHybridRetriever
from src.storage.sqlite_doc_store import SqliteDocumentRepository


def _build_fixture_docs():
    spdb = {
        "metainfo": {
            "doc_id": "600000_2024",
            "sha1_name": "600000_2024",
            "company_name": "浦发银行",
            "stock_code": "600000",
            "industry_l1": "货币金融服务",
            "report_year": 2024,
            "currency": "CNY",
        },
        "content": {
            "pages": [
                {"page": 1, "text": "浦发银行 2024 年年度报告封面与目录"},
                {"page": 19, "text": "主要会计数据：2024年浦发银行营业收入170,748百万元，同比下降1.55%。"},
                {"page": 45, "text": "风险管理与资本充足率说明"},
            ],
            "parent_chunks": [
                {"chunk_id": 10, "page": 19, "text": "父块：2024年浦发银行营业收入170,748百万元，归母净利润45,257百万元。"}
            ],
            "chunks": [
                {
                    "id": 0,
                    "chunk_id": 0,
                    "page": 1,
                    "chunk_type": "content",
                    "node_type": "child",
                    "parent_chunk_id": 10,
                    "section_name": "公司简介",
                    "topic_flags": ["has_pdfcrawl_metadata"],
                    "strategy_tags": ["数字化转型"],
                    "text": "浦发银行 2024 年年度报告封面",
                },
                {
                    "id": 1,
                    "chunk_id": 1,
                    "page": 19,
                    "chunk_type": "table",
                    "node_type": "child",
                    "parent_chunk_id": 10,
                    "section_name": "主要会计数据",
                    "table_id": "11",
                    "topic_flags": ["financial_performance"],
                    "strategy_tags": ["数字化转型"],
                    "text": "2024年浦发银行营业收入170748百万元，同比下降1.55%。",
                },
            ],
            "structured_tables": [
                {"table_id": "11", "page": 19, "markdown": "| 营业收入 | 170,748 |"}
            ],
        },
    }
    jn = {
        "metainfo": {
            "doc_id": "600791_2024",
            "sha1_name": "600791_2024",
            "company_name": "京能置业",
            "stock_code": "600791",
            "industry_l1": "房地产业",
            "report_year": 2024,
            "currency": "CNY",
        },
        "content": {
            "pages": [
                {"page": 5, "text": "京能置业 2024年营业收入54.59亿元，同比增长20.36%。"}
            ],
            "parent_chunks": [
                {"chunk_id": 0, "page": 5, "text": "父块：京能置业 2024年营业收入54.59亿元，净利润承压。"}
            ],
            "chunks": [
                {
                    "id": 0,
                    "chunk_id": 0,
                    "page": 5,
                    "chunk_type": "content",
                    "node_type": "child",
                    "parent_chunk_id": 0,
                    "section_name": "主要财务指标",
                    "topic_flags": ["financial_performance"],
                    "strategy_tags": [],
                    "text": "京能置业 2024年营业收入54.59亿元。",
                }
            ],
            "structured_tables": [],
        },
    }
    return [spdb, jn]


def _unit_vec(dim: int, hot_idx: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[hot_idx % dim] = 1.0
    return v


class StorageEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.chunked_dir = self.root / "chunked_reports"
        self.vector_dir = self.root / "vector_dbs"
        self.sparse_dir = self.root / "sparse_dbs"
        self.chunked_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

        import pickle
        import faiss

        docs = _build_fixture_docs()
        for doc_i, doc in enumerate(docs):
            did = doc["metainfo"]["doc_id"]
            (self.chunked_dir / f"{did}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            n_chunks = len(doc["content"]["chunks"])
            idx = faiss.IndexFlatIP(16)
            vecs = np.vstack([_unit_vec(16, doc_i * 4 + j) for j in range(n_chunks)])
            idx.add(vecs)
            faiss.write_index(idx, str(self.vector_dir / f"{did}.faiss"))
            lw = [{str(100 + doc_i * 10 + j): 0.85, "营业收入": 0.9 if "营业收入" in ch["text"] else 0.1} for j, ch in enumerate(doc["content"]["chunks"])]
            with open(self.sparse_dir / f"{did}.pkl", "wb") as fh:
                pickle.dump({"lexical_weights": lw}, fh)

    def tearDown(self):
        clear_document_store_cache()
        self.tmp_dir.cleanup()

    def test_sqlite_doc_repo_on_demand_lru_and_security(self):
        db_path = self.root / "finarag_docs.sqlite"
        summary = SqliteDocumentRepository.build_from_chunked_dir(
            self.chunked_dir,
            db_path,
            allowed_base_dir=self.root,
        )
        self.assertEqual(summary["reports_count"], 2)
        self.assertEqual(summary["chunks_count"], 3)
        self.assertEqual(len(summary["corpus_version"]), 16)

        repo = SqliteDocumentRepository(db_path, allowed_base_dir=self.root, page_cache_size=2)
        self.addCleanup(repo.close)

        # Page reads + LRU hit/miss/eviction verification
        p1 = repo.get_page("600000_2024", 1)
        self.assertIn("封面", p1["text"])
        _ = repo.get_page("600000_2024", 1)  # cache hit
        _ = repo.get_page("600000_2024", 19)
        _ = repo.get_page("600000_2024", 45)  # evicts page 1
        stats = repo.stats()
        self.assertEqual(stats["page_cache"]["hits"], 1)
        self.assertEqual(stats["page_cache"]["evictions"], 1)

        ch = repo.get_chunk_by_index("600000_2024", 1)
        self.assertEqual(ch["table_id"], "11")
        parent = repo.get_parent_chunk("600000_2024", 10)
        self.assertIn("45,257", parent["text"])
        tables = repo.get_tables("600000_2024", page=19)
        self.assertEqual(len(tables), 1)

        # Dual-track get_document_store(DOCUMENT_STORE_BACKEND=sqlite) compatibility
        clear_document_store_cache()
        with patch.dict(os.environ, {"DOCUMENT_STORE_BACKEND": "sqlite"}):
            store = get_document_store(self.chunked_dir)
            self.assertEqual(len(store.documents), 2)
            self.assertIn("600000_2024", store.metainfo_by_doc_id)
            spdb_entry = [e for e in store.documents if e["name"] == "600000_2024"][0]
            self.assertEqual(spdb_entry["document"]["metainfo"]["company_name"], "浦发银行")
            self.assertEqual(len(spdb_entry["document"]["content"]["pages"]), 3)

        # Security: traversal and sibling prefix trap rejected
        with self.assertRaises(StorageSecurityError):
            SqliteDocumentRepository(self.root / "../escape.sqlite", allowed_base_dir=self.root)
        sibling = self.root.parent / f"{self.root.name}-malicious" / "bad.sqlite"
        with self.assertRaises(StorageSecurityError):
            SqliteDocumentRepository(sibling, allowed_base_dir=self.root)

    def test_embedded_hybrid_store_sq8_filter_fts_and_colbert_maxsim(self):
        store_dir = self.root / "unified_sq8"
        store = QdrantHybridStore(
            mode="embedded",
            store_dir=store_dir,
            dense_dim=16,
            colbert_dim=16,
            allowed_base_dir=self.root,
        )
        self.addCleanup(store.close)

        # Build >= 260 points so SQ8 INT8 quantization path is exercised too
        points = []
        for i in range(270):
            comp = "浦发银行" if i % 2 == 0 else "京能置业"
            did = "600000_2024" if i % 2 == 0 else "600791_2024"
            text = (
                "浦发银行 2024年营业收入 170748 百万元" if i == 2
                else f"{comp} 附注事项 编号{i}"
            )
            points.append(
                {
                    "point_id": i,
                    "doc_id": did,
                    "chunk_idx": i,
                    "company_name": comp,
                    "report_year": 2024,
                    "industry_l1": "货币金融服务" if comp == "浦发银行" else "房地产业",
                    "page": (i % 40) + 1,
                    "node_type": "child",
                    "section_name": "主要会计数据",
                    "topic_flags": ["financial_performance"] if i == 2 else [],
                    "strategy_tags": ["数字化转型"],
                    "text": text,
                    "dense_vector": _unit_vec(16, 1 if i == 2 else (i % 14) + 2),
                    "bge_sparse": {"rev_key": 0.95} if i == 2 else {"noise": 0.1},
                    "colbert_vectors": np.vstack([_unit_vec(16, 1), _unit_vec(16, 3)]) if i == 2 else np.vstack([_unit_vec(16, 12)]),
                }
            )
        inserted = store.upsert_points(points)
        self.assertEqual(inserted, 270)
        stats = store.stats()
        self.assertEqual(stats["dense_quantization"], "sq8_int8")
        self.assertEqual(stats["vectors_count"], 270)

        hits = store.search_hybrid(
            dense_vector=_unit_vec(16, 1),
            query_text='营业收入 * OR ("浦发"):',
            bge_sparse={"rev_key": 0.95},
            colbert_vectors=np.vstack([_unit_vec(16, 1), _unit_vec(16, 3)]),
            filters=HybridSearchFilter(company_name="浦发银行", topic_flags=["financial_performance"]),
            top_k=5,
        )
        self.assertTrue(hits)
        self.assertEqual(hits[0].point_id, 2)
        self.assertEqual(hits[0].doc_id, "600000_2024")
        self.assertIn("colbert", hits[0].retrieval_sources)

        with self.assertRaises(StorageSecurityError):
            QdrantHybridStore(mode="embedded", store_dir=self.root / "../bad", allowed_base_dir=self.root)
        with self.assertRaises(StorageSecurityError):
            QdrantHybridStore(mode="http", qdrant_url="ftp://invalid.local")

    def test_http_qdrant_mode_schema_upsert_and_query_via_mock_transport(self):
        seen_requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8")) if request.content else None
            seen_requests.append((request.method, request.url.path, body))
            if request.url.path.endswith("/points/query"):
                return httpx.Response(
                    200,
                    json={
                        "result": {
                            "points": [
                                {
                                    "id": 7,
                                    "score": 0.912,
                                    "payload": {
                                        "doc_id": "600000_2024",
                                        "chunk_idx": 1,
                                        "page": 19,
                                        "text": "2024年营业收入170748百万元",
                                        "company_name": "浦发银行",
                                    },
                                }
                            ]
                        }
                    },
                )
            return httpx.Response(200, json={"result": {"status": "ok"}})

        store = QdrantHybridStore(
            mode="http",
            qdrant_url="http://127.0.0.1:6333",
            dense_dim=8,
            colbert_dim=8,
            http_transport=httpx.MockTransport(handler),
        )
        self.addCleanup(store.close)
        store.ensure_collection()
        # 1 PUT collection + 10 PUT payload indexes
        self.assertEqual(len(seen_requests), 11)
        col_req = seen_requests[0][2]
        self.assertEqual(col_req["vectors"]["dense"]["quantization_config"]["scalar"]["type"], "int8")
        self.assertEqual(col_req["vectors"]["colbert"]["multivector_config"]["comparator"], "max_sim")
        self.assertEqual(col_req["sparse_vectors"]["bm25_sparse"]["modifier"], "idf")

        store.upsert_points(
            [
                {
                    "point_id": 7,
                    "doc_id": "600000_2024",
                    "chunk_idx": 1,
                    "company_name": "浦发银行",
                    "report_year": 2024,
                    "page": 19,
                    "text": "2024年营业收入170748百万元",
                    "dense_vector": _unit_vec(8, 2),
                    "bge_sparse": {"100": 0.9},
                    "bm25_sparse": {"营业": 1.0},
                    "colbert_vectors": np.vstack([_unit_vec(8, 2)]),
                }
            ]
        )
        hits = store.search_hybrid(
            dense_vector=_unit_vec(8, 2),
            query_text="营业收入",
            bge_sparse={"100": 0.9},
            colbert_vectors=np.vstack([_unit_vec(8, 2)]),
            filters=HybridSearchFilter(company_name="浦发银行", report_year=2024),
            top_k=3,
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].doc_id, "600000_2024")
        query_body = seen_requests[-1][2]
        self.assertEqual(query_body["using"], "colbert")
        self.assertEqual(query_body["prefetch"]["fusion"], "rrf")
        self.assertEqual(len(query_body["prefetch"]["prefetch"]), 3)

    def test_migrate_corpus_and_unified_retriever_child_page_block(self):
        doc_db = self.root / "finarag_docs.sqlite"
        unified_dir = self.root / "unified_store"
        summary = migrate_corpus(
            chunked_reports_dir=self.chunked_dir,
            vector_db_dir=self.vector_dir,
            sparse_db_dir=self.sparse_dir,
            doc_sqlite_path=doc_db,
            unified_store_dir=unified_dir,
        )
        self.assertTrue(summary["parity_verified"])
        self.assertEqual(summary["document_store"]["reports_count"], 2)
        self.assertEqual(summary["hybrid_store"]["vectors_count"], 3)

        repo = SqliteDocumentRepository(doc_db)
        store = QdrantHybridStore(mode="embedded", store_dir=unified_dir, dense_dim=16)
        self.addCleanup(repo.close)
        self.addCleanup(store.close)

        retriever = UnifiedHybridRetriever(
            store=store,
            doc_repo=repo,
            dense_encoder=lambda q: _unit_vec(16, 1),
            sparse_encoder=lambda q: {"营业收入": 0.9},
        )
        filters = RetrievalFilters(company_name="浦发银行")
        child_res = retriever.retrieve_candidates_by_company_name(
            "浦发银行", "营业收入", top_n=2, parent_retrieval_mode="child", filters=filters
        )
        self.assertEqual(child_res[0]["result_scope"], "child")
        self.assertEqual(child_res[0]["page"], 19)

        page_res = retriever.retrieve_candidates_by_company_name(
            "浦发银行", "营业收入", top_n=2, parent_retrieval_mode="page", filters=filters
        )
        self.assertEqual(page_res[0]["result_scope"], "page")
        self.assertIn("同比下降1.55%", page_res[0]["text"])

        block_res = retriever.retrieve_candidates_by_company_name(
            "浦发银行", "营业收入", top_n=2, parent_retrieval_mode="block", filters=filters
        )
        self.assertEqual(block_res[0]["result_scope"], "parent")
        self.assertIn("父块", block_res[0]["text"])
        self.assertTrue(block_res[0]["matched_child_chunk_ids"])


if __name__ == "__main__":
    unittest.main()
