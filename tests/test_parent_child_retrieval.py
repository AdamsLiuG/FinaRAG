import json
import pickle
import tempfile
import unittest
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.retrieval import BM25Retriever, HybridRetriever
from src.text_normalization import tokenize_for_bm25


def _write_bm25_fixture(base_dir: Path, *, include_parent_schema: bool) -> tuple[Path, Path]:
    documents_dir = base_dir / "documents"
    bm25_dir = base_dir / "bm25"
    documents_dir.mkdir()
    bm25_dir.mkdir()

    chunks = [
        {
            "page": 1,
            "text": "revenue growth remained strong",
            "id": 0,
            "chunk_id": 0,
            "type": "content",
            "chunk_type": "content",
            "node_type": "child",
            "section_title": "Overview",
            "report_section": "Overview",
            "parent_block_id": "page1_block0",
            "parent_chunk_id": 0,
        },
        {
            "page": 1,
            "text": "revenue guidance and growth outlook improved",
            "id": 1,
            "chunk_id": 1,
            "type": "content",
            "chunk_type": "content",
            "node_type": "child",
            "section_title": "Overview",
            "report_section": "Overview",
            "parent_block_id": "page1_block0",
            "parent_chunk_id": 0,
        },
        {
            "page": 2,
            "text": "debt covenant remained unchanged",
            "id": 2,
            "chunk_id": 2,
            "type": "content",
            "chunk_type": "content",
            "node_type": "child",
            "section_title": "Debt",
            "report_section": "Debt",
            "parent_block_id": "page2_block0",
            "parent_chunk_id": 1,
        },
    ]
    if not include_parent_schema:
        for chunk in chunks:
            chunk.pop("parent_chunk_id")
            chunk.pop("node_type")

    document = {
        "metainfo": {
            "company_name": "Alpha Corp",
            "sha1_name": "alpha-sha",
            "currency": "USD",
        },
        "content": {
            "pages": [
                {"page": 1, "text": "Full page one context."},
                {"page": 2, "text": "Full page two context."},
            ],
            "chunks": chunks,
        },
    }
    if include_parent_schema:
        document["content"]["parent_chunks"] = [
            {
                "page": 1,
                "text": "Parent revenue block",
                "id": 0,
                "chunk_id": 0,
                "type": "content",
                "chunk_type": "content",
                "node_type": "parent",
                "section_title": "Overview",
                "report_section": "Overview",
                "parent_block_id": "page1_block0",
                "child_chunk_ids": [0, 1],
            },
            {
                "page": 2,
                "text": "Parent debt block",
                "id": 1,
                "chunk_id": 1,
                "type": "content",
                "chunk_type": "content",
                "node_type": "parent",
                "section_title": "Debt",
                "report_section": "Debt",
                "parent_block_id": "page2_block0",
                "child_chunk_ids": [2],
            },
        ]

    (documents_dir / "alpha.json").write_text(json.dumps(document), encoding="utf-8")

    tokenized_chunks = [tokenize_for_bm25(chunk["text"]) for chunk in document["content"]["chunks"]]
    with open(bm25_dir / "alpha-sha.pkl", "wb") as f:
        pickle.dump(BM25Okapi(tokenized_chunks), f)

    return documents_dir, bm25_dir


class ParentChildRetrievalTests(unittest.TestCase):
    def test_block_mode_aggregates_children_into_single_parent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir, bm25_dir = _write_bm25_fixture(Path(tmp_dir), include_parent_schema=True)
            retriever = BM25Retriever(bm25_db_dir=bm25_dir, documents_dir=documents_dir)

            results = retriever.retrieve_by_company_name(
                company_name="Alpha Corp",
                query="revenue growth",
                top_n=1,
                parent_retrieval_mode="block",
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["chunk_id"], 0)
            self.assertEqual(results[0]["result_scope"], "parent")
            self.assertEqual(results[0]["metadata"]["node_type"], "parent")
            self.assertCountEqual(results[0]["matched_child_chunk_ids"], [0, 1])
            self.assertEqual(results[0]["text"], "Parent revenue block")

    def test_page_mode_keeps_old_page_expansion_behavior(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir, bm25_dir = _write_bm25_fixture(Path(tmp_dir), include_parent_schema=False)
            retriever = BM25Retriever(bm25_db_dir=bm25_dir, documents_dir=documents_dir)

            results = retriever.retrieve_by_company_name(
                company_name="Alpha Corp",
                query="revenue growth",
                top_n=1,
                parent_retrieval_mode="page",
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["result_scope"], "page")
            self.assertEqual(results[0]["text"], "Full page one context.")

    def test_block_mode_requires_reprocessed_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            documents_dir, bm25_dir = _write_bm25_fixture(Path(tmp_dir), include_parent_schema=False)
            retriever = BM25Retriever(bm25_db_dir=bm25_dir, documents_dir=documents_dir)

            with self.assertRaisesRegex(ValueError, "Please re-run `process-reports`"):
                retriever.retrieve_by_company_name(
                    company_name="Alpha Corp",
                    query="revenue growth",
                    top_n=1,
                    parent_retrieval_mode="block",
                )


class HybridMergeTests(unittest.TestCase):
    def test_merge_results_dedupes_by_parent_identity(self):
        hybrid = HybridRetriever.__new__(HybridRetriever)
        hybrid.fusion_method = "rrf"
        hybrid.rrf_k = 60

        vector_results = [
            {
                "distance": 0.9,
                "page": 1,
                "text": "Parent block version A",
                "chunk_id": 7,
                "chunk_type": "content",
                "metadata": {
                    "sha1_name": "alpha-sha",
                    "chunk_id": 7,
                    "chunk_type": "content",
                    "node_type": "parent",
                },
                "matched_child_chunk_ids": [1],
                "retrieval_sources": ["vector"],
                "result_scope": "parent",
            }
        ]
        bm25_results = [
            {
                "distance": 2.8,
                "page": 1,
                "text": "Parent block version B",
                "chunk_id": 7,
                "chunk_type": "content",
                "metadata": {
                    "sha1_name": "alpha-sha",
                    "chunk_id": 7,
                    "chunk_type": "content",
                    "node_type": "parent",
                },
                "matched_child_chunk_ids": [2],
                "retrieval_sources": ["bm25"],
                "result_scope": "parent",
            }
        ]

        merged = hybrid._merge_retrieval_results(
            {
                "vector": vector_results,
                "bm25": bm25_results,
            },
            top_n=5,
        )

        self.assertEqual(len(merged), 1)
        self.assertCountEqual(merged[0]["matched_child_chunk_ids"], [1, 2])
        self.assertEqual(sorted(merged[0]["retrieval_sources"]), ["bm25", "vector"])


if __name__ == "__main__":
    unittest.main()
