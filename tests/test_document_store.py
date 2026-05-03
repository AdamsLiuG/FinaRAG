import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.document_store import clear_document_store_cache, get_document_store
from src.report_catalog import ReportCatalog
from src.retrieval import BM25Retriever


def _write_document(documents_dir: Path, stem: str, company_name: str) -> None:
    payload = {
        "metainfo": {
            "company_name": company_name,
            "sha1_name": f"{stem}-sha",
            "doc_id": f"{stem}-sha",
        },
        "content": {
            "pages": [{"page": 1, "text": f"{company_name} page"}],
            "chunks": [{"page": 1, "text": f"{company_name} chunk", "chunk_id": 0, "id": 0}],
        },
    }
    (documents_dir / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


class DocumentStoreTests(unittest.TestCase):
    def tearDown(self):
        clear_document_store_cache()

    def test_document_store_invalidates_when_document_set_changes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            documents_dir = Path(tempdir) / "documents"
            documents_dir.mkdir()
            _write_document(documents_dir, "alpha", "Alpha Corp")

            first_store = get_document_store(documents_dir)
            self.assertEqual(len(first_store.documents), 1)

            _write_document(documents_dir, "beta", "Beta Corp")

            second_store = get_document_store(documents_dir)
            self.assertIsNot(first_store, second_store)
            self.assertEqual(len(second_store.documents), 2)
            self.assertIn("beta-sha", second_store.metainfo_by_doc_id)

    def test_report_catalog_and_retriever_reuse_loaded_documents(self):
        with tempfile.TemporaryDirectory() as tempdir:
            documents_dir = Path(tempdir) / "documents"
            bm25_dir = Path(tempdir) / "bm25"
            documents_dir.mkdir()
            bm25_dir.mkdir()
            _write_document(documents_dir, "alpha", "Alpha Corp")
            _write_document(documents_dir, "beta", "Beta Corp")

            import src.document_store as document_store_module

            original_json_load = document_store_module.json.load
            json_load_calls = 0

            def counting_json_load(*args, **kwargs):
                nonlocal json_load_calls
                json_load_calls += 1
                return original_json_load(*args, **kwargs)

            with patch("src.document_store.json.load", side_effect=counting_json_load):
                catalog = ReportCatalog(None, documents_dir)
                first_meta = catalog._load_document_meta()
                second_meta = catalog._load_document_meta()
                retriever = BM25Retriever(bm25_dir, documents_dir)

            self.assertEqual(len(first_meta), 2)
            self.assertEqual(first_meta, second_meta)
            self.assertEqual(len(retriever.documents), 2)
            self.assertEqual(json_load_calls, 2)


if __name__ == "__main__":
    unittest.main()
