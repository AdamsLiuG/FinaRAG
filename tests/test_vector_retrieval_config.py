import unittest
from unittest.mock import patch

import faiss
import numpy as np

from src.ingestion import VectorDBIngestor
from src.questions_processing import QuestionsProcessor
from src.retrieval import VectorRetriever


class VectorRetrievalConfigTests(unittest.TestCase):
    def test_vector_db_ingestor_builds_flat_index_by_default(self):
        ingestor = object.__new__(VectorDBIngestor)
        ingestor.index_type = "flat"
        ingestor.hnsw_m = 32
        ingestor.hnsw_ef_construction = 200

        index = ingestor._create_vector_db([[1.0, 0.0], [0.0, 1.0]])

        self.assertEqual(type(index).__name__, "IndexFlatIP")
        self.assertEqual(index.metric_type, faiss.METRIC_INNER_PRODUCT)
        self.assertEqual(index.ntotal, 2)

    def test_vector_db_ingestor_can_build_hnsw_index(self):
        ingestor = object.__new__(VectorDBIngestor)
        ingestor.index_type = "hnsw"
        ingestor.ivf_nlist = 32
        ingestor.hnsw_m = 24
        ingestor.hnsw_ef_construction = 96

        index = ingestor._create_vector_db([[1.0, 0.0], [0.0, 1.0]])

        self.assertEqual(type(index).__name__, "IndexHNSWFlat")
        self.assertEqual(index.metric_type, faiss.METRIC_INNER_PRODUCT)
        self.assertEqual(index.hnsw.efConstruction, 96)
        self.assertEqual(index.ntotal, 2)

    def test_vector_db_ingestor_can_build_ivf_index(self):
        ingestor = object.__new__(VectorDBIngestor)
        ingestor.index_type = "ivf"
        ingestor.ivf_nlist = 4
        ingestor.hnsw_m = 32
        ingestor.hnsw_ef_construction = 200
        ingestor._resolve_ivf_nlist = VectorDBIngestor._resolve_ivf_nlist.__get__(ingestor, VectorDBIngestor)

        embeddings = [
            [float(i), float(i + 1), float(i % 7), float((i * 3) % 11)]
            for i in range(200)
        ]
        index = ingestor._create_vector_db(embeddings)

        self.assertEqual(type(index).__name__, "IndexIVFFlat")
        self.assertEqual(index.metric_type, faiss.METRIC_INNER_PRODUCT)
        self.assertTrue(index.is_trained)
        self.assertEqual(index.nlist, 4)
        self.assertEqual(index.ntotal, 200)

    def test_vector_retriever_resolves_search_k_for_child_and_parent_modes(self):
        retriever = object.__new__(VectorRetriever)
        retriever.vector_search_k = None

        self.assertEqual(retriever._resolve_search_k(top_n=4, num_chunks=100, parent_retrieval_mode="child"), 16)
        self.assertEqual(retriever._resolve_search_k(top_n=4, num_chunks=100, parent_retrieval_mode="block"), 32)

        retriever.vector_search_k = 6
        self.assertEqual(retriever._resolve_search_k(top_n=4, num_chunks=100, parent_retrieval_mode="child"), 6)

        retriever.vector_search_k = 2
        self.assertEqual(retriever._resolve_search_k(top_n=4, num_chunks=100, parent_retrieval_mode="block"), 4)

    def test_vector_retriever_configures_ivf_nprobe_safely(self):
        retriever = object.__new__(VectorRetriever)
        retriever.vector_search_k = None
        retriever.ivf_nprobe = 8
        retriever.hnsw_ef_search = 64

        embeddings = np.array(
            [[float(i), float(i + 1), float(i % 5), float((i * 2) % 7)] for i in range(200)],
            dtype="float32",
        )
        quantizer = faiss.IndexFlatIP(4)
        index = faiss.IndexIVFFlat(quantizer, 4, 4, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)

        configured = retriever._configure_index(index)

        self.assertEqual(configured.nprobe, 4)

    def test_questions_processor_caches_vector_retriever_when_enabled(self):
        processor = QuestionsProcessor(
            use_vector_dbs=True,
            use_bm25_db=False,
            use_sparse_lexical_db=False,
            retriever_cache_enabled=True,
        )

        with patch("src.questions_processing.VectorRetriever") as mock_vector_retriever:
            first_retriever = object()
            second_retriever = object()
            mock_vector_retriever.side_effect = [first_retriever, second_retriever]

            first_bundle = processor._build_retriever()
            second_bundle = processor._build_retriever()

        self.assertIs(first_bundle[0], second_bundle[0])
        self.assertEqual(first_bundle[1], "vector")
        self.assertEqual(mock_vector_retriever.call_count, 1)

    def test_questions_processor_rebuilds_vector_retriever_when_cache_disabled(self):
        processor = QuestionsProcessor(
            use_vector_dbs=True,
            use_bm25_db=False,
            use_sparse_lexical_db=False,
            retriever_cache_enabled=False,
        )

        with patch("src.questions_processing.VectorRetriever") as mock_vector_retriever:
            first_retriever = object()
            second_retriever = object()
            mock_vector_retriever.side_effect = [first_retriever, second_retriever]

            first_bundle = processor._build_retriever()
            second_bundle = processor._build_retriever()

        self.assertIsNot(first_bundle[0], second_bundle[0])
        self.assertEqual(first_bundle[1], "vector")
        self.assertEqual(second_bundle[1], "vector")
        self.assertEqual(mock_vector_retriever.call_count, 2)


if __name__ == "__main__":
    unittest.main()
