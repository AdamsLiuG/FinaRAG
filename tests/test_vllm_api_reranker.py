import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.reranking import VLLMApiReranker
from src.retrieval import HybridRetriever


class _FakeResponse:
    def __init__(self, payload, url, status_code=200):
        self._payload = payload
        self.url = url
        self.status_code = status_code
        self.text = str(payload)

    @property
    def ok(self):
        return 200 <= self.status_code < 400

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _doc(page, distance):
    return {
        "page": page,
        "text": f"Evidence block on page {page}",
        "distance": distance,
        "ranking_score": distance,
        "metadata": {"sha1_name": "alpha-sha", "chunk_id": page},
    }


class VLLMApiRerankerTests(unittest.TestCase):
    def test_uses_dedicated_reranking_base_url_for_private_vllm_service(self):
        docs = [_doc(1, 0.6), _doc(2, 0.4)]
        fake_response = _FakeResponse(
            payload={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            },
            url="http://192.168.1.200:9000/v1/rerank",
        )
        fake_session = _FakeSession(fake_response)

        with patch.dict(
            os.environ,
            {
                "RERANKING_BASE_URL": "http://192.168.1.200:9000/v1",
                "RERANKING_MODEL": "Qwen/Qwen3-Reranker-0.6B",
                "RERANKING_API_KEY": "secret-token",
            },
            clear=False,
        ):
            with patch("src.reranking.requests.Session", return_value=fake_session):
                reranker = VLLMApiReranker()
                results = reranker.rerank_documents("net profit", docs, llm_weight=0.7)

        self.assertFalse(fake_session.trust_env)
        self.assertEqual(len(fake_session.calls), 1)
        request = fake_session.calls[0]
        self.assertEqual(request["url"], "http://192.168.1.200:9000/v1/rerank")
        self.assertEqual(request["headers"]["Authorization"], "Bearer secret-token")
        self.assertEqual(request["json"]["model"], "Qwen/Qwen3-Reranker-0.6B")
        self.assertEqual(request["json"]["query"], "net profit")
        self.assertEqual(request["json"]["documents"], [doc["text"] for doc in docs])
        self.assertEqual(request["json"]["top_n"], 2)
        self.assertEqual([item["page"] for item in results], [2, 1])
        self.assertEqual(results[0]["relevance_score"], 0.9)
        self.assertEqual(results[0]["final_relevance_score"], 0.9)
        self.assertEqual(reranker.last_debug["final_reranking_backend"], "vllm_api")

    def test_normalizes_non_unit_scores_from_rerank_api(self):
        docs = [_doc(1, 0.6), _doc(2, 0.4)]
        fake_response = _FakeResponse(
            payload={
                "data": [
                    {"index": 0, "score": 5.0},
                    {"index": 1, "score": 3.0},
                ]
            },
            url="https://rerank.example/v1/rerank",
        )

        with patch.dict(
            os.environ,
            {
                "RERANKING_BASE_URL": "https://rerank.example",
                "RERANKING_MODEL": "Qwen/Qwen3-Reranker-0.6B",
            },
            clear=False,
        ):
            with patch("src.reranking.requests.post", return_value=fake_response) as mock_post:
                reranker = VLLMApiReranker()
                results = reranker.rerank_documents("revenue growth", docs, llm_weight=0.7)

        self.assertEqual(mock_post.call_args.kwargs["url"], "https://rerank.example/v1/rerank")
        self.assertEqual([item["relevance_score"] for item in results], [1.0, 0.0])
        self.assertEqual([item["page"] for item in results], [1, 2])

    def test_hybrid_retriever_accepts_vllm_api_as_final_backend(self):
        with patch("src.retrieval.VectorRetriever", return_value=object()):
            with patch("src.retrieval.HybridRetriever._build_reranker", return_value=object()):
                retriever = HybridRetriever(
                    documents_dir=Path("."),
                    vector_db_dir=Path("."),
                    use_vector_dbs=True,
                    use_bm25_db=False,
                    use_sparse_lexical_db=False,
                    use_tag_db=False,
                    final_reranking_backend="vllm_api",
                )

        self.assertEqual(retriever.final_reranking_backend, "vllm_api")

    def test_hybrid_retriever_defaults_cascade_colbert_model_to_bgem3(self):
        with patch.dict(os.environ, {"EMBEDDING_MODEL_NAME": "BAAI/bge-m3"}, clear=False):
            with patch("src.retrieval.VectorRetriever", return_value=object()):
                with patch("src.retrieval.HybridRetriever._build_final_reranker", return_value=object()):
                    with patch("src.retrieval.CascadeReranker", return_value=object()) as mock_cascade:
                        HybridRetriever(
                            documents_dir=Path("."),
                            vector_db_dir=Path("."),
                            use_vector_dbs=True,
                            use_bm25_db=False,
                            use_sparse_lexical_db=False,
                            use_tag_db=False,
                            reranking_strategy="cascade",
                            colbert_model=None,
                        )

        self.assertEqual(mock_cascade.call_args.kwargs["colbert_model"], "BAAI/bge-m3")

    def test_hybrid_retriever_uses_reranking_model_for_flag_embedding_cascade_final_stage(self):
        with patch.dict(
            os.environ,
            {
                "RERANKING_MODEL": "BAAI/bge-reranker-v2-m3",
            },
            clear=False,
        ):
            with patch("src.retrieval.VectorRetriever", return_value=object()):
                with patch("src.retrieval.HybridRetriever._build_final_reranker", return_value=object()) as mock_final:
                    with patch("src.retrieval.CascadeReranker", return_value=object()):
                        HybridRetriever(
                            documents_dir=Path("."),
                            vector_db_dir=Path("."),
                            use_vector_dbs=True,
                            use_bm25_db=False,
                            use_sparse_lexical_db=False,
                            use_tag_db=False,
                            reranking_strategy="cascade",
                            final_reranking_backend="flag_embedding",
                            model="Qwopus3.5-27B-v3",
                        )

        self.assertEqual(
            mock_final.call_args.args,
            ("flag_embedding", "qwen", "BAAI/bge-reranker-v2-m3"),
        )


if __name__ == "__main__":
    unittest.main()
