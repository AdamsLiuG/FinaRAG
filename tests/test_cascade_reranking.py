import unittest
from unittest.mock import patch

import torch

from src.reranking import CascadeReranker, ColBERTReranker


class _FakeColBERTReranker:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def score_documents(self, query, documents):
        self.calls.append(
            {
                "query": query,
                "pages": [doc.get("page") for doc in documents],
            }
        )
        return self.scores[: len(documents)]


class _FakeFinalReranker:
    def __init__(self):
        self.calls = []

    def rerank_documents(self, query, documents, documents_batch_size=4, llm_weight=0.7):
        self.calls.append(
            {
                "query": query,
                "pages": [doc.get("page") for doc in documents],
                "documents_batch_size": documents_batch_size,
            }
        )
        reranked = []
        for doc in documents:
            item = doc.copy()
            item["relevance_score"] = round(float(doc.get("distance", 0.0)) + 0.05, 4)
            item["final_relevance_score"] = item["relevance_score"]
            item["combined_score"] = round(float(doc.get("distance", 0.0)) + 0.02, 4)
            reranked.append(item)
        reranked.sort(key=lambda item: item["combined_score"], reverse=True)
        return reranked


class _FakeBGEM3Model:
    def __init__(self, score_by_passage_vec):
        self.score_by_passage_vec = dict(score_by_passage_vec)
        self.encode_calls = []
        self.colbert_calls = []

    def encode(
        self,
        texts,
        batch_size=1,
        max_length=None,
        return_dense=False,
        return_sparse=False,
        return_colbert_vecs=True,
    ):
        self.encode_calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "max_length": max_length,
                "return_dense": return_dense,
                "return_sparse": return_sparse,
                "return_colbert_vecs": return_colbert_vecs,
            }
        )
        return {"colbert_vecs": [f"vec::{text}" for text in texts]}

    def colbert_score(self, query_vecs, passage_vecs):
        self.colbert_calls.append(
            {
                "query_vecs": query_vecs,
                "passage_vecs": passage_vecs,
            }
        )
        return self.score_by_passage_vec[passage_vecs]


def _doc(page, distance):
    return {
        "page": page,
        "text": f"Evidence on page {page}",
        "distance": distance,
        "ranking_score": distance,
        "metadata": {"sha1_name": "alpha-sha", "chunk_id": page},
    }


class CascadeRerankingTests(unittest.TestCase):
    def test_bgem3_colbert_reranker_uses_colbert_vecs_and_colbert_score(self):
        fake_model = _FakeBGEM3Model(
            {
                "vec::Evidence on page 1": 0.2,
                "vec::Evidence on page 2": 0.9,
            }
        )

        with patch("src.reranking._load_bgem3_model", return_value=fake_model):
            reranker = ColBERTReranker(
                model_name="BAAI/bge-m3",
                device="cpu",
                batch_size=2,
                query_max_length=64,
                passage_max_length=256,
            )

        scores = reranker.score_documents("query", [_doc(1, 0.7), _doc(2, 0.6)])

        self.assertEqual(scores, [0.2, 0.9])
        self.assertEqual(len(fake_model.encode_calls), 2)
        self.assertEqual(fake_model.encode_calls[0]["texts"], ["query"])
        self.assertEqual(fake_model.encode_calls[1]["texts"], ["Evidence on page 1", "Evidence on page 2"])
        self.assertTrue(all(call["return_colbert_vecs"] for call in fake_model.encode_calls))
        self.assertTrue(all(not call["return_dense"] for call in fake_model.encode_calls))
        self.assertTrue(all(not call["return_sparse"] for call in fake_model.encode_calls))
        self.assertEqual(
            [call["passage_vecs"] for call in fake_model.colbert_calls],
            ["vec::Evidence on page 1", "vec::Evidence on page 2"],
        )

    def test_late_interaction_ignores_padding_and_special_tokens(self):
        query_embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [9.0, 9.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        query_mask = torch.tensor([True, False, True])
        doc_embeddings = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [7.0, 7.0]],
                [[0.0, 1.0], [1.0, 0.0], [7.0, 7.0]],
            ],
            dtype=torch.float32,
        )
        doc_mask = torch.tensor(
            [
                [True, True, False],
                [True, True, False],
            ]
        )

        scores = ColBERTReranker._late_interaction_scores(query_embeddings, query_mask, doc_embeddings, doc_mask)

        self.assertEqual([round(float(score), 4) for score in scores.tolist()], [2.0, 2.0])

    def test_cascade_reranker_applies_pool_cap_then_colbert_cut_then_final_rerank(self):
        cascade = CascadeReranker(
            colbert_model="dummy-colbert",
            cascade_candidate_pool_cap=4,
            colbert_top_n=2,
            final_reranker=_FakeFinalReranker(),
            final_reranking_backend="flag_embedding",
            colbert_reranker=_FakeColBERTReranker([0.1, 0.9, 0.4, 0.8]),
        )
        documents = [
            _doc(1, 0.95),
            _doc(2, 0.93),
            _doc(3, 0.91),
            _doc(4, 0.89),
            _doc(5, 0.87),
        ]

        results = cascade.rerank_documents("query", documents, documents_batch_size=3, llm_weight=0.7)

        self.assertEqual(cascade.last_debug["reranking_strategy"], "cascade")
        self.assertEqual(cascade.last_debug["initial_candidate_pool_size"], 4)
        self.assertEqual(cascade.last_debug["colbert_candidate_pool_size"], 2)
        self.assertEqual(cascade.last_debug["colbert_top_n"], 2)
        self.assertEqual(cascade.last_debug["final_reranking_backend"], "flag_embedding")
        self.assertEqual([item["page"] for item in results], [2, 4])
        self.assertEqual(results[0]["distance_rrf"], 0.93)
        self.assertEqual(results[0]["colbert_score"], 1.0)
        self.assertEqual(results[1]["colbert_score"], round((0.8 - 0.1) / (0.9 - 0.1), 4))
        self.assertEqual(results[0]["final_relevance_score"], results[0]["relevance_score"])


if __name__ == "__main__":
    unittest.main()
