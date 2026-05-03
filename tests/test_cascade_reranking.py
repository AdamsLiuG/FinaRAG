import unittest
from unittest.mock import patch

import torch

from src.reranking import CascadeReranker, ColBERTReranker, LLMReranker


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


class _FakeBatchEncoding(dict):
    def to(self, device):
        self["__device__"] = device
        return self


class _FakeTokenizer:
    special_tokens_map = {
        "cls_token": "[CLS]",
        "eos_token": "[SEP]",
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
    }

    _token_ids = {
        "[PAD]": 0,
        "[UNK]": 100,
        "[CLS]": 101,
        "[SEP]": 102,
    }

    def __init__(self, token_id_by_text):
        self.token_id_by_text = dict(token_id_by_text)

    def convert_tokens_to_ids(self, token):
        return self._token_ids[token]

    def __call__(self, texts, padding=True, truncation=True, max_length=None, return_tensors="pt"):
        text_list = [texts] if isinstance(texts, str) else list(texts)
        rows = []
        masks = []
        max_tokens = 0

        for text in text_list:
            token_id = int(self.token_id_by_text.get(text, len(str(text)) + 1000))
            tokens = [101, token_id, 102]
            rows.append(tokens)
            max_tokens = max(max_tokens, len(tokens))

        for index, tokens in enumerate(rows):
            mask = [1] * len(tokens)
            if padding:
                pad_length = max_tokens - len(tokens)
                if pad_length > 0:
                    rows[index] = tokens + [0] * pad_length
                    mask = mask + [0] * pad_length
            masks.append(mask)

        return _FakeBatchEncoding(
            {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        )


class _FakeParameter:
    def __init__(self):
        self.device = "cpu"
        self.dtype = torch.float32


class _FakeInferenceModel:
    def __init__(self, *, max_supported_batch_size: int | None = None):
        self.parameter = _FakeParameter()
        self.max_supported_batch_size = max_supported_batch_size
        self.forward_calls = []

    def parameters(self):
        return iter([self.parameter])

    def half(self):
        self.parameter.dtype = torch.float16
        return self

    def float(self):
        self.parameter.dtype = torch.float32
        return self

    def to(self, device):
        self.parameter.device = device
        return self

    def eval(self):
        return self

    def __call__(self, inputs, return_dense=False, return_sparse=False, return_colbert_vecs=False):
        batch_size = int(inputs["input_ids"].shape[0])
        self.forward_calls.append(
            {
                "batch_size": batch_size,
                "device": self.parameter.device,
                "dtype": self.parameter.dtype,
            }
        )
        if self.max_supported_batch_size is not None and batch_size > self.max_supported_batch_size:
            raise RuntimeError("synthetic batch overflow")

        outputs = {}
        if return_colbert_vecs:
            colbert_tokens = inputs["input_ids"][:, 1:].float()
            outputs["colbert_vecs"] = torch.stack([colbert_tokens, colbert_tokens / 100.0], dim=-1)
        return outputs


class _FakeBGEM3Model:
    def __init__(self, token_id_by_text, *, device="cpu", max_supported_batch_size: int | None = None):
        self.target_devices = [device]
        self.use_fp16 = str(device).startswith("cuda")
        self.tokenizer = _FakeTokenizer(token_id_by_text)
        self.model = _FakeInferenceModel(max_supported_batch_size=max_supported_batch_size)
        self.colbert_calls = []

    def encode(self, *args, **kwargs):  # pragma: no cover - any call means the fix regressed
        raise AssertionError("ColBERTReranker should not call FlagEmbedding.encode anymore")

    def colbert_score(self, query_vecs, passage_vecs):
        self.colbert_calls.append(
            {
                "query_vecs": query_vecs,
                "passage_vecs": passage_vecs,
            }
        )
        return float(passage_vecs[0][0] / 100.0)


def _doc(page, distance):
    return {
        "page": page,
        "text": f"Evidence on page {page}",
        "distance": distance,
        "ranking_score": distance,
        "metadata": {"sha1_name": "alpha-sha", "chunk_id": page},
    }


class CascadeRerankingTests(unittest.TestCase):
    def test_bgem3_colbert_reranker_uses_direct_forward_and_colbert_score(self):
        fake_model = _FakeBGEM3Model(
            {
                "query": 5,
                "Evidence on page 1": 20,
                "Evidence on page 2": 90,
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
        self.assertEqual([call["batch_size"] for call in fake_model.model.forward_calls], [1, 2])
        self.assertEqual(
            [round(float(call["passage_vecs"][0][0]), 4) for call in fake_model.colbert_calls],
            [20.0, 90.0],
        )

    def test_bgem3_colbert_reranker_reduces_passage_batch_size_without_hitting_empty_pad_path(self):
        fake_model = _FakeBGEM3Model(
            {
                "query": 5,
                "Evidence on page 1": 20,
                "Evidence on page 2": 90,
            },
            max_supported_batch_size=1,
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
        self.assertEqual([call["batch_size"] for call in fake_model.model.forward_calls], [1, 2, 1, 1])

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
            final_reranking_batch_size=2,
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

    def test_cascade_reranker_uses_configured_final_reranking_batch_size(self):
        final_reranker = _FakeFinalReranker()
        cascade = CascadeReranker(
            colbert_model="dummy-colbert",
            cascade_candidate_pool_cap=4,
            colbert_top_n=2,
            final_reranking_batch_size=8,
            final_reranker=final_reranker,
            final_reranking_backend="llm_prompt",
            colbert_reranker=_FakeColBERTReranker([0.9, 0.8, 0.7, 0.6]),
        )
        documents = [
            _doc(1, 0.95),
            _doc(2, 0.93),
            _doc(3, 0.91),
            _doc(4, 0.89),
        ]

        cascade.rerank_documents("query", documents, documents_batch_size=3, llm_weight=0.7)

        self.assertEqual(final_reranker.calls[0]["documents_batch_size"], 8)

    def test_llm_reranker_falls_back_to_single_block_when_batch_parse_fails(self):
        reranker = LLMReranker.__new__(LLMReranker)
        reranker.get_rank_for_multiple_blocks = lambda query, texts: (_ for _ in ()).throw(
            ValueError("Structured response parsing failed")
        )
        reranker.get_rank_for_single_block = lambda query, text: {
            "relevance_score": 0.9 if "page 1" in text else 0.4
        }

        results = reranker.rerank_documents(
            "query",
            [_doc(1, 0.2), _doc(2, 0.1)],
            documents_batch_size=2,
            llm_weight=0.7,
        )

        self.assertEqual([item["page"] for item in results], [1, 2])
        self.assertEqual([round(item["relevance_score"], 4) for item in results], [0.9, 0.4])
        self.assertGreater(results[0]["combined_score"], results[1]["combined_score"])

    def test_llm_reranker_backfills_missing_batch_rankings_with_single_block_scores(self):
        reranker = LLMReranker.__new__(LLMReranker)
        reranker.get_rank_for_multiple_blocks = lambda query, texts: {
            "block_rankings": [
                {"reasoning": "Block 1", "relevance_score": 0.8},
            ]
        }
        reranker.get_rank_for_single_block = lambda query, text: {
            "relevance_score": 0.6 if "page 2" in text else 0.1
        }

        results = reranker.rerank_documents(
            "query",
            [_doc(1, 0.2), _doc(2, 0.1)],
            documents_batch_size=2,
            llm_weight=0.7,
        )

        by_page = {item["page"]: item for item in results}
        self.assertEqual(round(by_page[1]["relevance_score"], 4), 0.8)
        self.assertEqual(round(by_page[2]["relevance_score"], 4), 0.6)


if __name__ == "__main__":
    unittest.main()
