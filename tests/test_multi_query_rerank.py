import unittest
from unittest.mock import Mock

from src.query_plan import QueryPlan
from src.questions_processing import QuestionsProcessor
from src.retrieval_filters import RetrievalFilters


def _make_result(
    *,
    page: int,
    chunk_id: int,
    score: float,
    source: str,
    doc_id: str = "alpha-sha",
    text: str | None = None,
    matched_child_chunk_ids=None,
    matched_tags=None,
):
    return {
        "page": page,
        "text": text or f"Evidence block on page {page}",
        "distance": score,
        "ranking_score": score,
        "chunk_id": chunk_id,
        "chunk_type": "content",
        "result_scope": "child",
        "matched_child_chunk_ids": list(matched_child_chunk_ids or []),
        "matched_tags": list(matched_tags or []),
        "retrieval_sources": [source],
        "metadata": {
            "chunk_id": chunk_id,
            "chunk_type": "content",
            "node_type": "child",
            "parent_chunk_id": None,
            "section_title": "Management Discussion",
            "section_name": "管理层讨论与分析",
            "report_section": "管理层讨论与分析",
            "company_name": "Alpha Corp",
            "stock_code": "600000",
            "currency": "CNY",
            "report_year": 2024,
            "report_type": "annual",
            "doc_source_type": "annual_report",
            "topic_flags": [],
            "sha1_name": doc_id,
        },
    }


class FakeHybridRetriever:
    def __init__(self, candidate_map=None, single_query_results=None, rerank_debug_override=None):
        self.candidate_map = candidate_map or {}
        self.single_query_results = single_query_results or {}
        self.rerank_debug_override = rerank_debug_override or {}
        self.candidate_calls = []
        self.rerank_calls = []
        self.retrieve_calls = []
        self.last_rerank_debug = {}

    def retrieve_candidates_by_company_name(
        self,
        company_name: str,
        query: str,
        top_n: int = 28,
        parent_retrieval_mode: str = "child",
        filters=None,
        candidate_doc_ids=None,
        backend_scope: str = "all",
    ):
        self.candidate_calls.append(
            {
                "company_name": company_name,
                "query": query,
                "top_n": top_n,
                "parent_retrieval_mode": parent_retrieval_mode,
                "candidate_doc_ids": list(candidate_doc_ids or []),
                "backend_scope": backend_scope,
            }
        )
        return [item.copy() for item in self.candidate_map.get(query, [])][:top_n]

    def rerank_candidates(
        self,
        query: str,
        candidate_results,
        documents_batch_size: int = 2,
        top_n: int = 6,
        llm_weight: float = 0.7,
    ):
        self.rerank_calls.append(
            {
                "query": query,
                "pages": [item["page"] for item in candidate_results],
                "documents_batch_size": documents_batch_size,
                "top_n": top_n,
                "candidate_results": [item.copy() for item in candidate_results],
            }
        )
        reranked = []
        for item in candidate_results:
            scored = item.copy()
            scored["combined_score"] = round(float(item.get("distance", 0.0)) + 0.1, 4)
            if self.rerank_debug_override.get("reranking_strategy") == "cascade":
                scored["distance_rrf"] = round(float(item.get("distance", 0.0)), 4)
                scored["colbert_score"] = round(float(item.get("distance", 0.0)), 4)
                scored["final_relevance_score"] = round(float(item.get("distance", 0.0)) + 0.2, 4)
            reranked.append(scored)
        reranked.sort(key=lambda item: item["combined_score"], reverse=True)
        self.last_rerank_debug = self.rerank_debug_override or {
            "reranking_strategy": "single",
            "initial_candidate_pool_size": len(candidate_results),
            "colbert_candidate_pool_size": None,
            "colbert_top_n": None,
            "final_reranking_backend": "flag_embedding",
        }
        return reranked[:top_n]

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 2,
        top_n: int = 6,
        llm_weight: float = 0.7,
        parent_retrieval_mode: str = "child",
        filters=None,
        candidate_doc_ids=None,
    ):
        self.retrieve_calls.append(
            {
                "company_name": company_name,
                "query": query,
                "llm_reranking_sample_size": llm_reranking_sample_size,
                "top_n": top_n,
                "parent_retrieval_mode": parent_retrieval_mode,
            }
        )
        return [item.copy() for item in self.single_query_results.get(query, [])][:top_n]


class MultiQueryRerankTests(unittest.TestCase):
    def test_merge_multi_query_candidates_tracks_provenance_and_respects_pool_cap(self):
        processor = QuestionsProcessor()
        retrieval_runs = [
            (
                "query one",
                [
                    _make_result(page=1, chunk_id=11, score=0.61, source="vector", matched_child_chunk_ids=[101]),
                    _make_result(page=2, chunk_id=22, score=0.52, source="bm25"),
                ],
            ),
            (
                "query two",
                [
                    _make_result(page=1, chunk_id=11, score=0.83, source="sparse", matched_child_chunk_ids=[102]),
                    _make_result(page=3, chunk_id=33, score=0.57, source="vector"),
                ],
            ),
            (
                "query three",
                [
                    _make_result(page=1, chunk_id=11, score=0.72, source="tag", matched_tags=["国产替代"]),
                    _make_result(page=4, chunk_id=44, score=0.21, source="vector"),
                ],
            ),
        ]

        merged = processor.merge_multi_query_candidates(retrieval_runs, pool_cap=3)

        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["page"], 1)
        self.assertEqual(merged[0]["matched_queries"], ["query one", "query two", "query three"])
        self.assertEqual(merged[0]["query_hit_count"], 3)
        self.assertEqual(merged[0]["retrieval_sources"], ["sparse", "tag", "vector"])
        self.assertEqual(merged[0]["matched_child_chunk_ids"], [101, 102])
        self.assertEqual(merged[0]["matched_tags"], ["国产替代"])
        self.assertEqual([item["page"] for item in merged], [1, 3, 2])

    def test_multi_query_global_rerank_uses_original_question_once(self):
        processor = QuestionsProcessor(
            llm_reranking=True,
            top_n_retrieval=2,
            llm_reranking_sample_size=8,
            parallel_requests=1,
        )
        retriever = FakeHybridRetriever(
            candidate_map={
                "orig": [
                    _make_result(page=1, chunk_id=11, score=0.71, source="vector"),
                    _make_result(page=2, chunk_id=22, score=0.44, source="bm25"),
                ],
                "alias": [
                    _make_result(page=2, chunk_id=22, score=0.93, source="sparse"),
                    _make_result(page=3, chunk_id=33, score=0.62, source="vector"),
                ],
                "fallback": [
                    _make_result(page=3, chunk_id=33, score=0.66, source="tag"),
                ],
            }
        )

        query_plan = QueryPlan(
            original_query="original user question",
            normalized_query="original user question",
            search_queries=["orig", "alias", "fallback"],
            filters=RetrievalFilters(company_name="Alpha Corp", question_kind="name"),
            route_mode="explicit_company",
            expected_answer_type="entity",
        )

        processor._build_retriever = lambda: (retriever, "hybrid_rerank")
        processor.api_processor.get_answer_from_rag_context = Mock(
            return_value={
                "final_answer": "Alpha",
                "relevant_pages": [2, 3],
                "reasoning_summary": "Grounded answer.",
                "step_by_step_analysis": "Grounded answer.",
            }
        )

        answer = processor.get_answer_for_company(
            company_name="Alpha Corp",
            question="original user question",
            schema="name",
            query_plan=query_plan,
            route_info={"route_mode": "explicit_company", "selected_report": {"sha1": "alpha-sha"}},
        )

        self.assertEqual(len(retriever.candidate_calls), 3)
        self.assertEqual(len(retriever.rerank_calls), 1)
        self.assertEqual(retriever.rerank_calls[0]["query"], "original user question")
        self.assertEqual(answer["candidate_pool_size_before_rerank"], 3)
        self.assertEqual(answer["reranking_strategy"], "single")
        self.assertCountEqual(retriever.rerank_calls[0]["pages"], [1, 2, 3])
        self.assertEqual([item["page"] for item in answer["retrieval_results"]], [2, 1])
        second_result = answer["retrieval_results"][0]
        self.assertEqual(second_result["matched_queries"], ["orig", "alias"])
        self.assertEqual(second_result["query_hit_count"], 2)
        self.assertEqual(second_result["retrieval_sources"], ["bm25", "sparse"])

    def test_multi_query_cascade_debug_fields_are_exposed(self):
        processor = QuestionsProcessor(
            llm_reranking=True,
            top_n_retrieval=2,
            llm_reranking_sample_size=8,
            parallel_requests=1,
            reranking_strategy="cascade",
            cascade_candidate_pool_cap=24,
            colbert_top_n=8,
        )
        retriever = FakeHybridRetriever(
            candidate_map={
                "orig": [
                    _make_result(page=1, chunk_id=11, score=0.71, source="vector"),
                    _make_result(page=2, chunk_id=22, score=0.44, source="bm25"),
                ],
                "alias": [
                    _make_result(page=3, chunk_id=33, score=0.62, source="vector"),
                ],
            },
            rerank_debug_override={
                "reranking_strategy": "cascade",
                "initial_candidate_pool_size": 3,
                "colbert_candidate_pool_size": 2,
                "colbert_top_n": 2,
                "final_reranking_backend": "flag_embedding",
            },
        )

        query_plan = QueryPlan(
            original_query="original user question",
            normalized_query="original user question",
            search_queries=["orig", "alias"],
            filters=RetrievalFilters(company_name="Alpha Corp", question_kind="name"),
            route_mode="explicit_company",
            expected_answer_type="entity",
        )

        processor._build_retriever = lambda: (retriever, "hybrid_rerank")
        processor.api_processor.get_answer_from_rag_context = Mock(
            return_value={
                "final_answer": "Alpha",
                "relevant_pages": [1],
                "reasoning_summary": "Grounded answer.",
                "step_by_step_analysis": "Grounded answer.",
            }
        )

        answer = processor.get_answer_for_company(
            company_name="Alpha Corp",
            question="original user question",
            schema="name",
            query_plan=query_plan,
            route_info={"route_mode": "explicit_company", "selected_report": {"sha1": "alpha-sha"}},
        )

        self.assertEqual(answer["reranking_strategy"], "cascade")
        self.assertEqual(answer["initial_candidate_pool_size"], 3)
        self.assertEqual(answer["colbert_candidate_pool_size"], 2)
        self.assertEqual(answer["colbert_top_n"], 2)
        self.assertEqual(answer["final_reranking_backend"], "flag_embedding")
        self.assertIsNotNone(answer["retrieval_results"][0]["distance_rrf"])
        self.assertIsNotNone(answer["retrieval_results"][0]["colbert_score"])
        self.assertIsNotNone(answer["retrieval_results"][0]["final_relevance_score"])

    def test_single_query_hybrid_rerank_keeps_existing_path(self):
        processor = QuestionsProcessor(
            llm_reranking=True,
            top_n_retrieval=2,
            llm_reranking_sample_size=8,
            parallel_requests=1,
        )
        retriever = FakeHybridRetriever(
            single_query_results={
                "orig": [
                    {
                        **_make_result(page=2, chunk_id=22, score=0.91, source="vector"),
                        "combined_score": 0.96,
                    }
                ]
            }
        )
        query_plan = QueryPlan(
            original_query="original user question",
            normalized_query="original user question",
            search_queries=["orig"],
            filters=RetrievalFilters(company_name="Alpha Corp", question_kind="name"),
            route_mode="explicit_company",
            expected_answer_type="entity",
        )

        processor._build_retriever = lambda: (retriever, "hybrid_rerank")
        processor.api_processor.get_answer_from_rag_context = Mock(
            return_value={
                "final_answer": "Alpha",
                "relevant_pages": [2],
                "reasoning_summary": "Grounded answer.",
                "step_by_step_analysis": "Grounded answer.",
            }
        )

        answer = processor.get_answer_for_company(
            company_name="Alpha Corp",
            question="original user question",
            schema="name",
            query_plan=query_plan,
            route_info={"route_mode": "explicit_company", "selected_report": {"sha1": "alpha-sha"}},
        )

        self.assertEqual(len(retriever.retrieve_calls), 1)
        self.assertEqual(retriever.retrieve_calls[0]["query"], "orig")
        self.assertEqual(len(retriever.candidate_calls), 0)
        self.assertEqual(len(retriever.rerank_calls), 0)
        self.assertIsNone(answer["candidate_pool_size_before_rerank"])

    def test_multi_query_without_rerank_keeps_merge_only_behavior(self):
        processor = QuestionsProcessor(
            llm_reranking=False,
            top_n_retrieval=2,
            parallel_requests=1,
        )
        retriever = FakeHybridRetriever(
            candidate_map={
                "orig": [_make_result(page=1, chunk_id=11, score=0.71, source="vector")],
                "alias": [_make_result(page=2, chunk_id=22, score=0.93, source="sparse")],
            }
        )
        query_plan = QueryPlan(
            original_query="original user question",
            normalized_query="original user question",
            search_queries=["orig", "alias"],
            filters=RetrievalFilters(company_name="Alpha Corp", question_kind="name"),
            route_mode="explicit_company",
            expected_answer_type="entity",
        )

        processor._build_retriever = lambda: (retriever, "hybrid")
        processor.api_processor.get_answer_from_rag_context = Mock(
            return_value={
                "final_answer": "Alpha",
                "relevant_pages": [1, 2],
                "reasoning_summary": "Grounded answer.",
                "step_by_step_analysis": "Grounded answer.",
            }
        )

        answer = processor.get_answer_for_company(
            company_name="Alpha Corp",
            question="original user question",
            schema="name",
            query_plan=query_plan,
            route_info={"route_mode": "explicit_company", "selected_report": {"sha1": "alpha-sha"}},
        )

        self.assertEqual(len(retriever.candidate_calls), 2)
        self.assertEqual(len(retriever.rerank_calls), 0)
        self.assertEqual([item["page"] for item in answer["retrieval_results"]], [2, 1])
        self.assertIsNone(answer["candidate_pool_size_before_rerank"])

    def test_multi_query_hyde_fallback_uses_dense_only_and_reranks_again(self):
        processor = QuestionsProcessor(
            llm_reranking=True,
            top_n_retrieval=2,
            llm_reranking_sample_size=8,
            parallel_requests=1,
            hyde_enabled=True,
            hyde_trigger_mode="fallback",
            hyde_top_score_threshold=0.55,
            hyde_margin_threshold=0.05,
        )
        retriever = FakeHybridRetriever(
            candidate_map={
                "orig": [
                    _make_result(page=1, chunk_id=11, score=0.21, source="vector"),
                    _make_result(page=2, chunk_id=22, score=0.16, source="bm25"),
                ],
                "alias": [
                    _make_result(page=3, chunk_id=33, score=0.19, source="sparse"),
                ],
                "hyde synthetic paragraph": [
                    _make_result(page=9, chunk_id=99, score=0.88, source="vector"),
                ],
            }
        )
        query_plan = QueryPlan(
            original_query="original user question",
            normalized_query="original user question",
            search_queries=["orig", "alias"],
            filters=RetrievalFilters(company_name="Alpha Corp", question_kind="boolean"),
            route_mode="explicit_company",
            expected_answer_type="boolean",
        )

        processor._build_retriever = lambda: (retriever, "hybrid_rerank")
        processor.hyde_generator = Mock()
        processor.hyde_generator.generate.return_value = "hyde synthetic paragraph"
        processor.api_processor.get_answer_from_rag_context = Mock(
            return_value={
                "final_answer": True,
                "relevant_pages": [9],
                "reasoning_summary": "Grounded answer.",
                "step_by_step_analysis": "Grounded answer.",
            }
        )

        answer = processor.get_answer_for_company(
            company_name="Alpha Corp",
            question="original user question",
            schema="boolean",
            query_plan=query_plan,
            route_info={"route_mode": "explicit_company", "selected_report": {"sha1": "alpha-sha"}},
        )

        self.assertEqual(len(retriever.candidate_calls), 3)
        self.assertEqual(retriever.candidate_calls[-1]["backend_scope"], "vector_only")
        self.assertEqual(retriever.candidate_calls[-1]["query"], "hyde synthetic paragraph")
        self.assertEqual(len(retriever.rerank_calls), 2)
        self.assertEqual(retriever.rerank_calls[0]["query"], "original user question")
        self.assertEqual(retriever.rerank_calls[1]["query"], "original user question")
        processor.hyde_generator.generate.assert_called_once()
        self.assertEqual(answer["candidate_pool_size_before_rerank"], 4)
        self.assertTrue(answer["hyde"]["triggered"])
        self.assertIn("low_top_score", answer["hyde"]["trigger_reasons"])
        self.assertEqual(answer["hyde"]["initial_candidate_pool_size"], 3)
        self.assertEqual(answer["hyde"]["final_candidate_pool_size"], 4)
        self.assertEqual(answer["hyde"]["generated_text"], "hyde synthetic paragraph")
        self.assertEqual([item["page"] for item in answer["retrieval_results"]], [9, 1])
        self.assertEqual(answer["retrieval_results"][0]["matched_queries"], ["__hyde__"])

    def test_numeric_multi_query_skips_hyde_even_when_first_pass_is_weak(self):
        processor = QuestionsProcessor(
            llm_reranking=True,
            top_n_retrieval=2,
            llm_reranking_sample_size=8,
            parallel_requests=1,
            hyde_enabled=True,
            hyde_trigger_mode="fallback",
            hyde_top_score_threshold=0.55,
            hyde_margin_threshold=0.05,
        )
        retriever = FakeHybridRetriever(
            candidate_map={
                "orig": [_make_result(page=1, chunk_id=11, score=0.21, source="vector")],
                "alias": [_make_result(page=2, chunk_id=22, score=0.19, source="bm25")],
            }
        )
        query_plan = QueryPlan(
            original_query="original user question",
            normalized_query="original user question",
            search_queries=["orig", "alias"],
            filters=RetrievalFilters(company_name="Alpha Corp", question_kind="number"),
            route_mode="explicit_company",
            expected_answer_type="numeric",
        )

        processor._build_retriever = lambda: (retriever, "hybrid_rerank")
        processor.hyde_generator = Mock()
        processor.api_processor.get_answer_from_rag_context = Mock(
            return_value={
                "final_answer": 42,
                "relevant_pages": [1],
                "reasoning_summary": "Grounded answer.",
                "step_by_step_analysis": "Grounded answer.",
            }
        )

        answer = processor.get_answer_for_company(
            company_name="Alpha Corp",
            question="original user question",
            schema="number",
            query_plan=query_plan,
            route_info={"route_mode": "explicit_company", "selected_report": {"sha1": "alpha-sha"}},
        )

        processor.hyde_generator.generate.assert_not_called()
        self.assertEqual(len(retriever.candidate_calls), 2)
        self.assertEqual(len(retriever.rerank_calls), 1)
        self.assertFalse(answer["hyde"]["triggered"])
        self.assertEqual(answer["hyde"]["initial_candidate_pool_size"], 2)

    def test_multi_query_rerank_keeps_expansion_only_hit_in_candidate_pool(self):
        processor = QuestionsProcessor(
            llm_reranking=True,
            top_n_retrieval=2,
            llm_reranking_sample_size=8,
            parallel_requests=1,
        )
        retriever = FakeHybridRetriever(
            candidate_map={
                "orig": [_make_result(page=1, chunk_id=11, score=0.61, source="vector")],
                "expanded": [_make_result(page=9, chunk_id=99, score=0.95, source="sparse")],
            }
        )
        query_plan = QueryPlan(
            original_query="dividend question",
            normalized_query="dividend question",
            search_queries=["orig", "expanded"],
            filters=RetrievalFilters(company_name="Alpha Corp", question_kind="boolean"),
            route_mode="explicit_company",
            expected_answer_type="boolean",
        )

        processor._build_retriever = lambda: (retriever, "hybrid_rerank")
        processor.api_processor.get_answer_from_rag_context = Mock(
            return_value={
                "final_answer": True,
                "relevant_pages": [9],
                "reasoning_summary": "Grounded answer.",
                "step_by_step_analysis": "Grounded answer.",
            }
        )

        answer = processor.get_answer_for_company(
            company_name="Alpha Corp",
            question="dividend question",
            schema="boolean",
            query_plan=query_plan,
            route_info={"route_mode": "explicit_company", "selected_report": {"sha1": "alpha-sha"}},
        )

        self.assertEqual(len(retriever.rerank_calls), 1)
        self.assertIn(9, retriever.rerank_calls[0]["pages"])
        self.assertEqual(answer["retrieval_results"][0]["page"], 9)


if __name__ == "__main__":
    unittest.main()
