import concurrent.futures
import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from src.answer_validation import validate_answer
from src.api_requests import APIProcessor
from src.citation_formatter import build_citations, compute_confidence, dedupe_citations, dedupe_references
from src.document_manifest import load_document_manifest
from src.hyde import HYDE_QUERY_MARKER, HyDEGenerator, should_trigger_hyde
from src.query_plan import QueryPlan
from src.query_rewrite import QuestionRewriter
from src.report_catalog import ReportCatalog
from src.retrieval import BM25Retriever, BGEM3SparseRetriever, HybridRetriever, TagRetriever, VectorRetriever
from src.table_grounding import TableGrounder


def _result_score(result: Dict) -> float:
    return float(result.get("combined_score", result.get("ranking_score", result.get("distance", 0.0))))


def _result_merge_key(result: Dict) -> tuple:
    metadata = result.get("metadata") or {}
    source_name = metadata.get("sha1_name")
    result_scope = result.get("result_scope") or metadata.get("node_type") or "child"
    chunk_id = result.get("chunk_id") or metadata.get("chunk_id")

    if result_scope == "page":
        return source_name, result_scope, result.get("page")
    if chunk_id is not None:
        return source_name, result_scope, chunk_id
    return source_name, result_scope, result.get("page"), result.get("text")


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = './vector_dbs',
        bm25_db_path: Optional[Union[str, Path]] = None,
        sparse_db_dir: Optional[Union[str, Path]] = None,
        tag_db_dir: Optional[Union[str, Path]] = None,
        documents_dir: Union[str, Path] = './documents',
        questions_file_path: Optional[Union[str, Path]] = None,
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,
        parent_retrieval_mode: str = "page",
        use_vector_dbs: bool = True,
        use_bm25_db: bool = False,
        use_sparse_lexical_db: bool = False,
        use_tag_db: bool = False,
        llm_reranking: bool = False,
        llm_reranking_sample_size: int = 20,
        top_n_retrieval: int = 10,
        vector_search_k: Optional[int] = None,
        vector_ivf_nprobe: int = 8,
        vector_hnsw_ef_search: int = 64,
        retriever_cache_enabled: bool = True,
        parallel_requests: int = 10,
        api_provider: str = "qwen",
        answering_model: str = "Qwen3.5-35B-A3B-AWQ-4bit",
        answer_temperature: float = 0.0,
        full_context: bool = False,
        document_language: str = "en",
        doc_router_enabled: bool = False,
        candidate_doc_top_k: int = 5,
        numeric_grounding_enabled: bool = False,
        reasoning_debug_enabled: bool = True,
        hyde_enabled: bool = False,
        hyde_trigger_mode: str = "off",
        hyde_generation_model: Optional[str] = None,
        hyde_generation_temperature: float = 0.2,
        hyde_max_tokens: int = 192,
        hyde_top_score_threshold: float = 0.55,
        hyde_margin_threshold: float = 0.05,
        reranking_strategy: str = "single",
        cascade_candidate_pool_cap: int = 50,
        colbert_top_n: int = 10,
        colbert_model: Optional[str] = None,
        colbert_device: Optional[str] = None,
        colbert_batch_size: int = 16,
        colbert_query_max_length: int = 128,
        colbert_passage_max_length: int = 512,
        final_reranking_backend: Optional[str] = None,
        final_reranking_model: Optional[str] = None,
    ):
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.bm25_db_path = Path(bm25_db_path) if bm25_db_path else None
        self.sparse_db_dir = Path(sparse_db_dir) if sparse_db_dir else None
        self.tag_db_dir = Path(tag_db_dir) if tag_db_dir else None
        self.subset_path = Path(subset_path) if subset_path else None

        resolved_parent_mode = (parent_retrieval_mode or "page").strip().lower()
        if resolved_parent_mode not in {"page", "block"}:
            raise ValueError("parent_retrieval_mode must be either 'page' or 'block'.")
        self.parent_retrieval_mode = resolved_parent_mode if parent_document_retrieval else "child"
        self.use_vector_dbs = use_vector_dbs
        self.use_bm25_db = use_bm25_db
        self.use_sparse_lexical_db = use_sparse_lexical_db
        self.use_tag_db = use_tag_db
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.vector_search_k = max(1, int(vector_search_k)) if vector_search_k else None
        self.vector_ivf_nprobe = max(1, int(vector_ivf_nprobe))
        self.vector_hnsw_ef_search = max(1, int(vector_hnsw_ef_search))
        self.retriever_cache_enabled = retriever_cache_enabled
        self.answering_model = answering_model
        self.answer_temperature = float(answer_temperature)
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.api_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context
        self.document_language = document_language
        self.doc_router_enabled = doc_router_enabled
        self.candidate_doc_top_k = max(1, candidate_doc_top_k)
        self.numeric_grounding_enabled = numeric_grounding_enabled
        self.reasoning_debug_enabled = reasoning_debug_enabled
        self.hyde_enabled = bool(hyde_enabled)
        self.hyde_trigger_mode = str(hyde_trigger_mode or "off").strip().lower()
        if self.hyde_trigger_mode not in {"off", "fallback"}:
            raise ValueError("hyde_trigger_mode must be either 'off' or 'fallback'.")
        self.hyde_generation_model = hyde_generation_model
        self.hyde_generation_temperature = float(hyde_generation_temperature)
        self.hyde_max_tokens = max(1, int(hyde_max_tokens))
        self.hyde_top_score_threshold = float(hyde_top_score_threshold)
        self.hyde_margin_threshold = float(hyde_margin_threshold)
        self.reranking_strategy = str(reranking_strategy or "single").strip().lower()
        if self.reranking_strategy not in {"single", "cascade"}:
            raise ValueError("reranking_strategy must be either 'single' or 'cascade'.")
        self.cascade_candidate_pool_cap = max(1, int(cascade_candidate_pool_cap))
        self.colbert_top_n = max(1, int(colbert_top_n))
        self.colbert_model = colbert_model
        self.colbert_device = colbert_device
        self.colbert_batch_size = max(1, int(colbert_batch_size))
        self.colbert_query_max_length = max(8, int(colbert_query_max_length))
        self.colbert_passage_max_length = max(16, int(colbert_passage_max_length))
        self.final_reranking_backend = (final_reranking_backend or "").strip().lower() or None
        self.final_reranking_model = final_reranking_model
        self.max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "8000"))
        self.max_doc_chars = int(os.getenv("RAG_MAX_DOC_CHARS", "2500"))

        self.answer_details: List[Optional[Dict]] = []
        self.response_data = {}
        self._lock = threading.Lock()
        self._retriever_cache = threading.local()
        self.question_rewriter = QuestionRewriter()
        self.companies_df: Optional[pd.DataFrame] = None
        self.report_catalog = ReportCatalog(self.subset_path, self.documents_dir) if self.subset_path else None
        self.table_grounder = TableGrounder(self.documents_dir) if self.numeric_grounding_enabled else None
        self.hyde_generator = (
            HyDEGenerator(
                provider=self.api_provider,
                model=self.hyde_generation_model or self.answering_model,
                temperature=self.hyde_generation_temperature,
                max_tokens=self.hyde_max_tokens,
                document_language=self.document_language,
            )
            if self.hyde_enabled
            else None
        )

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _load_companies_df(self) -> pd.DataFrame:
        if self.companies_df is None:
            if self.subset_path is None:
                raise ValueError("subset_path must be provided to use company metadata.")
            manifest = load_document_manifest(self.subset_path)
            if manifest:
                self.companies_df = pd.DataFrame(list(manifest.values()))
            else:
                self.companies_df = pd.read_csv(self.subset_path)

            if not self.companies_df.empty:
                if "sha1" not in self.companies_df.columns and "doc_id" in self.companies_df.columns:
                    self.companies_df["sha1"] = self.companies_df["doc_id"]
                if "sha1_name" not in self.companies_df.columns and "sha1" in self.companies_df.columns:
                    self.companies_df["sha1_name"] = self.companies_df["sha1"]
        return self.companies_df

    def _format_retrieval_results(self, retrieval_results: List[Dict]) -> str:
        if not retrieval_results:
            return ""

        context_parts = []
        total_chars = 0
        for result in retrieval_results:
            page_number = result['page']
            text = result['text']
            metadata = result.get("metadata", {})
            section_name = metadata.get("section_name") or metadata.get("section_title")
            chunk_type = metadata.get("chunk_type")
            node_type = metadata.get("node_type")
            matched_tags = result.get("matched_tags") or []

            if self.max_doc_chars > 0 and len(text) > self.max_doc_chars:
                text = text[: self.max_doc_chars].rstrip() + "\n...[truncated]"

            label = f"Text retrieved from page {page_number}"
            if section_name:
                label += f" | section: {section_name}"
            if chunk_type:
                label += f" | chunk_type: {chunk_type}"
            if node_type:
                label += f" | node_type: {node_type}"
            if matched_tags:
                label += f" | matched_tags: {', '.join(matched_tags)}"

            part = f'{label}: \n"""\n{text}\n"""'

            if self.max_context_chars > 0 and total_chars + len(part) > self.max_context_chars:
                remaining = self.max_context_chars - total_chars
                if remaining <= 0:
                    break
                part = part[:remaining].rstrip() + "\n...[truncated]"

            context_parts.append(part)
            total_chars += len(part)

            if self.max_context_chars > 0 and total_chars >= self.max_context_chars:
                break

        return "\n\n---\n\n".join(context_parts)

    def _extract_references(self, pages_list: List[int], company_name: str, pdf_sha1: Optional[str] = None) -> List[Dict]:
        company_sha1 = pdf_sha1 or ""
        if not company_sha1:
            if self.report_catalog is not None:
                report = self.report_catalog.get_report_by_company_name(company_name)
                if report is not None:
                    company_sha1 = report.sha1
            companies_df = self._load_companies_df()
            if not company_sha1 and "company_name" in companies_df.columns:
                matching_rows = companies_df[companies_df["company_name"] == company_name]
                if not matching_rows.empty:
                    for column in ("sha1", "sha1_name", "doc_id"):
                        if column in matching_rows.columns:
                            company_sha1 = matching_rows.iloc[0][column]
                            if company_sha1:
                                break
        return [{"pdf_sha1": company_sha1, "page_index": page} for page in pages_list]

    @staticmethod
    def _extract_references_from_results(pages_list: List[int], retrieval_results: List[Dict]) -> List[Dict]:
        pages = set(pages_list or [])
        references = []
        seen = set()
        for result in retrieval_results:
            page = result.get("page")
            if page not in pages:
                continue
            metadata = result.get("metadata") or {}
            pdf_sha1 = metadata.get("sha1_name") or metadata.get("doc_id")
            key = (pdf_sha1, page)
            if key in seen:
                continue
            seen.add(key)
            references.append(
                {
                    "pdf_sha1": pdf_sha1,
                    "page_index": page,
                }
            )
        return references

    def _build_query_plan(
        self,
        question: str,
        schema: str,
        company_name: Optional[str] = None,
        mentioned_companies: Optional[List[str]] = None,
        route_mode: Optional[str] = None,
    ) -> QueryPlan:
        query_plan = self.question_rewriter.rewrite(
            question,
            schema=schema,
            company_name=company_name,
            mentioned_companies=mentioned_companies,
        )
        if company_name:
            query_plan.filters.company_name = company_name
        if route_mode:
            query_plan.route_mode = route_mode
        return query_plan

    def _serialize_retrieval_result(self, result: Dict) -> Dict:
        metadata = result.get("metadata", {})
        return {
            "page": result.get("page"),
            "chunk_id": metadata.get("chunk_id"),
            "chunk_type": metadata.get("chunk_type"),
            "node_type": metadata.get("node_type"),
            "parent_chunk_id": metadata.get("parent_chunk_id"),
            "section_title": metadata.get("section_title"),
            "section_name": metadata.get("section_name"),
            "report_section": metadata.get("report_section"),
            "company_name": metadata.get("company_name"),
            "company_aliases": metadata.get("company_aliases", []),
            "security_code": metadata.get("security_code"),
            "stock_code": metadata.get("stock_code"),
            "broker_name": metadata.get("broker_name"),
            "currency": metadata.get("currency"),
            "exchange": metadata.get("exchange"),
            "board": metadata.get("board"),
            "market_type": metadata.get("market_type"),
            "industry_l1": metadata.get("industry_l1"),
            "industry_l2": metadata.get("industry_l2"),
            "report_year": metadata.get("report_year"),
            "report_type": metadata.get("report_type"),
            "doc_source_type": metadata.get("doc_source_type"),
            "report_date": metadata.get("report_date"),
            "fiscal_year": metadata.get("fiscal_year"),
            "period": metadata.get("period"),
            "unit_hint": metadata.get("unit_hint"),
            "language": metadata.get("language"),
            "topic_flags": metadata.get("topic_flags", []),
            "business_tags": metadata.get("business_tags", []),
            "strategy_tags": metadata.get("strategy_tags", []),
            "factor_tags": metadata.get("factor_tags", []),
            "chain_position_major": metadata.get("chain_position_major"),
            "chain_position_minor": metadata.get("chain_position_minor", []),
            "listing_tags": metadata.get("listing_tags", []),
            "ownership_tags": metadata.get("ownership_tags", []),
            "status_tags": metadata.get("status_tags", []),
            "style_tags": metadata.get("style_tags", []),
            "table_id": metadata.get("table_id"),
            "matched_child_chunk_ids": result.get("matched_child_chunk_ids", []),
            "matched_tags": result.get("matched_tags", []),
            "matched_queries": result.get("matched_queries", []),
            "query_hit_count": int(result.get("query_hit_count", len(result.get("matched_queries", [])))),
            "result_scope": result.get("result_scope"),
            "retrieval_sources": result.get("retrieval_sources", []),
            "score": round(_result_score(result), 4),
            "final_score": round(_result_score(result), 4),
            "distance_rrf": result.get("distance_rrf"),
            "colbert_score": result.get("colbert_score"),
            "final_relevance_score": result.get("final_relevance_score"),
            "text": result.get("text"),
            "text_preview": " ".join((result.get("text") or "").split())[:220],
        }

    def _aggregate_retrieval_results_by_report(self, retrieval_results: List[Dict]) -> List[Dict]:
        grouped: Dict[tuple, Dict] = {}
        for result in retrieval_results:
            metadata = result.get("metadata") or {}
            key = (
                metadata.get("sha1_name"),
                metadata.get("company_name"),
                metadata.get("stock_code") or metadata.get("security_code"),
                metadata.get("report_year"),
            )
            existing = grouped.get(key)
            serialized = self._serialize_retrieval_result(result)
            if existing is None:
                grouped[key] = {
                    "doc_id": metadata.get("sha1_name"),
                    "company_name": metadata.get("company_name"),
                    "stock_code": metadata.get("stock_code") or metadata.get("security_code"),
                    "report_year": metadata.get("report_year"),
                    "final_score": round(_result_score(result), 4),
                    "evidence_count": 1,
                    "matched_tags": list(result.get("matched_tags", [])),
                    "evidence_chunks": [serialized],
                }
                continue

            existing["final_score"] = max(existing["final_score"], round(_result_score(result), 4))
            existing["evidence_count"] += 1
            existing["matched_tags"] = sorted(
                set(existing.get("matched_tags", [])) | set(result.get("matched_tags", []))
            )
            if len(existing["evidence_chunks"]) < 3:
                existing["evidence_chunks"].append(serialized)

        aggregated = list(grouped.values())
        for item in aggregated:
            item["aggregation_score"] = round(float(item["final_score"]) + 0.02 * max(0, item["evidence_count"] - 1), 4)
        aggregated.sort(key=lambda item: item["aggregation_score"], reverse=True)
        return aggregated

    def _validate_page_references(self, claimed_pages: List[int], retrieval_results: List[Dict], min_pages: int = 2, max_pages: int = 8) -> List[int]:
        if claimed_pages is None:
            claimed_pages = []

        retrieved_pages = [result['page'] for result in retrieval_results]
        validated_pages = [page for page in claimed_pages if page in retrieved_pages]

        if len(validated_pages) < len(claimed_pages):
            removed_pages = set(claimed_pages) - set(validated_pages)
            print(f"Warning: Removed {len(removed_pages)} hallucinated page references: {removed_pages}")

        if len(validated_pages) < min_pages and retrieval_results:
            existing_pages = set(validated_pages)
            for result in retrieval_results:
                page = result['page']
                if page not in existing_pages:
                    validated_pages.append(page)
                    existing_pages.add(page)
                if len(validated_pages) >= min_pages:
                    break

        if len(validated_pages) > max_pages:
            print(f"Trimming references from {len(validated_pages)} to {max_pages} pages")
            validated_pages = validated_pages[:max_pages]

        return validated_pages

    def _build_retriever(self):
        cached_bundle = getattr(self._retriever_cache, "bundle", None) if self.retriever_cache_enabled else None
        if cached_bundle is not None:
            return cached_bundle

        if self.full_context:
            bundle = (
                VectorRetriever(
                    vector_db_dir=self.vector_db_dir,
                    documents_dir=self.documents_dir,
                    vector_search_k=self.vector_search_k,
                    ivf_nprobe=self.vector_ivf_nprobe,
                    hnsw_ef_search=self.vector_hnsw_ef_search,
                ),
                "full_context",
            )
        elif self.llm_reranking:
            bundle = (
                HybridRetriever(
                    documents_dir=self.documents_dir,
                    vector_db_dir=self.vector_db_dir,
                    bm25_db_dir=self.bm25_db_path,
                    sparse_db_dir=self.sparse_db_dir,
                    tag_db_dir=self.tag_db_dir,
                    use_vector_dbs=self.use_vector_dbs,
                    use_bm25_db=self.use_bm25_db,
                    use_sparse_lexical_db=self.use_sparse_lexical_db,
                    use_tag_db=self.use_tag_db,
                    vector_search_k=self.vector_search_k,
                    vector_ivf_nprobe=self.vector_ivf_nprobe,
                    vector_hnsw_ef_search=self.vector_hnsw_ef_search,
                    provider=self.api_provider,
                    model=self.answering_model,
                    reranking_strategy=self.reranking_strategy,
                    cascade_candidate_pool_cap=self.cascade_candidate_pool_cap,
                    colbert_top_n=self.colbert_top_n,
                    colbert_model=self.colbert_model,
                    colbert_device=self.colbert_device,
                    colbert_batch_size=self.colbert_batch_size,
                    colbert_query_max_length=self.colbert_query_max_length,
                    colbert_passage_max_length=self.colbert_passage_max_length,
                    final_reranking_backend=self.final_reranking_backend,
                    final_reranking_model=self.final_reranking_model,
                ),
                "hybrid_rerank",
            )
        elif sum(1 for enabled in (self.use_vector_dbs, self.use_bm25_db, self.use_sparse_lexical_db, self.use_tag_db) if enabled) > 1:
            bundle = (
                HybridRetriever(
                    documents_dir=self.documents_dir,
                    vector_db_dir=self.vector_db_dir,
                    bm25_db_dir=self.bm25_db_path,
                    sparse_db_dir=self.sparse_db_dir,
                    tag_db_dir=self.tag_db_dir,
                    use_vector_dbs=self.use_vector_dbs,
                    use_bm25_db=self.use_bm25_db,
                    use_sparse_lexical_db=self.use_sparse_lexical_db,
                    use_tag_db=self.use_tag_db,
                    vector_search_k=self.vector_search_k,
                    vector_ivf_nprobe=self.vector_ivf_nprobe,
                    vector_hnsw_ef_search=self.vector_hnsw_ef_search,
                    provider=self.api_provider,
                    model=self.answering_model,
                ),
                "hybrid",
            )
        elif self.use_vector_dbs:
            bundle = (
                VectorRetriever(
                    vector_db_dir=self.vector_db_dir,
                    documents_dir=self.documents_dir,
                    vector_search_k=self.vector_search_k,
                    ivf_nprobe=self.vector_ivf_nprobe,
                    hnsw_ef_search=self.vector_hnsw_ef_search,
                ),
                "vector",
            )
        elif self.use_sparse_lexical_db:
            if self.sparse_db_dir is None:
                raise ValueError("sparse_db_dir is required when sparse lexical retrieval is enabled.")
            bundle = (
                BGEM3SparseRetriever(sparse_db_dir=self.sparse_db_dir, documents_dir=self.documents_dir),
                "sparse",
            )
        elif self.use_tag_db:
            if self.tag_db_dir is None:
                raise ValueError("tag_db_dir is required when tag retrieval is enabled.")
            bundle = (
                TagRetriever(tag_db_dir=self.tag_db_dir, documents_dir=self.documents_dir),
                "tag",
            )
        else:
            if self.bm25_db_path is None:
                raise ValueError("bm25_db_path is required when BM25 retrieval is enabled.")
            bundle = (
                BM25Retriever(bm25_db_dir=self.bm25_db_path, documents_dir=self.documents_dir),
                "bm25",
            )

        if self.retriever_cache_enabled:
            self._retriever_cache.bundle = bundle
        return bundle

    def _run_retrieval(self, retriever, mode: str, company_name: str, query: str, filters, candidate_doc_ids: Optional[List[str]] = None) -> List[Dict]:
        if mode == "full_context":
            return retriever.retrieve_all(company_name, filters=filters, candidate_doc_ids=candidate_doc_ids)
        if mode == "hybrid_rerank":
            return retriever.retrieve_by_company_name(
                company_name=company_name,
                query=query,
                llm_reranking_sample_size=self.llm_reranking_sample_size,
                top_n=self.top_n_retrieval,
                parent_retrieval_mode=self.parent_retrieval_mode,
                filters=filters,
                candidate_doc_ids=candidate_doc_ids,
            )
        if mode == "hybrid":
            return retriever.retrieve_candidates_by_company_name(
                company_name=company_name,
                query=query,
                top_n=self.top_n_retrieval,
                parent_retrieval_mode=self.parent_retrieval_mode,
                filters=filters,
                candidate_doc_ids=candidate_doc_ids,
            )
        return retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=self.top_n_retrieval,
            parent_retrieval_mode=self.parent_retrieval_mode,
            filters=filters,
            candidate_doc_ids=candidate_doc_ids,
        )

    @staticmethod
    def _candidate_pool_sort_key(result: Dict) -> tuple:
        return (
            int(result.get("query_hit_count", len(result.get("matched_queries", [])))),
            _result_score(result),
            len(result.get("retrieval_sources", [])),
        )

    def merge_multi_query_candidates(
        self,
        retrieval_runs: List[Tuple[str, List[Dict]]],
        pool_cap: int,
    ) -> List[Dict]:
        merged: Dict[tuple, Dict] = {}
        for search_query, retrieval_results in retrieval_runs:
            for result in retrieval_results:
                key = _result_merge_key(result)
                existing = merged.get(key)
                if existing is None:
                    merged_result = result.copy()
                    merged_result["matched_queries"] = [search_query]
                    merged_result["query_hit_count"] = 1
                    merged[key] = merged_result
                    continue

                matched_queries = list(existing.get("matched_queries", []))
                if search_query not in matched_queries:
                    matched_queries.append(search_query)

                merged_child_ids = list(existing.get("matched_child_chunk_ids", []))
                for child_chunk_id in result.get("matched_child_chunk_ids", []):
                    if child_chunk_id not in merged_child_ids:
                        merged_child_ids.append(child_chunk_id)

                merged_sources = sorted(
                    set(existing.get("retrieval_sources", [])) | set(result.get("retrieval_sources", []))
                )
                merged_tags = sorted(set(existing.get("matched_tags", [])) | set(result.get("matched_tags", [])))

                best_result = result.copy() if _result_score(result) > _result_score(existing) else existing.copy()
                best_result["matched_queries"] = matched_queries
                best_result["query_hit_count"] = len(matched_queries)
                best_result["matched_child_chunk_ids"] = merged_child_ids
                best_result["retrieval_sources"] = merged_sources
                best_result["matched_tags"] = merged_tags
                merged[key] = best_result

        merged_results = list(merged.values())
        merged_results.sort(key=self._candidate_pool_sort_key, reverse=True)
        return merged_results[:max(1, pool_cap)]

    def _merge_multi_query_results(self, retrieval_runs: List[List[Dict]], top_n: int) -> List[Dict]:
        merged: Dict[tuple, Dict] = {}
        for retrieval_results in retrieval_runs:
            for result in retrieval_results:
                key = _result_merge_key(result)
                existing = merged.get(key)
                if existing is None:
                    merged[key] = result.copy()
                    continue

                existing_children = list(existing.get("matched_child_chunk_ids", []))
                for child_chunk_id in result.get("matched_child_chunk_ids", []):
                    if child_chunk_id not in existing_children:
                        existing_children.append(child_chunk_id)

                if _result_score(result) > _result_score(existing):
                    existing.update(result)

                sources = set(existing.get("retrieval_sources", []))
                sources.update(result.get("retrieval_sources", []))
                existing["retrieval_sources"] = sorted(sources)
                existing["matched_child_chunk_ids"] = existing_children
                existing["matched_tags"] = sorted(set(existing.get("matched_tags", [])) | set(result.get("matched_tags", [])))

        merged_results = list(merged.values())
        merged_results.sort(key=_result_score, reverse=True)
        return merged_results[:top_n]

    def _run_multi_query_global_rerank(
        self,
        retriever,
        company_name: str,
        question: str,
        search_queries: List[str],
        filters,
        candidate_doc_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Dict], int, List[Tuple[str, List[Dict]]], List[Dict]]:
        retrieval_runs = self._retrieve_multi_query_candidate_runs(
            retriever=retriever,
            company_name=company_name,
            search_queries=search_queries,
            filters=filters,
            candidate_doc_ids=candidate_doc_ids,
        )
        candidate_pool_cap = self._candidate_pool_cap(len(search_queries))
        reranked_results, merged_candidates = self._rerank_merged_candidate_pool(
            retriever=retriever,
            question=question,
            retrieval_runs=retrieval_runs,
            pool_cap=candidate_pool_cap,
        )
        return reranked_results, len(merged_candidates), retrieval_runs, merged_candidates

    def _retrieve_multi_query_candidate_runs(
        self,
        retriever,
        company_name: str,
        search_queries: List[str],
        filters,
        candidate_doc_ids: Optional[List[str]] = None,
        backend_scope: str = "all",
    ) -> List[Tuple[str, List[Dict]]]:
        retrieval_runs: List[Tuple[str, List[Dict]]] = []
        for search_query in search_queries:
            results = retriever.retrieve_candidates_by_company_name(
                company_name=company_name,
                query=search_query,
                top_n=self.llm_reranking_sample_size,
                parent_retrieval_mode=self.parent_retrieval_mode,
                filters=filters,
                candidate_doc_ids=candidate_doc_ids,
                backend_scope=backend_scope,
            )
            if results:
                retrieval_runs.append((search_query, results))
        return retrieval_runs

    def _rerank_merged_candidate_pool(
        self,
        retriever,
        question: str,
        retrieval_runs: List[Tuple[str, List[Dict]]],
        pool_cap: int,
    ) -> Tuple[List[Dict], List[Dict]]:
        merged_candidates = self.merge_multi_query_candidates(retrieval_runs, pool_cap=pool_cap)
        if not merged_candidates:
            return [], []

        reranked_results = retriever.rerank_candidates(
            query=question,
            candidate_results=merged_candidates,
            documents_batch_size=2,
            top_n=self.top_n_retrieval,
        )
        return reranked_results, merged_candidates

    def _candidate_pool_cap(self, num_queries: int) -> int:
        if self.reranking_strategy == "cascade":
            return self.cascade_candidate_pool_cap
        return max(1, self.llm_reranking_sample_size * max(1, num_queries))

    def _build_reranking_debug_payload(self, retriever, candidate_pool_size_before_rerank: Optional[int]) -> Dict:
        debug_payload = dict(getattr(retriever, "last_rerank_debug", {}) or {})
        default_backend = self.final_reranking_backend or os.getenv("RERANKING_BACKEND", "llm_prompt").lower()
        return {
            "reranking_strategy": debug_payload.get("reranking_strategy", self.reranking_strategy),
            "initial_candidate_pool_size": debug_payload.get("initial_candidate_pool_size", candidate_pool_size_before_rerank),
            "colbert_candidate_pool_size": debug_payload.get("colbert_candidate_pool_size"),
            "colbert_top_n": debug_payload.get("colbert_top_n", self.colbert_top_n if self.reranking_strategy == "cascade" else None),
            "final_reranking_backend": debug_payload.get("final_reranking_backend", default_backend),
        }

    @staticmethod
    def _top_result_score(retrieval_results: List[Dict]) -> Optional[float]:
        if not retrieval_results:
            return None
        return round(_result_score(retrieval_results[0]), 4)

    @staticmethod
    def _score_margin(retrieval_results: List[Dict]) -> Optional[float]:
        if len(retrieval_results) < 2:
            return None
        return round(_result_score(retrieval_results[0]) - _result_score(retrieval_results[1]), 4)

    def _should_attempt_hyde(
        self,
        *,
        mode: str,
        schema: str,
        search_queries: List[str],
    ) -> bool:
        return (
            self.hyde_enabled
            and self.hyde_trigger_mode == "fallback"
            and mode == "hybrid_rerank"
            and self.use_vector_dbs
            and (schema or "").lower() != "number"
            and len(search_queries) > 1
            and self.hyde_generator is not None
        )

    def _build_hyde_debug_payload(
        self,
        *,
        retrieval_results: List[Dict],
        candidate_pool_size_before_rerank: Optional[int],
    ) -> Dict:
        return {
            "enabled": bool(self.hyde_enabled),
            "triggered": False,
            "trigger_reasons": [],
            "generation_model": self.hyde_generation_model or self.answering_model,
            "generated_text": "",
            "backend_scope": "dense",
            "initial_top_score": self._top_result_score(retrieval_results),
            "initial_score_margin": self._score_margin(retrieval_results),
            "initial_candidate_pool_size": candidate_pool_size_before_rerank,
            "final_candidate_pool_size": candidate_pool_size_before_rerank,
        }

    def _confidence_from_individual_answers(self, individual_answers: Dict[str, Dict]) -> str:
        levels = [answer.get("confidence", "low") for answer in individual_answers.values()]
        if levels and all(level == "high" for level in levels):
            return "high"
        if any(level in {"high", "medium"} for level in levels):
            return "medium"
        return "low"

    @staticmethod
    def _should_use_multi_document_routing(schema: str, query_plan: QueryPlan) -> bool:
        if (schema or "").lower() not in {"names", "boolean"}:
            return False

        filters = query_plan.filters
        scalar_filters = (
            filters.exchange,
            filters.board,
            filters.market_type,
            filters.industry_l1,
            filters.industry_l2,
            filters.section_name,
            filters.chain_position_major,
        )
        list_filters = (
            filters.business_tags,
            filters.strategy_tags,
            filters.factor_tags,
            filters.chain_position_minor,
            filters.listing_tags,
            filters.ownership_tags,
            filters.status_tags,
            filters.style_tags,
        )
        return any(value for value in scalar_filters) or any(values for values in list_filters)

    def route_question(self, question: str, schema: str) -> Dict:
        extracted_companies = self._extract_companies_from_subset(question)

        if len(extracted_companies) > 1:
            query_plan = self._build_query_plan(
                question,
                schema=schema,
                mentioned_companies=extracted_companies,
                route_mode="comparative_explicit",
            )
            return {
                "companies": extracted_companies,
                "query_plan": query_plan,
                "route_info": {
                    "route_mode": "comparative_explicit",
                    "selected_company": None,
                    "candidate_companies": extracted_companies,
                    "selection_reasons": ["multiple_companies_mentioned_in_question"],
                },
                "is_comparative": True,
            }

        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            query_plan = self._build_query_plan(
                question,
                schema=schema,
                company_name=company_name,
                mentioned_companies=extracted_companies,
                route_mode="explicit_company",
            )
            if self.report_catalog is not None and self.doc_router_enabled:
                query_plan.filters.company_name = company_name
                _, route_info = self.report_catalog.resolve_single_company(query_plan, limit=self.candidate_doc_top_k)
                route_info["route_mode"] = "explicit_company"
                candidate_doc_ids = route_info.get("candidate_doc_ids") or []
                if candidate_doc_ids:
                    query_plan.filters.candidate_doc_ids = candidate_doc_ids
            else:
                route_info = {
                    "route_mode": "explicit_company",
                    "selected_company": company_name,
                    "candidate_companies": extracted_companies,
                    "selection_reasons": ["company_mentioned_in_question"],
                }
            return {
                "company_name": company_name,
                "companies": extracted_companies,
                "query_plan": query_plan,
                "route_info": route_info,
                "is_comparative": False,
            }

        if self.report_catalog is None:
            raise ValueError("No company name found in the question.")

        query_plan = self._build_query_plan(
            question,
            schema=schema,
            mentioned_companies=[],
            route_mode="document_catalog",
        )

        if self.doc_router_enabled and self._should_use_multi_document_routing(schema, query_plan):
            route_info = self.report_catalog.resolve_candidate_reports(query_plan)
            candidate_doc_ids = route_info.get("candidate_doc_ids") or []
            candidate_companies = route_info.get("candidate_companies") or []
            if candidate_doc_ids:
                query_plan.filters.company_name = None
                query_plan.filters.candidate_doc_ids = candidate_doc_ids
                query_plan.mentioned_companies = candidate_companies
                query_plan.route_mode = route_info.get("route_mode", "document_catalog_multi")
                return {
                    "company_name": candidate_companies[0] if candidate_companies else "",
                    "companies": candidate_companies,
                    "query_plan": query_plan,
                    "route_info": route_info,
                    "is_comparative": False,
                }

        company_name, route_info = self.report_catalog.resolve_single_company(query_plan, limit=self.candidate_doc_top_k)
        query_plan.filters.company_name = company_name
        candidate_doc_ids = route_info.get("candidate_doc_ids") or []
        if candidate_doc_ids:
            query_plan.filters.candidate_doc_ids = candidate_doc_ids
        query_plan.mentioned_companies = [company_name]
        query_plan.route_mode = route_info.get("route_mode", "document_catalog")
        return {
            "company_name": company_name,
            "companies": [company_name],
            "query_plan": query_plan,
            "route_info": route_info,
            "is_comparative": False,
        }

    def get_answer_for_company(
        self,
        company_name: str,
        question: str,
        schema: str,
        query_plan: Optional[QueryPlan] = None,
        route_info: Optional[Dict] = None,
    ) -> Dict:
        if not self.use_vector_dbs and not self.use_bm25_db and not self.use_sparse_lexical_db and not self.use_tag_db:
            raise ValueError("At least one retrieval backend must be enabled.")

        rewrite_result = query_plan or self._build_query_plan(
            question,
            schema=schema,
            company_name=company_name,
            mentioned_companies=[company_name],
            route_mode="explicit_company",
        )
        route_mode = (route_info or {}).get("route_mode") or rewrite_result.route_mode
        if route_mode == "document_catalog_multi":
            rewrite_result.filters.company_name = None
        else:
            rewrite_result.filters.company_name = company_name
        candidate_doc_ids = list((route_info or {}).get("candidate_doc_ids") or rewrite_result.filters.candidate_doc_ids or [])
        if candidate_doc_ids:
            rewrite_result.filters.candidate_doc_ids = candidate_doc_ids
        retriever, mode = self._build_retriever()
        candidate_pool_size_before_rerank = None
        reranking_debug = self._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
        hyde_debug = self._build_hyde_debug_payload(
            retrieval_results=[],
            candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
        )

        if mode == "full_context":
            retrieval_results = self._run_retrieval(retriever, mode, company_name, question, rewrite_result.filters, candidate_doc_ids)
            reranking_debug = self._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
            hyde_debug = self._build_hyde_debug_payload(
                retrieval_results=retrieval_results,
                candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
            )
        elif mode == "hybrid_rerank" and len(rewrite_result.search_queries) > 1:
            retrieval_results, candidate_pool_size_before_rerank, retrieval_runs, _ = self._run_multi_query_global_rerank(
                retriever=retriever,
                company_name=company_name,
                question=question,
                search_queries=rewrite_result.search_queries,
                filters=rewrite_result.filters,
                candidate_doc_ids=candidate_doc_ids,
            )
            reranking_debug = self._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
            hyde_debug = self._build_hyde_debug_payload(
                retrieval_results=retrieval_results,
                candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
            )
            if self._should_attempt_hyde(
                mode=mode,
                schema=schema,
                search_queries=rewrite_result.search_queries,
            ):
                hyde_triggered, hyde_reasons = should_trigger_hyde(
                    retrieval_results=retrieval_results,
                    top_score_threshold=self.hyde_top_score_threshold,
                    margin_threshold=self.hyde_margin_threshold,
                )
                hyde_debug["triggered"] = hyde_triggered
                hyde_debug["trigger_reasons"] = hyde_reasons
                if hyde_triggered:
                    try:
                        generated_hyde = self.hyde_generator.generate(
                            question=question,
                            schema=schema,
                            query_plan=rewrite_result,
                            route_info=route_info,
                            model=self.hyde_generation_model or self.answering_model,
                            provider=self.api_provider,
                        )
                        hyde_debug["generated_text"] = generated_hyde
                        if generated_hyde:
                            hyde_candidate_runs = self._retrieve_multi_query_candidate_runs(
                                retriever=retriever,
                                company_name=company_name,
                                search_queries=[generated_hyde],
                                filters=rewrite_result.filters,
                                candidate_doc_ids=candidate_doc_ids,
                                backend_scope="vector_only",
                            )
                            hyde_candidate_runs = [
                                (HYDE_QUERY_MARKER, results)
                                for _, results in hyde_candidate_runs
                            ]
                            if hyde_candidate_runs:
                                retrieval_results, merged_candidates = self._rerank_merged_candidate_pool(
                                    retriever=retriever,
                                    question=question,
                                    retrieval_runs=retrieval_runs + hyde_candidate_runs,
                                    pool_cap=self._candidate_pool_cap(len(rewrite_result.search_queries) + 1),
                                )
                                candidate_pool_size_before_rerank = len(merged_candidates)
                                reranking_debug = self._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
                                hyde_debug["final_candidate_pool_size"] = candidate_pool_size_before_rerank
                    except Exception as exc:
                        print(f"Warning: HyDE fallback failed: {exc}")
        else:
            retrieval_runs = []
            for search_query in rewrite_result.search_queries:
                results = self._run_retrieval(retriever, mode, company_name, search_query, rewrite_result.filters, candidate_doc_ids)
                if results:
                    retrieval_runs.append(results)
            retrieval_results = self._merge_multi_query_results(retrieval_runs, self.top_n_retrieval)
            reranking_debug = self._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
            hyde_debug = self._build_hyde_debug_payload(
                retrieval_results=retrieval_results,
                candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
            )

        if not retrieval_results:
            raise ValueError("No relevant context found")

        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.api_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model,
            temperature=self.answer_temperature,
        )
        self.response_data = dict(getattr(self.api_processor, "response_data", {}) or {})

        if schema == "number" and self.table_grounder is not None:
            grounding_result = self.table_grounder.ground_number_query(
                question=question,
                retrieval_results=retrieval_results,
                filters=rewrite_result.filters,
                candidate_doc_ids=candidate_doc_ids,
            )
            if grounding_result is not None and grounding_result.get("normalized_value") is not None:
                grounded_value = grounding_result["normalized_value"]
                if isinstance(grounded_value, float) and grounded_value.is_integer():
                    grounded_value = int(grounded_value)
                answer_dict["final_answer"] = grounded_value
                answer_dict["table_grounding_result"] = grounding_result
                answer_dict["reasoning_summary"] = (
                    f"答案已基于表格 grounding 校准，证据来自表 {grounding_result.get('table_id')} 的第 {grounding_result.get('page')} 页。"
                )
                pages = list(answer_dict.get("relevant_pages") or [])
                if grounding_result.get("page") is not None and grounding_result["page"] not in pages:
                    pages.append(grounding_result["page"])
                answer_dict["relevant_pages"] = pages

        pages = answer_dict.get("relevant_pages", [])
        validated_pages = self._validate_page_references(pages, retrieval_results)
        selected_report = (route_info or {}).get("selected_report") or {}
        answer_dict["relevant_pages"] = validated_pages
        if route_mode == "document_catalog_multi":
            answer_dict["references"] = self._extract_references_from_results(validated_pages, retrieval_results)
        else:
            answer_dict["references"] = self._extract_references(
                validated_pages,
                company_name,
                pdf_sha1=selected_report.get("sha1"),
            )
        answer_dict["citations"] = build_citations(
            retrieval_results,
            validated_pages,
            table_grounding_result=answer_dict.get("table_grounding_result"),
        )
        answer_dict["confidence"] = compute_confidence(answer_dict, retrieval_results)
        answer_dict["search_queries"] = rewrite_result.search_queries
        answer_dict["query_plan"] = rewrite_result.to_dict()
        answer_dict["route_info"] = route_info or {
            "route_mode": rewrite_result.route_mode,
            "selected_company": company_name,
            "candidate_companies": [company_name],
        }
        answer_dict["candidate_pool_size_before_rerank"] = candidate_pool_size_before_rerank
        answer_dict["reranking_strategy"] = reranking_debug["reranking_strategy"]
        answer_dict["initial_candidate_pool_size"] = reranking_debug["initial_candidate_pool_size"]
        answer_dict["colbert_candidate_pool_size"] = reranking_debug["colbert_candidate_pool_size"]
        answer_dict["colbert_top_n"] = reranking_debug["colbert_top_n"]
        answer_dict["final_reranking_backend"] = reranking_debug["final_reranking_backend"]
        answer_dict["hyde"] = hyde_debug
        answer_dict["retrieval_pages"] = [result.get("page") for result in retrieval_results]
        answer_dict["retrieval_results"] = [self._serialize_retrieval_result(result) for result in retrieval_results]
        answer_dict["retrieval_report_groups"] = self._aggregate_retrieval_results_by_report(retrieval_results)
        answer_dict["response_data"] = self.response_data
        if not self.reasoning_debug_enabled:
            answer_dict["step_by_step_analysis"] = ""
        validated_answer = validate_answer(answer_dict, retrieval_results, rewrite_result)
        return validated_answer.answer

    def _extract_companies_from_subset(self, question_text: str) -> List[str]:
        if self.report_catalog is not None:
            return self.report_catalog.extract_companies_from_question(question_text)

        companies_df = self._load_companies_df()
        company_names = sorted(companies_df['company_name'].unique(), key=len, reverse=True)
        found_companies = []
        for company in company_names:
            escaped_company = re.escape(company)
            pattern = rf'{escaped_company}(?:\W|$)'
            if re.search(pattern, question_text, re.IGNORECASE):
                found_companies.append(company)
                question_text = re.sub(pattern, '', question_text, flags=re.IGNORECASE)
        return found_companies

    def process_question(self, question: str, schema: str):
        route_decision = self.route_question(question, schema)
        if route_decision["is_comparative"]:
            return self.process_comparative_question(
                question,
                route_decision["companies"],
                schema,
            )

        return self.get_answer_for_company(
            company_name=route_decision["company_name"],
            question=question,
            schema=schema,
            query_plan=route_decision["query_plan"],
            route_info=route_decision["route_info"],
        )

    def _create_answer_detail_ref(self, answer_dict: Dict, question_index: int) -> str:
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict.get('step_by_step_analysis'),
                "reasoning_summary": answer_dict.get('reasoning_summary'),
                "relevant_pages": answer_dict.get('relevant_pages'),
                "citations": answer_dict.get("citations", []),
                "confidence": answer_dict.get("confidence", "low"),
                "confidence_reason": answer_dict.get("confidence_reason", ""),
                "validation_flags": answer_dict.get("validation_flags", []),
                "search_queries": answer_dict.get("search_queries", []),
                "query_plan": answer_dict.get("query_plan", {}),
                "route_info": answer_dict.get("route_info", {}),
                "retrieval_pages": answer_dict.get("retrieval_pages", []),
                "retrieval_results": answer_dict.get("retrieval_results", []),
                "retrieval_report_groups": answer_dict.get("retrieval_report_groups", []),
                "table_grounding_result": answer_dict.get("table_grounding_result"),
                "response_data": answer_dict.get("response_data", {}),
                "self": ref_id
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[Dict], print_stats: bool = False) -> Dict:
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(1 for q in processed_questions if q.get("value") == "N/A")
        success_count = total_questions - error_count - na_count
        if print_stats and total_questions:
            print(f"\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count/total_questions)*100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count/total_questions)*100:.1f}%)")
            print(f"Successfully answered: {success_count} ({(success_count/total_questions)*100:.1f}%)\n")

        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count
        }

    def process_questions_list(self, questions_list: List[Dict], output_path: str = None, pipeline_details: str = "") -> Dict:
        total_questions = len(questions_list)
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            for question_data in tqdm(questions_with_index, desc="Processing questions"):
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(processed_questions, output_path, pipeline_details=pipeline_details)
        else:
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i: i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        batch_results = list(executor.map(self._process_single_question, batch))
                    processed_questions.extend(batch_results)

                    if output_path:
                        self._save_progress(processed_questions, output_path, pipeline_details=pipeline_details)
                    pbar.update(len(batch_results))

        statistics = self._calculate_statistics(processed_questions, print_stats=True)
        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics
        }

    def _process_single_question(self, question_data: Dict) -> Dict:
        question_index = question_data.get("_question_index", 0)
        question_text = question_data.get("text")
        schema = question_data.get("kind")
        try:
            answer_dict = self.process_question(question_text, schema)

            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
                return {
                    "question_text": question_text,
                    "kind": schema,
                    "value": None,
                    "references": [],
                    "citations": [],
                    "confidence": "low",
                    "confidence_reason": answer_dict.get("confidence_reason", ""),
                    "validation_flags": answer_dict.get("validation_flags", []),
                    "route_info": answer_dict.get("route_info", {}),
                    "error": answer_dict["error"],
                    "answer_details": {"$ref": detail_ref}
                }

            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            return {
                "question_text": question_text,
                "kind": schema,
                "value": answer_dict.get("final_answer"),
                "references": answer_dict.get("references", []),
                "citations": answer_dict.get("citations", []),
                "confidence": answer_dict.get("confidence", "low"),
                "confidence_reason": answer_dict.get("confidence_reason", ""),
                "validation_flags": answer_dict.get("validation_flags", []),
                "route_info": answer_dict.get("route_info", {}),
                "answer_details": {"$ref": detail_ref}
            }
        except Exception as err:
            return self._handle_processing_error(question_text, schema, err, question_index)

    def _handle_processing_error(self, question_text: str, schema: str, err: Exception, question_index: int) -> Dict:
        import traceback
        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {
            "error_traceback": tb,
            "self": error_ref
        }

        with self._lock:
            self.answer_details[question_index] = error_detail

        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")

        return {
            "question_text": question_text,
            "kind": schema,
            "value": None,
            "references": [],
            "citations": [],
            "confidence": "low",
            "confidence_reason": f"处理失败：{type(err).__name__}: {error_message}",
            "validation_flags": ["processing_error"],
            "route_info": {},
            "error": f"处理失败：{type(err).__name__}: {error_message}",
            "answer_details": {"$ref": error_ref}
        }

    def _post_process_submission_answers(self, processed_questions: List[Dict]) -> List[Dict]:
        submission_answers = []

        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = "N/A" if "error" in q else q.get("value")
            references = q.get("references", [])
            citations = q.get("citations", [])
            confidence = q.get("confidence", "low")
            confidence_reason = q.get("confidence_reason", "")
            validation_flags = q.get("validation_flags", [])
            route_info = q.get("route_info", {})

            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            step_by_step_analysis = None
            table_grounding_result = None
            if answer_details_ref and answer_details_ref.startswith("#/answer_details/"):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                        step_by_step_analysis = self.answer_details[index].get("step_by_step_analysis")
                        table_grounding_result = self.answer_details[index].get("table_grounding_result")
                except (ValueError, IndexError):
                    pass

            if value == "N/A":
                references = []
                citations = []
            else:
                references = [
                    {
                        "pdf_sha1": ref["pdf_sha1"],
                        "page_index": ref["page_index"] - 1
                    }
                    for ref in references
                ]

            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
                "citations": citations,
                "confidence": confidence,
                "confidence_reason": confidence_reason,
                "validation_flags": validation_flags,
                "route_info": route_info,
            }

            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis
            if table_grounding_result:
                submission_answer["table_grounding_result"] = table_grounding_result

            submission_answers.append(submission_answer)

        return submission_answers

    def _save_progress(self, processed_questions: List[Dict], output_path: Optional[str], pipeline_details: str = ""):
        if not output_path:
            return

        statistics = self._calculate_statistics(processed_questions)
        result = {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics
        }

        output_file = Path(output_path)
        debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)
        with open(debug_file, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        answers = self._post_process_submission_answers(processed_questions)
        result_output = {
            "answers": answers,
            "details": pipeline_details
        }
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(result_output, file, ensure_ascii=False, indent=2)

    def process_all_questions(self, output_path: str = 'questions_with_answers.json', pipeline_details: str = "") -> Dict:
        return self.process_questions_list(
            self.questions,
            output_path,
            pipeline_details=pipeline_details
        )

    def process_comparative_question(self, question: str, companies: List[str], schema: str) -> Dict:
        rephrased_questions = self.api_processor.get_rephrased_questions(
            original_question=question,
            companies=companies,
            model=self.answering_model,
        )

        individual_answers: Dict[str, Dict] = {}
        aggregated_references = []
        aggregated_citations = []

        def process_company_question(company: str) -> tuple[str, Dict]:
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(f"Could not generate sub-question for company: {company}")
            answer_dict = self.get_answer_for_company(
                company_name=company,
                question=sub_question,
                schema=schema
            )
            return company, answer_dict

        max_workers = min(max(1, self.parallel_requests), len(companies))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_company = {
                executor.submit(process_company_question, company): company
                for company in companies
            }

            for future in concurrent.futures.as_completed(future_to_company):
                company = future_to_company[future]
                try:
                    _, answer_dict = future.result()
                except Exception as exc:
                    print(f"Error processing company {company}: {str(exc)}")
                    raise

                individual_answers[company] = answer_dict
                aggregated_references.extend(answer_dict.get("references", []))
                aggregated_citations.extend(answer_dict.get("citations", []))

        comparative_answer = self.api_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model,
            temperature=self.answer_temperature,
        )
        self.response_data = dict(self.api_processor.response_data)
        comparative_answer["references"] = dedupe_references(aggregated_references)
        comparative_answer["citations"] = dedupe_citations(aggregated_citations)
        comparative_answer["confidence"] = self._confidence_from_individual_answers(individual_answers)
        comparative_answer["confidence_reason"] = "当前置信度由比较问答模式下的各公司答案聚合而来。"
        comparative_answer["validation_flags"] = []
        comparative_answer["search_queries"] = [question]
        comparative_answer["query_plan"] = self._build_query_plan(
            question,
            schema="comparative",
            mentioned_companies=companies,
            route_mode="comparative_explicit",
        ).to_dict()
        comparative_answer["route_info"] = {
            "route_mode": "comparative_explicit",
            "selected_company": None,
            "candidate_companies": companies,
            "selection_reasons": ["multiple_companies_mentioned_in_question"],
        }
        comparative_answer["retrieval_pages"] = sorted(
            {citation.get("page") for citation in aggregated_citations if citation.get("page") is not None}
        )
        comparative_answer["retrieval_results"] = []
        comparative_answer["response_data"] = self.response_data
        return comparative_answer
