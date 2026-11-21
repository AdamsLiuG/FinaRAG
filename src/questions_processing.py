import concurrent.futures
from collections import defaultdict
from dataclasses import replace
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from src.analysis_playbooks import build_playbook_hint, get_default_playbook_store
from src.adaptive_retrieval import (
    adaptive_max_rounds,
    adaptive_retrieval_enabled,
    merge_supplement_results,
)
from src.answer_validation import validate_answer
from src.api_requests import APIProcessor
from src.chart_grounding import ChartGrounder
from src.citation_formatter import build_citations, compute_confidence, dedupe_citations, dedupe_references
from src.document_manifest import load_document_manifest
from src.hyde import HYDE_QUERY_MARKER, HyDEGenerator, should_trigger_hyde
from src.query_plan import QueryPlan
from src.query_rewrite import QuestionRewriter
from src.report_catalog import ReportCatalog
from src.retrieval import BM25Retriever, BGEM3SparseRetriever, HybridRetriever, TagRetriever, VectorRetriever
from src.retrieval_filters import build_result_metadata
from src.table_grounding import TableGrounder
from src.text_normalization import normalize_text, parse_numeric_value
from src.graph.builder import (
    build_comparative_graph,
    build_single_company_graph,
    run_finarag_graph,
)
from src.graph.legacy_runner import (
    run_legacy_comparative,
    run_legacy_question,
    run_legacy_single_company,
)
from src.graph.plugins import cash_dividend as _cash_dividend_plugin
from src.graph.plugins import legal_rep as _legal_rep_plugin
from src.graph.plugins import metadata_agg as _metadata_agg_plugin
from src.structured.query_router import StructuredQueryRouter
from src.structured.sql_store import FinancialSQLStore


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


def _legal_rep_event():
    """法代识别知识自 EVT-001 事件原子获取（委托至 graph.plugins.legal_rep）。"""
    return _legal_rep_plugin.legal_rep_event()


def _dividend_mention_terms() -> tuple:
    """分红事实表述词表——唯一真源是 PLB-001 剧本（委托至 graph.plugins.cash_dividend）。"""
    return _cash_dividend_plugin.dividend_mention_terms()

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
        retrieval_debug_top_n: Optional[int] = None,
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
        final_reranking_batch_size: int = 2,
        chart_confidence_threshold: float = 0.70,
        structured_db_path: Optional[Union[str, Path]] = None,
        graph_engine: Optional[str] = None,
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
        self.top_n_retrieval = max(1, int(top_n_retrieval))
        debug_top_n = self.top_n_retrieval if retrieval_debug_top_n is None else int(retrieval_debug_top_n)
        self.retrieval_debug_top_n = max(self.top_n_retrieval, max(1, debug_top_n))
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
        self.final_reranking_batch_size = max(1, int(final_reranking_batch_size))
        self.colbert_model = colbert_model
        self.colbert_device = colbert_device
        self.colbert_batch_size = max(1, int(colbert_batch_size))
        self.colbert_query_max_length = max(8, int(colbert_query_max_length))
        self.colbert_passage_max_length = max(16, int(colbert_passage_max_length))
        self.final_reranking_backend = (final_reranking_backend or "").strip().lower() or None
        self.final_reranking_model = final_reranking_model
        self.chart_confidence_threshold = float(chart_confidence_threshold)
        self.max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "8000"))
        self.max_doc_chars = int(os.getenv("RAG_MAX_DOC_CHARS", "2500"))

        self.answer_details: List[Optional[Dict]] = []
        self.response_data = {}
        self._lock = threading.Lock()
        self._retriever_cache = threading.local()
        self._document_cache: Dict[str, Dict[str, Any]] = {}
        # D1 并发修复：读路径原先在锁外（GIL 下无害但不规范），统一锁内；
        # 服务化多线程共享 processor 实例时的字典一致性保障
        self._document_cache_lock = threading.Lock()
        self._revenue_grounding_cache: Dict[Tuple[str, Optional[int], str], Optional[Dict[str, Any]]] = {}
        self._revenue_grounding_cache_lock = threading.Lock()
        self.question_rewriter = QuestionRewriter()
        self.companies_df: Optional[pd.DataFrame] = None
        self.report_catalog = ReportCatalog(self.subset_path, self.documents_dir) if self.subset_path else None
        self.table_grounder = TableGrounder(self.documents_dir) if self.numeric_grounding_enabled else None
        self.chart_grounder = (
            ChartGrounder(self.documents_dir, confidence_threshold=self.chart_confidence_threshold)
            if self.numeric_grounding_enabled
            else None
        )
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
        resolved_engine = (graph_engine or os.getenv("FINARAG_GRAPH_ENGINE", "langgraph")).strip().lower() or "langgraph"
        if resolved_engine not in {"langgraph", "legacy"}:
            raise ValueError("graph_engine must be either 'langgraph' or 'legacy'.")
        self.graph_engine = resolved_engine

        self.structured_router: Optional[StructuredQueryRouter] = None
        self._structured_sql_explicit = False
        if structured_db_path is not None:
            db_path = Path(structured_db_path)
            if db_path.exists():
                self.structured_router = StructuredQueryRouter(FinancialSQLStore(db_path))
                self._structured_sql_explicit = True
        elif os.getenv("FINARAG_STRUCTURED_SQL_ENABLED", "0").strip().lower() in {"1", "true", "yes"}:
            env_db = os.getenv("FINARAG_STRUCTURED_DB_PATH")
            candidate_db = Path(env_db) if env_db else (self.documents_dir.parent / "finarag_facts.sqlite")
            if candidate_db.exists():
                self.structured_router = StructuredQueryRouter(FinancialSQLStore(candidate_db))

    def _resolved_graph_engine(self) -> str:
        engine = (getattr(self, "graph_engine", None) or os.getenv("FINARAG_GRAPH_ENGINE", "langgraph")).strip().lower()
        return engine if engine in {"langgraph", "legacy"} else "langgraph"

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            questions = payload.get("questions")
            if isinstance(questions, list):
                return questions
            raise ValueError("Questions file dict payload must contain a list-valued 'questions' field.")
        if isinstance(payload, list):
            return payload
        raise ValueError("Questions file must contain either a list of questions or a dict with a 'questions' list.")

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
            doc_id = metadata.get("sha1_name") or metadata.get("doc_id")
            company = metadata.get("company_name")
            report_year = metadata.get("report_year") or metadata.get("fiscal_year")

            if self.max_doc_chars > 0 and len(text) > self.max_doc_chars:
                text = text[: self.max_doc_chars].rstrip() + "\n...[truncated]"

            label = f"Text retrieved from page {page_number}"
            if company:
                label += f" | company: {company}"
            if doc_id:
                label += f" | doc_id: {doc_id}"
            if report_year:
                label += f" | report_year: {report_year}"
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
        return [{"pdf_sha1": company_sha1, "page": page} for page in pages_list]

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
                    "page": page,
                }
            )
        return references

    def _load_document_payload(self, doc_id: Optional[str]) -> Optional[Dict[str, Any]]:
        doc_id = str(doc_id or "").strip()
        if not doc_id:
            return None
        # 整个 load 持锁（double-checked）：分离的读锁/写锁存在 check-then-act
        # 竞态——并发 cold miss 时多线程各自加载产生不同 payload 对象
        # （并发一致性测试抓到）。锁内 IO 串行化冷加载可接受（一次性成本），
        # 热路径缓存命中快速返回。
        with self._document_cache_lock:
            cached = self._document_cache.get(doc_id)
            if cached is not None:
                return cached

            candidate_paths = []
            if doc_id.endswith(".json"):
                candidate_paths.append(self.documents_dir / doc_id)
            else:
                candidate_paths.append(self.documents_dir / f"{doc_id}.json")
                candidate_paths.append(self.documents_dir / doc_id)

            for path in candidate_paths:
                if not path.exists() or not path.is_file():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                self._document_cache[doc_id] = payload
                return payload

            for path in self.documents_dir.glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                metainfo = payload.get("metainfo") or {}
                observed_doc_id = metainfo.get("sha1_name") or metainfo.get("doc_id") or path.stem
                if str(observed_doc_id) == doc_id:
                    self._document_cache[doc_id] = payload
                    return payload
            return None

    def _selected_doc_id(
        self,
        company_name: str,
        route_info: Optional[Dict[str, Any]],
        query_plan: Optional[QueryPlan] = None,
    ) -> Optional[str]:
        selected_report = (route_info or {}).get("selected_report") or {}
        if isinstance(selected_report, dict):
            for key in ("sha1", "doc_id"):
                if selected_report.get(key):
                    return str(selected_report[key])

        candidate_doc_ids = list((route_info or {}).get("candidate_doc_ids") or [])
        if not candidate_doc_ids and query_plan is not None:
            candidate_doc_ids = list(query_plan.filters.candidate_doc_ids or [])
        if len(candidate_doc_ids) == 1:
            return str(candidate_doc_ids[0])

        if self.report_catalog is not None and company_name:
            report = self.report_catalog.get_report_by_company_name(company_name)
            if report is not None:
                return report.sha1
        return None

    @staticmethod
    def _document_pages(document: Dict[str, Any]) -> List[Tuple[int, str]]:
        content = document.get("content") or {}
        pages: List[Tuple[int, str]] = []
        for page in content.get("pages") or []:
            try:
                page_number = int(page.get("page"))
            except (TypeError, ValueError):
                continue
            text = str(page.get("text") or "")
            if text:
                pages.append((page_number, text))
        if pages:
            return pages

        grouped: Dict[int, List[str]] = defaultdict(list)
        for chunk in content.get("chunks") or []:
            try:
                page_number = int(chunk.get("page") or chunk.get("page_start"))
            except (TypeError, ValueError):
                continue
            text = str(chunk.get("text") or chunk.get("search_text") or "")
            if text:
                grouped[page_number].append(text)
        return [(page, "\n".join(texts)) for page, texts in sorted(grouped.items())]

    def _document_page_result(
        self,
        *,
        document: Dict[str, Any],
        page: int,
        text: str,
        chunk_type: str,
        source: str,
        score: float = 1.0,
    ) -> Dict[str, Any]:
        metainfo = dict(document.get("metainfo") or {})
        doc_id = str(metainfo.get("sha1_name") or metainfo.get("doc_id") or "")
        if doc_id:
            metainfo["sha1_name"] = doc_id
        chunk = {
            "chunk_id": f"{doc_id or 'document'}:{chunk_type}:{page}",
            "chunk_type": chunk_type,
            "type": chunk_type,
            "node_type": "page",
            "page": page,
            "page_start": page,
            "page_end": page,
            "text": text,
            "search_text": text,
            "evidence_type": chunk_type,
        }
        metadata = build_result_metadata(metainfo, chunk)
        if doc_id and not metadata.get("sha1_name"):
            metadata["sha1_name"] = doc_id
        return {
            "distance": score,
            "combined_score": score,
            "ranking_score": score,
            "page": page,
            "text": text,
            "metadata": metadata,
            "chunk_id": chunk["chunk_id"],
            "chunk_type": chunk_type,
            "section_title": metadata.get("section_title"),
            "retrieval_sources": [source],
            "matched_child_chunk_ids": [],
            "matched_tags": [],
            "result_scope": "page",
        }

    def _build_rule_answer(
        self,
        *,
        final_answer: Any,
        question: str,
        schema: str,
        query_plan: QueryPlan,
        route_info: Optional[Dict[str, Any]],
        retrieval_results: List[Dict[str, Any]],
        relevant_pages: List[int],
        rule_name: str,
        reasoning: str,
        extra_debug: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pages = [] if final_answer in (None, "N/A") else list(dict.fromkeys(relevant_pages or []))
        references = [] if final_answer in (None, "N/A") else self._extract_references_from_results(pages, retrieval_results)
        citations = [] if final_answer in (None, "N/A") else build_citations(retrieval_results, pages)
        rule_route_info = dict(route_info or {})
        rule_route_info["route_mode"] = rule_name
        if extra_debug:
            rule_route_info[rule_name] = extra_debug

        answer = {
            "final_answer": final_answer if final_answer is not None else "N/A",
            "relevant_pages": pages,
            "references": references,
            "citations": citations,
            "confidence": "high" if final_answer not in (None, "N/A") else "low",
            "confidence_reason": f"{rule_name} 规则命中并生成了可追溯证据。" if final_answer not in (None, "N/A") else f"{rule_name} 规则未找到完整证据。",
            "reasoning_summary": reasoning,
            "step_by_step_analysis": reasoning if self.reasoning_debug_enabled else "",
            "search_queries": query_plan.search_queries if query_plan else [question],
            "query_plan": query_plan.to_dict() if query_plan else {},
            "route_info": rule_route_info,
            "candidate_pool_size_before_rerank": None,
            "reranking_strategy": self.reranking_strategy,
            "initial_candidate_pool_size": None,
            "colbert_candidate_pool_size": None,
            "colbert_top_n": self.colbert_top_n if self.reranking_strategy == "cascade" else None,
            "final_reranking_backend": self.final_reranking_backend or os.getenv("RERANKING_BACKEND", "llm_prompt").lower(),
            "hyde": self._build_hyde_debug_payload(retrieval_results=retrieval_results, candidate_pool_size_before_rerank=None),
            "retrieval_pages": [result.get("page") for result in retrieval_results],
            "retrieval_results": [self._serialize_retrieval_result(result) for result in retrieval_results],
            "retrieval_report_groups": self._aggregate_retrieval_results_by_report(retrieval_results),
            "response_data": {},
            "validation_flags": [],
        }
        return answer

    @staticmethod
    def _clean_person_name(raw_value: str) -> Optional[str]:
        return _legal_rep_plugin.clean_person_name(raw_value)

    @staticmethod
    def _legal_page_priority(page: int, text: str) -> float:
        return _legal_rep_plugin.legal_page_priority(page, text)

    def _find_legal_representative(self, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return _legal_rep_plugin.find_legal_representative(self, document)

    def _answer_legal_representative_question(
        self,
        *,
        question: str,
        schema: str,
        company_name: str,
        query_plan: QueryPlan,
        route_info: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return _legal_rep_plugin.answer_legal_representative_question(
            self,
            question=question,
            schema=schema,
            company_name=company_name,
            query_plan=query_plan,
            route_info=route_info,
        )

    @staticmethod
    def _is_cash_dividend_mention_question(question: str, schema: str) -> bool:
        return _cash_dividend_plugin.is_cash_dividend_mention_question(question, schema)

    def _find_cash_dividend_mention(self, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return _cash_dividend_plugin.find_cash_dividend_mention(self, document)

    def _answer_cash_dividend_mention_question(
        self,
        *,
        question: str,
        schema: str,
        company_name: str,
        query_plan: QueryPlan,
        route_info: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return _cash_dividend_plugin.answer_cash_dividend_mention_question(
            self,
            question=question,
            schema=schema,
            company_name=company_name,
            query_plan=query_plan,
            route_info=route_info,
        )

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
            "sha1_name": metadata.get("sha1_name"),
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
            "chart_id": metadata.get("chart_id"),
            "picture_id": metadata.get("picture_id"),
            "chart_type": metadata.get("chart_type"),
            "series_name": metadata.get("series_name"),
            "x_label": metadata.get("x_label"),
            "chart_confidence": metadata.get("chart_confidence"),
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

    def _validate_page_references(self, claimed_pages: List[int], retrieval_results: List[Dict], min_pages: int = 1, max_pages: int = 8) -> List[int]:
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
                    final_reranking_batch_size=self.final_reranking_batch_size,
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

    def _run_retrieval(
        self,
        retriever,
        mode: str,
        company_name: str,
        query: str,
        filters,
        candidate_doc_ids: Optional[List[str]] = None,
        task_type: Optional[str] = None,
    ) -> List[Dict]:
        if mode == "full_context":
            return retriever.retrieve_all(company_name, filters=filters, candidate_doc_ids=candidate_doc_ids)
        if not str(query or "").strip():
            return []
        top_n = self.retrieval_debug_top_n
        if mode == "hybrid_rerank":
            return retriever.retrieve_by_company_name(
                company_name=company_name,
                query=query,
                llm_reranking_sample_size=self.llm_reranking_sample_size,
                documents_batch_size=self.final_reranking_batch_size,
                top_n=top_n,
                parent_retrieval_mode=self.parent_retrieval_mode,
                filters=filters,
                candidate_doc_ids=candidate_doc_ids,
                task_type=task_type,
            )
        if mode == "hybrid":
            return retriever.retrieve_candidates_by_company_name(
                company_name=company_name,
                query=query,
                top_n=top_n,
                parent_retrieval_mode=self.parent_retrieval_mode,
                filters=filters,
                candidate_doc_ids=candidate_doc_ids,
            )
        return retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=top_n,
            parent_retrieval_mode=self.parent_retrieval_mode,
            filters=filters,
            candidate_doc_ids=candidate_doc_ids,
        )

    def _collect_supplement(
        self,
        retriever,
        company_name: str,
        rewrite_result,
        gap,
        candidate_doc_ids: Optional[List[str]] = None,
        parent_mode_override: Optional[str] = None,
    ) -> List[Dict]:
        """单轮补检收集：gap.supplement_queries 逐条走纯召回（hybrid 无 rerank）。

        parent_mode_override：非 None 时覆盖 self.parent_retrieval_mode
        （③ parent_recall 策略传 "page"；默认 None 走 processor 配置）。
        """
        supplement: List[Dict] = []
        for supplement_query in gap.supplement_queries:
            if parent_mode_override is not None:
                results = retriever.retrieve_candidates_by_company_name(
                    company_name=company_name,
                    query=supplement_query,
                    top_n=max(self.llm_reranking_sample_size, self.retrieval_debug_top_n),
                    parent_retrieval_mode=parent_mode_override,
                    filters=rewrite_result.filters,
                    candidate_doc_ids=candidate_doc_ids,
                )
            else:
                results = self._run_retrieval(
                    retriever, "hybrid", company_name, supplement_query,
                    rewrite_result.filters, candidate_doc_ids,
                    task_type=getattr(rewrite_result, "task_type", None),
                )
            supplement.extend(results)
        return supplement

    def _run_adaptive_supplement(
        self,
        retriever,
        company_name: str,
        question: str,
        rewrite_result,
        retrieval_results: List[Dict],
        candidate_doc_ids: Optional[List[str]] = None,
        state: Optional[Dict] = None,
    ) -> Optional[List[Dict]]:
        """预检多轮循环：detect_gap → 补检 → merge → rerank，直到收敛。

        收敛三条件（任一满足即停）：
        1. 缺口闭合：detect_gap 返回 None（剧本 required_metrics 全覆盖——
           停止条件知识化，不是跑 N 轮碰运气）
        2. 无新增证据：merge 后池长不变（边际收益为零）
        3. 预算耗尽：rounds 达 adaptive_max_rounds()
        附加防护：查询幂等停——本轮 gap.queries 与上轮完全相同 → break
        （同查询重复补检是浪费）。
        state 为预检与校验后闭环共享的预算池（局部 dict 传递，不进实例
        属性——D1 教训）。返回最终 retrieval_results；未触发返回 None。
        """
        from src.adaptive_retrieval import (
            adaptive_max_rounds,
            detect_gap,
            merge_supplement_results,
        )

        state = state if state is not None else {"rounds_used": 0, "traces": []}

        top_score: Optional[float] = None
        for item in retrieval_results:
            score = item.get("combined_score")
            if score is not None:
                numeric = float(score)
                top_score = numeric if top_score is None else max(top_score, numeric)

        gap = detect_gap(rewrite_result, retrieval_results, top_score=top_score)
        if gap is None:
            return None

        current_results = list(retrieval_results)
        prior_queries: List[str] = []
        prior_strategy: Optional[str] = None
        rounds_this_call = 0

        while gap is not None:
            if state["rounds_used"] >= adaptive_max_rounds():
                break                                   # 收敛条件 3：预算耗尽
            # 查询幂等停（同查询**同策略**才停——允许同查询换粒度重试）
            from src.adaptive_retrieval import _choose_strategy

            strategy = _choose_strategy(
                round_index=state["rounds_used"],
                prior_evidence_gain=state.get("last_evidence_gain", True),
            )
            if gap.supplement_queries == prior_queries and strategy == prior_strategy:
                break                                   # 幂等：同查询同策略
            prior_queries = list(gap.supplement_queries)
            prior_strategy = strategy

            # ③ 策略选择：上轮有新增维持 dense；无新增切父粒度（低置信第二手段）
            parent_override = "page" if strategy == "parent_recall" else None
            supplement = self._collect_supplement(
                retriever, company_name, rewrite_result, gap,
                candidate_doc_ids=candidate_doc_ids,
                parent_mode_override=parent_override,
            )

            merged = merge_supplement_results(current_results, supplement)
            if len(merged) == len(current_results):
                # 当前策略无新增：dense 未试过 parent → 切粒度重试一轮
                # （child 检索不到的页级证据可能存在于 parent 粒度）；
                # 已试过 parent 仍无新增 → 真收敛（条件 2）
                if strategy != "parent_recall" and not state.get("parent_tried"):
                    state["parent_tried"] = True
                    state["last_evidence_gain"] = False   # 驱动下轮选 parent
                    continue
                break
            current_results = merged

            current_results = retriever.rerank_candidates(
                query=question,
                candidate_results=current_results,
                documents_batch_size=self.final_reranking_batch_size,
                top_n=self.retrieval_debug_top_n,
                task_type=getattr(rewrite_result, "task_type", None),
            )
            state["rounds_used"] += 1
            rounds_this_call += 1
            state["last_evidence_gain"] = True   # 本轮有新增（无新增已 break）
            self._record_adaptive_trace(state, {
                "question": question,
                "company_name": company_name,
                "trigger": gap.trigger,
                "strategy": strategy,
                "queries": list(gap.supplement_queries),
                "new_hits": len(supplement),
                "rounds": state["rounds_used"],
                "atom_id": (getattr(rewrite_result, "metric_atom", None) or {}).get("atom_id"),
            })

            gap = detect_gap(rewrite_result, current_results)   # 重判：收敛条件 1

        return current_results if rounds_this_call else None

    @staticmethod
    def _record_adaptive_trace(state: Dict, entry: Optional[Dict] = None) -> None:
        """trace 入 state（entry 非 None）或统一落盘（entry 为 None）。

        一次请求一条聚合 trace：主流程收尾调 _record_adaptive_trace(state)
        落盘；写入失败不阻断主流程（可观测性不得反噬可用性）。
        """
        if entry is not None:
            state.setdefault("traces", []).append(entry)
            return
        for trace in state.get("traces", []):
            QuestionsProcessor._append_adaptive_trace(trace)

    @staticmethod
    def _append_adaptive_trace(entry: Dict) -> None:
        """补检 trace 追加 JSONL（v1 兼容保留；v2 走 _record_adaptive_trace）。"""
        try:
            from datetime import datetime

            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), **entry}
            with open(logs_dir / "adaptive_trace.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"Warning: adaptive trace 写入失败（不影响主流程）：{exc}")

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
        task_type: Optional[str] = None,
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
            task_type=task_type,
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
        # A2 层 3：多查询并行。线程预算：内层四路并行固定 4 workers，
        # 外层 = RETRIEVAL_MAX_THREADS // 4（默认 8 → 2 条查询并行），两层乘积
        # 不超过全局上限。
        cleaned_queries = [q for q in search_queries if str(q or "").strip()]
        if not cleaned_queries:
            return []
        retrieve_kwargs = dict(
            top_n=max(self.llm_reranking_sample_size, self.retrieval_debug_top_n),
            parent_retrieval_mode=self.parent_retrieval_mode,
            filters=filters,
            candidate_doc_ids=candidate_doc_ids,
            backend_scope=backend_scope,
        )

        def _run_single_query(search_query: str) -> List[Dict]:
            return retriever.retrieve_candidates_by_company_name(
                company_name=company_name, query=search_query, **retrieve_kwargs
            )

        parallel = getattr(retriever, "_parallel_backends_enabled", False)
        if parallel and len(cleaned_queries) > 1:
            from concurrent.futures import ThreadPoolExecutor

            max_threads = int(os.getenv("RETRIEVAL_MAX_THREADS", "8"))
            outer_workers = max(1, max_threads // 4)
            with ThreadPoolExecutor(max_workers=min(len(cleaned_queries), outer_workers)) as pool:
                mapped = list(pool.map(_run_single_query, cleaned_queries))
        else:
            mapped = [_run_single_query(q) for q in cleaned_queries]

        retrieval_runs: List[Tuple[str, List[Dict]]] = []
        for search_query, results in zip(cleaned_queries, mapped):
            if results:
                retrieval_runs.append((search_query, results))
        return retrieval_runs

    def _rerank_merged_candidate_pool(
        self,
        retriever,
        question: str,
        retrieval_runs: List[Tuple[str, List[Dict]]],
        pool_cap: int,
        task_type: Optional[str] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        merged_candidates = self.merge_multi_query_candidates(retrieval_runs, pool_cap=pool_cap)
        if not merged_candidates:
            return [], []

        reranked_results = retriever.rerank_candidates(
            query=question,
            candidate_results=merged_candidates,
            documents_batch_size=self.final_reranking_batch_size,
            top_n=self.retrieval_debug_top_n,
            task_type=task_type,
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

        if self.doc_router_enabled and (
            self._should_use_multi_document_routing(schema, query_plan)
            or self._is_metadata_count_question(question, schema)
        ):
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

    @staticmethod
    def _is_metadata_membership_question(question: str, schema: str) -> bool:
        return _metadata_agg_plugin.is_metadata_membership_question(question, schema)

    def _membership_report_from_route(self, company_name: str, route_info: Optional[Dict]) -> Optional[Any]:
        return _metadata_agg_plugin.membership_report_from_route(self, company_name, route_info)

    @staticmethod
    def _membership_reference_pages(evidence_rows: List[Dict[str, Any]]) -> List[int]:
        return _metadata_agg_plugin.membership_reference_pages(evidence_rows)

    @staticmethod
    def _metadata_scoring_page(evidence_rows: List[Dict[str, Any]]) -> int:
        return _metadata_agg_plugin.metadata_scoring_page(evidence_rows)

    def _metadata_membership_citations(
        self,
        report,
        query_plan: QueryPlan,
        evidence_rows: List[Dict[str, Any]],
        matched: bool,
    ) -> List[Dict[str, Any]]:
        return _metadata_agg_plugin.metadata_membership_citations(self, report, query_plan, evidence_rows, matched)

    @staticmethod
    def _metadata_label_terms(question: str, query_plan: QueryPlan) -> List[str]:
        return _metadata_agg_plugin.metadata_label_terms(question, query_plan)

    def _report_matches_strategy_labels(
        self,
        report,
        labels: List[str],
    ) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        return _metadata_agg_plugin.report_matches_strategy_labels(self, report, labels)

    @staticmethod
    def _is_metadata_count_question(question: str, schema: str) -> bool:
        return _metadata_agg_plugin.is_metadata_count_question(question, schema)

    def _metadata_collection_query_plan(self, query_plan: QueryPlan) -> QueryPlan:
        return _metadata_agg_plugin.metadata_collection_query_plan(query_plan)

    @staticmethod
    def _metadata_reference(report, page: int) -> Dict[str, Any]:
        return _metadata_agg_plugin.metadata_reference(report, page)

    def _metadata_retrieval_result_from_citation(self, citation: Dict[str, Any]) -> Dict[str, Any]:
        return _metadata_agg_plugin.metadata_retrieval_result_from_citation(citation)

    def _metadata_retrieval_results_from_citations(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _metadata_agg_plugin.metadata_retrieval_results_from_citations(self, citations)

    def _resolve_metadata_label_reports(
        self,
        *,
        question: str,
        query_plan: QueryPlan,
        route_info: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return _metadata_agg_plugin.resolve_metadata_label_reports(
            self,
            question=question,
            query_plan=query_plan,
            route_info=route_info,
        )

    def _metadata_answer_payload(
        self,
        *,
        final_answer: Any,
        query_plan: QueryPlan,
        route_info: Dict[str, Any],
        resolution: Dict[str, Any],
        confidence_reason: str,
        reasoning: str,
    ) -> Dict[str, Any]:
        return _metadata_agg_plugin.metadata_answer_payload(
            self,
            final_answer=final_answer,
            query_plan=query_plan,
            route_info=route_info,
            resolution=resolution,
            confidence_reason=confidence_reason,
            reasoning=reasoning,
        )

    @staticmethod
    def _is_metadata_names_question(question: str, schema: str) -> bool:
        return _metadata_agg_plugin.is_metadata_names_question(question, schema)

    def _answer_metadata_names_question(
        self,
        *,
        question: str,
        schema: str,
        route_decision: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        return _metadata_agg_plugin.answer_metadata_names_question(
            self,
            question=question,
            schema=schema,
            route_decision=route_decision,
        )

    @staticmethod
    def _build_metric_caliber_hint(query_plan: QueryPlan) -> Optional[str]:
        # FinaWiki 知识注入：指标口径（单指标查询）+ 分析判断依据（隐含意图问答）；均按需注入，未命中零成本
        hints: List[str] = []
        payload = getattr(query_plan, "metric_atom", None) or {}
        standard_name = payload.get("standard_name")
        if standard_name:
            parts = [f"【指标口径提示】{standard_name}：{payload.get('definition')}"]
            caliber_notes = payload.get("caliber_notes")
            if caliber_notes:
                parts.append(str(caliber_notes).rstrip("。") + "。")
            confusable_names = payload.get("confusable_names") or []
            if confusable_names:
                parts.append(
                    f"易混淆口径：{'、'.join(confusable_names)}。若上下文证据为上述其他口径而非 {standard_name}，"
                    "请在 answer 中标注实际口径或返回 N/A，禁止混用口径作答。"
                )
            hints.append("".join(parts))
        playbook_hint = build_playbook_hint(getattr(query_plan, "playbook", None) or {})
        if playbook_hint:
            hints.append(playbook_hint)
        return "\n".join(hints) if hints else None

    def _answer_metric_definition_question(
        self,
        *,
        question: str,
        schema: str,
        route_decision: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        # FinaWiki 分层路由：指标定义类问题由原子库直接回答（结构化优先、语义兜底），不走检索；
        # 未命中原子或 schema 不适配（非 text）时回落通用 RAG 路径（fail-closed）。
        query_plan: QueryPlan = route_decision["query_plan"]
        if query_plan.route != "metric_atom_store" or (schema or "").lower() != "text":
            return None
        payload = query_plan.metric_atom or {}
        standard_name = payload.get("standard_name")
        if not standard_name:
            return None
        answer_lines = [f"{standard_name}：{payload.get('definition')}。"]
        caliber_notes = payload.get("caliber_notes")
        if caliber_notes:
            answer_lines.append(f"口径说明：{str(caliber_notes).rstrip('。')}。")
        confusable_names = payload.get("confusable_names") or []
        if confusable_names:
            answer_lines.append(f"注意与以下易混淆口径区分：{'、'.join(confusable_names)}。")
        if payload.get("unit"):
            answer_lines.append(f"计量单位：{payload['unit']}。")
        answer_text = "\n".join(answer_lines)
        route_info = dict(route_decision.get("route_info") or {})
        route_info["route_mode"] = "metric_atom_store"
        route_info["metric_atom_id"] = payload.get("atom_id")
        reasoning = (
            f"定义类问题命中指标原子 {payload.get('atom_id')}（{standard_name}），"
            "由口径原子库直接回答，未触发检索。"
        )
        return {
            "final_answer": answer_text,
            "relevant_pages": [],
            "references": [],
            "citations": [],
            "confidence": "high",
            "confidence_reason": "答案来自指标口径原子库（人工校准的领域知识），非检索生成。",
            "reasoning_summary": reasoning,
            "step_by_step_analysis": reasoning if self.reasoning_debug_enabled else "",
            "search_queries": query_plan.search_queries,
            "query_plan": query_plan.to_dict(),
            "route_info": route_info,
            "candidate_pool_size_before_rerank": None,
            "reranking_strategy": self.reranking_strategy,
            "initial_candidate_pool_size": None,
            "colbert_candidate_pool_size": None,
            "colbert_top_n": self.colbert_top_n if self.reranking_strategy == "cascade" else None,
            "final_reranking_backend": self.final_reranking_backend or os.getenv("RERANKING_BACKEND", "llm_prompt").lower(),
            "hyde": self._build_hyde_debug_payload(retrieval_results=[], candidate_pool_size_before_rerank=None),
            "retrieval_pages": [],
            "retrieval_results": [],
            "retrieval_report_groups": {},
            "response_data": {},
            "validation_flags": [],
        }

    def _answer_metadata_count_question(
        self,
        *,
        question: str,
        schema: str,
        route_decision: Dict[str, Any],
        question_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        return _metadata_agg_plugin.answer_metadata_count_question(
            self,
            question=question,
            schema=schema,
            route_decision=route_decision,
            question_meta=question_meta,
        )

    @staticmethod
    def _clean_industry_filter(value: Optional[str]) -> Optional[str]:
        return _metadata_agg_plugin.clean_industry_filter(value)

    @staticmethod
    def _metadata_count_allows_expected_filters(question_meta: Dict[str, Any]) -> bool:
        return _metadata_agg_plugin.metadata_count_allows_expected_filters(question_meta)

    @staticmethod
    def _normalize_expected_report_filters(expected_filters: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        return _metadata_agg_plugin.normalize_expected_report_filters(expected_filters)

    def _expected_metadata_count_query_plan(
        self,
        query_plan: QueryPlan,
        question_meta: Dict[str, Any],
    ) -> Optional[QueryPlan]:
        return _metadata_agg_plugin.expected_metadata_count_query_plan(self, query_plan, question_meta)

    def _relaxed_metadata_count_query_plan(self, query_plan: QueryPlan) -> QueryPlan:
        return _metadata_agg_plugin.relaxed_metadata_count_query_plan(self, query_plan)

    def _resolve_metadata_count_reports(
        self,
        *,
        question: str,
        query_plan: QueryPlan,
        route_info: Optional[Dict[str, Any]],
        question_meta: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], str]:
        return _metadata_agg_plugin.resolve_metadata_count_reports(
            self,
            question=question,
            query_plan=query_plan,
            route_info=route_info,
            question_meta=question_meta,
        )

    def _answer_metadata_membership_question(
        self,
        *,
        question: str,
        schema: str,
        route_decision: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        return _metadata_agg_plugin.answer_metadata_membership_question(
            self,
            question=question,
            schema=schema,
            route_decision=route_decision,
        )

    @staticmethod
    def _grounded_number_value(grounding_result: Dict[str, Any]) -> Any:
        value = grounding_result.get("answer_value")
        if value is None:
            value = grounding_result.get("normalized_value")
        target_unit = grounding_result.get("target_unit")
        decimal_places = {
            "元": 2,
            "万元": 4,
            "亿元": 6,
            "百万元": 4,
        }.get(target_unit)
        if decimal_places is not None and isinstance(value, (int, float)):
            try:
                quant = Decimal("1") if decimal_places == 0 else Decimal("1").scaleb(-decimal_places)
                value = float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))
            except (InvalidOperation, ValueError):
                pass
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    def _table_grounding_retrieval_result(
        self,
        grounding_result: Dict[str, Any],
        *,
        chunk_type: str = "serialized_table",
        retrieval_source: str = "table_grounding",
    ) -> Dict[str, Any]:
        doc_id = str(grounding_result.get("source_doc_id") or "")
        document = self.table_grounder._load_document(doc_id) if self.table_grounder is not None and doc_id else None
        metainfo = (document or {}).get("metainfo") or {}
        metadata = {
            "sha1_name": doc_id,
            "company_name": metainfo.get("company_name"),
            "company_aliases": list(metainfo.get("company_aliases") or []),
            "security_code": metainfo.get("security_code"),
            "stock_code": metainfo.get("stock_code") or metainfo.get("security_code"),
            "currency": metainfo.get("currency"),
            "major_industry": metainfo.get("major_industry"),
            "report_year": metainfo.get("report_year") or metainfo.get("fiscal_year"),
            "report_type": metainfo.get("report_type"),
            "doc_source_type": metainfo.get("doc_source_type"),
            "period": grounding_result.get("period"),
            "unit_hint": grounding_result.get("unit"),
            "language": metainfo.get("language"),
            "chunk_id": None,
            "chunk_type": chunk_type,
            "section_title": None,
            "section_name": None,
            "report_section": None,
            "table_id": grounding_result.get("table_id"),
            "logical_table_id": grounding_result.get("logical_table_id"),
            "logical_role": None,
            "page_span": list(grounding_result.get("page_span") or []),
            "merge_confidence": None,
            "merge_state": "confirmed" if grounding_result.get("logical_table_id") else None,
            "row_idx": grounding_result.get("row_idx"),
            "col_idx": grounding_result.get("col_idx"),
            "matched_row_headers": grounding_result.get("matched_row_headers") or [],
            "matched_col_headers": grounding_result.get("matched_col_headers") or [],
            "parent_block_id": None,
            "parent_chunk_id": None,
            "child_chunk_ids": [],
            "node_type": "table",
            "evidence_type": "table",
            "has_table_context": True,
            "page_start": grounding_result.get("page"),
            "page_end": grounding_result.get("page"),
            "topic_flags": [],
        }
        return {
            "distance": 1.0,
            "combined_score": 1.0,
            "ranking_score": 1.0,
            "page": grounding_result.get("page"),
            "text": grounding_result.get("table_snippet") or "",
            "metadata": metadata,
            "chunk_id": None,
            "chunk_type": chunk_type,
            "section_title": None,
            "table_id": grounding_result.get("table_id"),
            "retrieval_sources": [retrieval_source],
            "matched_child_chunk_ids": [],
            "matched_tags": [],
            "result_scope": "table",
        }

    def _ensure_table_grounding_retrieval_result(
        self,
        retrieval_results: List[Dict],
        grounding_result: Optional[Dict[str, Any]],
    ) -> List[Dict]:
        if not grounding_result or grounding_result.get("page") is None:
            return retrieval_results
        doc_id = str(grounding_result.get("source_doc_id") or "")
        page = grounding_result.get("page")
        for result in retrieval_results:
            metadata = result.get("metadata") or {}
            if str(metadata.get("sha1_name") or "") == doc_id and result.get("page") == page:
                return retrieval_results
        return [self._table_grounding_retrieval_result(grounding_result)] + list(retrieval_results)

    def _ensure_table_support_retrieval_results(
        self,
        retrieval_results: List[Dict],
        grounding_result: Optional[Dict[str, Any]],
    ) -> List[Dict]:
        if not grounding_result:
            return retrieval_results
        results = list(retrieval_results)
        seen = {
            (str((result.get("metadata") or {}).get("sha1_name") or ""), result.get("page"))
            for result in results
        }
        for support_result in self._table_support_results(grounding_result):
            doc_id = str(support_result.get("source_doc_id") or "")
            page = support_result.get("page")
            if not doc_id or page is None or (doc_id, page) in seen:
                continue
            results.append(
                self._table_grounding_retrieval_result(
                    support_result,
                    chunk_type="table_support",
                    retrieval_source="table_support",
                )
            )
            seen.add((doc_id, page))
        return results

    @staticmethod
    def _table_support_results(grounding_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not grounding_result:
            return []
        return list(grounding_result.get("supporting_matches") or []) + list(
            grounding_result.get("logical_context_matches") or []
        )

    def _chart_grounding_retrieval_result(
        self,
        grounding_result: Dict[str, Any],
        *,
        chunk_type: str = "chart_to_table",
        retrieval_source: str = "chart_grounding",
    ) -> Dict[str, Any]:
        doc_id = str(grounding_result.get("source_doc_id") or "")
        document = self.chart_grounder._load_document(doc_id) if self.chart_grounder is not None and doc_id else None
        metainfo = (document or {}).get("metainfo") or {}
        metadata = {
            "sha1_name": doc_id,
            "company_name": metainfo.get("company_name"),
            "company_aliases": list(metainfo.get("company_aliases") or []),
            "security_code": metainfo.get("security_code"),
            "stock_code": metainfo.get("stock_code") or metainfo.get("security_code"),
            "currency": metainfo.get("currency"),
            "major_industry": metainfo.get("major_industry"),
            "report_year": metainfo.get("report_year") or metainfo.get("fiscal_year"),
            "report_type": metainfo.get("report_type"),
            "doc_source_type": metainfo.get("doc_source_type"),
            "period": grounding_result.get("period"),
            "unit_hint": grounding_result.get("unit"),
            "language": metainfo.get("language"),
            "chunk_id": None,
            "chunk_type": chunk_type,
            "section_title": None,
            "section_name": None,
            "report_section": None,
            "table_id": None,
            "chart_id": grounding_result.get("chart_id"),
            "picture_id": grounding_result.get("picture_id"),
            "series_name": grounding_result.get("series_name"),
            "x_label": grounding_result.get("x_label"),
            "chart_confidence": grounding_result.get("chart_confidence") or grounding_result.get("confidence"),
            "parent_block_id": None,
            "parent_chunk_id": None,
            "child_chunk_ids": [],
            "node_type": "chart",
            "evidence_type": "chart",
            "has_table_context": False,
            "has_chart_context": True,
            "page_start": grounding_result.get("page"),
            "page_end": grounding_result.get("page"),
            "topic_flags": [],
        }
        return {
            "distance": 1.0,
            "combined_score": 1.0,
            "ranking_score": 1.0,
            "page": grounding_result.get("page"),
            "text": grounding_result.get("chart_context") or "",
            "metadata": metadata,
            "chunk_id": None,
            "chunk_type": chunk_type,
            "section_title": None,
            "chart_id": grounding_result.get("chart_id"),
            "retrieval_sources": [retrieval_source],
            "matched_child_chunk_ids": [],
            "matched_tags": [],
            "result_scope": "chart",
        }

    def _ensure_chart_grounding_retrieval_result(
        self,
        retrieval_results: List[Dict],
        grounding_result: Optional[Dict[str, Any]],
    ) -> List[Dict]:
        if not grounding_result or grounding_result.get("page") is None:
            return retrieval_results
        doc_id = str(grounding_result.get("source_doc_id") or "")
        page = grounding_result.get("page")
        chart_id = grounding_result.get("chart_id")
        for result in retrieval_results:
            metadata = result.get("metadata") or {}
            if (
                str(metadata.get("sha1_name") or "") == doc_id
                and result.get("page") == page
                and metadata.get("chart_id") == chart_id
            ):
                return retrieval_results
        return [self._chart_grounding_retrieval_result(grounding_result)] + list(retrieval_results)

    def _build_table_grounded_number_answer(self, grounding_result: Dict[str, Any]) -> Dict[str, Any]:
        grounded_value = self._grounded_number_value(grounding_result)
        unit_note = f"，并按问题要求换算为{grounding_result['target_unit']}" if grounding_result.get("target_unit") else ""
        logical_note = ""
        if grounding_result.get("logical_table_id") and grounding_result.get("logical_table_materialized"):
            logical_note = f"，并结合跨页逻辑表 {grounding_result.get('logical_table_id')} 的继承表头/单位"
        page = grounding_result.get("page")
        pages = [page] if page is not None else []
        for logical_page in grounding_result.get("member_pages") or []:
            if logical_page is not None and logical_page not in pages:
                pages.append(logical_page)
        for support_result in grounding_result.get("supporting_matches") or []:
            support_page = support_result.get("page")
            if support_page is not None and support_page not in pages:
                pages.append(support_page)
            if len(pages) >= 3:
                break
        pages = sorted(pages)
        return {
            "final_answer": grounded_value,
            "relevant_pages": pages,
            "references": [],
            "citations": [],
            "confidence": "high",
            "reasoning_summary": (
                f"答案来自表格 grounding：表 {grounding_result.get('table_id')} 第 {page} 页的单元格"
                f"{logical_note}{unit_note}。"
            ),
            "step_by_step_analysis": (
                f"匹配行头：{grounding_result.get('matched_row_headers') or []}；"
                f"匹配列头：{grounding_result.get('matched_col_headers') or []}；"
                f"原始值：{grounding_result.get('raw_value')}；标准值：{grounding_result.get('normalized_value')}。"
            ),
            "table_grounding_result": grounding_result,
        }

    def _build_chart_grounded_number_answer(self, grounding_result: Dict[str, Any]) -> Dict[str, Any]:
        grounded_value = self._grounded_number_value(grounding_result)
        unit_note = f"，并按问题要求换算为{grounding_result['target_unit']}" if grounding_result.get("target_unit") else ""
        page = grounding_result.get("page")
        pages = [page] if page is not None else []
        return {
            "final_answer": grounded_value,
            "relevant_pages": pages,
            "references": [],
            "citations": [],
            "confidence": "medium",
            "reasoning_summary": (
                f"答案来自 chart-to-table grounding：图表 {grounding_result.get('chart_id')} "
                f"第 {page} 页的 {grounding_result.get('series_name')} / {grounding_result.get('x_label')}"
                f"{unit_note}。"
            ),
            "step_by_step_analysis": (
                f"图表ID：{grounding_result.get('chart_id')}；"
                f"图片ID：{grounding_result.get('picture_id')}；"
                f"指标：{grounding_result.get('series_name')}；"
                f"横轴：{grounding_result.get('x_label')}；"
                f"原始值：{grounding_result.get('raw_value')}；标准值：{grounding_result.get('normalized_value')}。"
            ),
            "chart_grounding_result": grounding_result,
        }

    @staticmethod
    def _is_revenue_metric_question(question: str) -> bool:
        normalized = normalize_text(question or "")
        return "营业收入" in normalized or "营收" in normalized

    @staticmethod
    def _companies_in_question_order(question: str, companies: List[str]) -> List[str]:
        normalized_question = normalize_text(question or "").replace(" ", "")
        indexed = []
        for index, company in enumerate(companies):
            normalized_company = normalize_text(company).replace(" ", "")
            position = normalized_question.find(normalized_company) if normalized_company else -1
            indexed.append((position if position >= 0 else 10_000 + index, index, company))
        indexed.sort()
        return [company for _, _, company in indexed]

    @staticmethod
    def _threshold_value_yuan(question: str) -> Optional[float]:
        match = re.search(
            r"(?:超过|高于|大于)\s*([0-9０-９][0-9０-９,，]*(?:\.\d+)?)\s*(百万元|千万元|亿元|万元|元)?",
            question or "",
        )
        if not match:
            return None
        raw_number = match.group(1)
        unit = match.group(2) or "元"
        return parse_numeric_value(f"{raw_number}{unit}")

    @staticmethod
    def _target_unit_from_question(question: str) -> Optional[str]:
        question = question or ""
        for unit in ("百万元", "千万元", "亿元", "万元"):
            if unit in question:
                return unit
        if "元" in question and not any(currency_unit in question for currency_unit in ("美元", "港元", "欧元", "日元")):
            return "元"
        return None

    @staticmethod
    def _convert_yuan_to_target(value: Optional[float], target_unit: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        factors = {
            "元": 1.0,
            "万元": 1e4,
            "亿元": 1e8,
            "百万元": 1e6,
            "千万元": 1e7,
        }
        factor = factors.get(target_unit or "")
        return float(value) / factor if factor else value

    def _apply_target_unit_to_grounding(self, grounding_result: Dict[str, Any], question: str) -> Dict[str, Any]:
        result = dict(grounding_result)
        target_unit = self._target_unit_from_question(question)
        normalized_value = result.get("normalized_value")
        result["target_unit"] = target_unit
        result["answer_value"] = self._convert_yuan_to_target(normalized_value, target_unit)
        if target_unit:
            result["unit_conversion"] = {
                "from": result.get("unit"),
                "to": target_unit,
                "base_value": normalized_value,
                "converted_value": result["answer_value"],
            }
        return result

    def _number_query_plan_for_company(self, company_name: str, question: str) -> Tuple[QueryPlan, Dict[str, Any]]:
        query_plan = self._build_query_plan(
            question,
            schema="number",
            company_name=company_name,
            mentioned_companies=[company_name],
            route_mode="explicit_company",
        )
        query_plan.filters.company_name = company_name
        query_plan.filters.question_kind = "number"

        if self.report_catalog is not None:
            reports = [
                report
                for report in self.report_catalog.get_reports()
                if report.company_name == company_name
            ]
            if query_plan.filters.year is not None:
                reports = [report for report in reports if report.report_year in (None, query_plan.filters.year)]
            if query_plan.filters.doc_source_type:
                reports = [
                    report
                    for report in reports
                    if not report.doc_source_type or report.doc_source_type == query_plan.filters.doc_source_type
                ]
            if reports:
                reports.sort(key=lambda report: (report.report_year or -1, report.sha1), reverse=True)
                report = reports[0]
                query_plan.filters.candidate_doc_ids = [report.sha1]
                return query_plan, {
                    "route_mode": "explicit_company",
                    "selected_company": report.company_name,
                    "candidate_companies": [report.company_name],
                    "candidate_doc_ids": [report.sha1],
                    "selected_report": report.to_dict(),
                    "selection_reasons": ["company_name_exact_for_numeric_rule"],
                }
            try:
                _, route_info = self.report_catalog.resolve_single_company(query_plan, limit=self.candidate_doc_top_k)
                route_info["route_mode"] = "explicit_company"
                candidate_doc_ids = route_info.get("candidate_doc_ids") or []
                if candidate_doc_ids:
                    query_plan.filters.candidate_doc_ids = candidate_doc_ids
                return query_plan, route_info
            except Exception:
                pass

        return query_plan, {
            "route_mode": "explicit_company",
            "selected_company": company_name,
            "candidate_companies": [company_name],
            "candidate_doc_ids": list(query_plan.filters.candidate_doc_ids or []),
            "selection_reasons": ["company_argument"],
        }

    def _cached_revenue_grounding(
        self,
        *,
        company_name: str,
        question: str,
        query_plan: QueryPlan,
        candidate_doc_ids: List[str],
        allowed_doc_ids: List[str],
        retrieval_results: Optional[List[Dict]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.table_grounder is None:
            return None
        doc_id = str((allowed_doc_ids or candidate_doc_ids or [""])[0] or "")
        if not doc_id:
            return None
        cache_key = (doc_id, query_plan.filters.year, "营业收入")
        with self._revenue_grounding_cache_lock:
            cached = self._revenue_grounding_cache.get(cache_key)
        if cached is None and cache_key not in self._revenue_grounding_cache:
            year_text = f"{query_plan.filters.year}年报" if query_plan.filters.year is not None else ""
            sub_question = f"{company_name}{year_text}营业收入是多少元？"
            computed = self.table_grounder.ground_number_query(
                question=sub_question,
                retrieval_results=retrieval_results or [],
                filters=query_plan.filters,
                candidate_doc_ids=candidate_doc_ids,
                allowed_doc_ids=allowed_doc_ids,
            )
            with self._revenue_grounding_cache_lock:
                self._revenue_grounding_cache.setdefault(cache_key, computed)
                cached = self._revenue_grounding_cache.get(cache_key)
        if cached is None:
            return None
        return self._apply_target_unit_to_grounding(cached, question)

    def _ground_revenue_for_company(self, company_name: str, question: str) -> Dict[str, Any]:
        if self.table_grounder is None:
            return {"company_name": company_name, "missing_reason": "numeric_grounding_disabled"}

        query_plan, route_info = self._number_query_plan_for_company(company_name, question)
        selected_doc_id = self._selected_doc_id(company_name, route_info, query_plan)
        allowed_doc_ids = [selected_doc_id] if selected_doc_id else list(query_plan.filters.candidate_doc_ids or [])
        if selected_doc_id:
            query_plan.filters.candidate_doc_ids = [selected_doc_id]
        grounding_result = self._cached_revenue_grounding(
            company_name=company_name,
            question=question,
            query_plan=query_plan,
            candidate_doc_ids=allowed_doc_ids,
            allowed_doc_ids=allowed_doc_ids,
            retrieval_results=[],
        )
        if grounding_result is None or grounding_result.get("normalized_value") is None:
            return {
                "company_name": company_name,
                "query_plan": query_plan,
                "route_info": route_info,
                "missing_reason": "revenue_grounding_not_found",
            }

        return {
            "company_name": company_name,
            "value_yuan": grounding_result.get("normalized_value"),
            "grounded_answer": self._grounded_number_value(grounding_result),
            "grounding_result": grounding_result,
            "retrieval_result": self._table_grounding_retrieval_result(grounding_result),
            "retrieval_results": [
                self._table_grounding_retrieval_result(grounding_result),
                *[
                    self._table_grounding_retrieval_result(
                        support_result,
                        chunk_type="table_support",
                        retrieval_source="table_support",
                    )
                    for support_result in self._table_support_results(grounding_result)
                ],
            ],
            "query_plan": query_plan,
            "route_info": route_info,
        }

    def _numeric_rule_citations(self, grounded_items: List[Dict[str, Any]], pages: List[int]) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        for item in grounded_items:
            retrieval_results = item.get("retrieval_results") or ([item["retrieval_result"]] if item.get("retrieval_result") else [])
            grounding_result = item.get("grounding_result")
            if not retrieval_results or not grounding_result:
                continue
            page = grounding_result.get("page")
            if page not in pages:
                continue
            citations.extend(
                build_citations(
                    retrieval_results,
                    pages,
                    table_grounding_result=grounding_result,
                    table_support_results=self._table_support_results(grounding_result),
                )
            )
        return dedupe_citations(citations)

    def _build_numeric_rule_answer(
        self,
        *,
        question: str,
        final_answer: Any,
        query_plan: QueryPlan,
        route_info: Dict[str, Any],
        grounded_items: List[Dict[str, Any]],
        rule_name: str,
        reasoning: str,
    ) -> Dict[str, Any]:
        retrieval_results = []
        for item in grounded_items:
            retrieval_results.extend(item.get("retrieval_results") or ([item["retrieval_result"]] if item.get("retrieval_result") else []))
        pages = [
            result.get("page")
            for item in grounded_items
            for result in (item.get("retrieval_results") or ([item["retrieval_result"]] if item.get("retrieval_result") else []))
            if result.get("page") is not None
        ]
        pages = list(dict.fromkeys(pages))
        if final_answer in (None, "N/A"):
            references: List[Dict[str, Any]] = []
            citations: List[Dict[str, Any]] = []
            pages = []
        else:
            references = self._extract_references_from_results(pages, retrieval_results)
            citations = self._numeric_rule_citations(grounded_items, pages)

        answer = {
            "final_answer": final_answer if final_answer is not None else "N/A",
            "relevant_pages": pages,
            "references": references,
            "citations": citations,
            "confidence": "high" if final_answer not in (None, "N/A") else "low",
            "confidence_reason": f"{rule_name} 由表格 grounding 的结构化数值直接计算。" if final_answer not in (None, "N/A") else f"{rule_name} 缺少关键数值，返回 N/A。",
            "reasoning_summary": reasoning,
            "step_by_step_analysis": reasoning if self.reasoning_debug_enabled else "",
            "search_queries": [question],
            "query_plan": query_plan.to_dict(),
            "route_info": route_info,
            "candidate_pool_size_before_rerank": None,
            "reranking_strategy": self.reranking_strategy,
            "initial_candidate_pool_size": None,
            "colbert_candidate_pool_size": None,
            "colbert_top_n": self.colbert_top_n if self.reranking_strategy == "cascade" else None,
            "final_reranking_backend": self.final_reranking_backend or os.getenv("RERANKING_BACKEND", "llm_prompt").lower(),
            "hyde": self._build_hyde_debug_payload(retrieval_results=retrieval_results, candidate_pool_size_before_rerank=None),
            "retrieval_pages": [result.get("page") for result in retrieval_results],
            "retrieval_results": [self._serialize_retrieval_result(result) for result in retrieval_results],
            "retrieval_report_groups": self._aggregate_retrieval_results_by_report(retrieval_results),
            "response_data": {},
            "validation_flags": [],
        }
        if grounded_items and len(grounded_items) == 1 and grounded_items[0].get("grounding_result"):
            answer["table_grounding_result"] = grounded_items[0]["grounding_result"]
        return answer

    def _answer_structured_revenue_comparison(
        self,
        question: str,
        companies: List[str],
        schema: str,
    ) -> Optional[Dict[str, Any]]:
        if self.table_grounder is None or len(companies) < 2 or not self._is_revenue_metric_question(question):
            return None
        if not any(term in (question or "") for term in ("谁", "更高", "高于", "超过", "大于")):
            return None

        ordered_companies = self._companies_in_question_order(question, companies)[:2]
        grounded_items = [self._ground_revenue_for_company(company, question) for company in ordered_companies]
        missing = [item for item in grounded_items if item.get("value_yuan") is None]
        query_plan = self._build_query_plan(
            question,
            schema="comparative",
            mentioned_companies=ordered_companies,
            route_mode="numeric_comparison_rule",
        )
        route_info = {
            "route_mode": "numeric_comparison_rule",
            "selected_company": None,
            "candidate_companies": ordered_companies,
            "selection_reasons": ["structured_revenue_comparison"],
            "numeric_comparison_rule": {
                "metric": "营业收入",
                "company_values": [
                    {
                        "company_name": item.get("company_name"),
                        "value_yuan": item.get("value_yuan"),
                        "source_doc_id": (item.get("grounding_result") or {}).get("source_doc_id"),
                        "page": (item.get("grounding_result") or {}).get("page"),
                        "missing_reason": item.get("missing_reason"),
                    }
                    for item in grounded_items
                ],
            },
        }

        if missing:
            reasoning = "结构化营业收入比较缺少至少一个公司的表格 grounding 数值，因此返回 N/A。"
            return self._build_numeric_rule_answer(
                question=question,
                final_answer="N/A",
                query_plan=query_plan,
                route_info=route_info,
                grounded_items=grounded_items,
                rule_name="numeric_comparison_rule",
                reasoning=reasoning,
            )

        first, second = grounded_items[0], grounded_items[1]
        if "谁" in question or "哪" in question:
            winner = first if float(first["value_yuan"]) >= float(second["value_yuan"]) else second
            final_answer = winner["company_name"]
            reasoning = (
                f"结构化比较营业收入：{first['company_name']}={first['value_yuan']} 元，"
                f"{second['company_name']}={second['value_yuan']} 元，因此答案为 {final_answer}。"
            )
        elif "是否" in question and any(term in question for term in ("高于", "超过", "大于")):
            final_answer = bool(float(first["value_yuan"]) > float(second["value_yuan"]))
            reasoning = (
                f"结构化判断营业收入：{first['company_name']}={first['value_yuan']} 元，"
                f"{second['company_name']}={second['value_yuan']} 元，比较结果为 {final_answer}。"
            )
        else:
            return None

        return self._build_numeric_rule_answer(
            question=question,
            final_answer=final_answer,
            query_plan=query_plan,
            route_info=route_info,
            grounded_items=grounded_items,
            rule_name="numeric_comparison_rule",
            reasoning=reasoning,
        )

    def _answer_structured_revenue_threshold(
        self,
        *,
        question: str,
        schema: str,
        route_decision: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self.table_grounder is None or (schema or "").lower() != "boolean" or not self._is_revenue_metric_question(question):
            return None
        if "是否" not in (question or "") or not any(term in question for term in ("超过", "高于", "大于")):
            return None
        threshold_yuan = self._threshold_value_yuan(question)
        if threshold_yuan is None:
            return None

        company_name = route_decision.get("company_name") or ""
        if not company_name:
            return None
        grounded_item = self._ground_revenue_for_company(company_name, question)
        query_plan: QueryPlan = route_decision["query_plan"]
        route_info = dict(route_decision.get("route_info") or {})
        route_info.update(
            {
                "route_mode": "numeric_threshold_rule",
                "numeric_threshold_rule": {
                    "metric": "营业收入",
                    "threshold_yuan": threshold_yuan,
                    "company_value_yuan": grounded_item.get("value_yuan"),
                    "source_doc_id": (grounded_item.get("grounding_result") or {}).get("source_doc_id"),
                    "page": (grounded_item.get("grounding_result") or {}).get("page"),
                    "missing_reason": grounded_item.get("missing_reason"),
                },
            }
        )
        if grounded_item.get("value_yuan") is None:
            reasoning = "结构化营业收入阈值判断缺少表格 grounding 数值，因此返回 N/A。"
            final_answer: Any = "N/A"
        else:
            final_answer = bool(float(grounded_item["value_yuan"]) > float(threshold_yuan))
            reasoning = (
                f"结构化阈值判断营业收入：{company_name}={grounded_item['value_yuan']} 元，"
                f"阈值={threshold_yuan} 元，比较结果为 {final_answer}。"
            )

        return self._build_numeric_rule_answer(
            question=question,
            final_answer=final_answer,
            query_plan=query_plan,
            route_info=route_info,
            grounded_items=[grounded_item],
            rule_name="numeric_threshold_rule",
            reasoning=reasoning,
        )

    def get_answer_for_company(
        self,
        company_name: str,
        question: str,
        schema: str,
        query_plan: Optional[QueryPlan] = None,
        route_info: Optional[Dict] = None,
    ) -> Dict:
        if self._resolved_graph_engine() == "legacy":
            return run_legacy_single_company(
                self,
                company_name=company_name,
                question=question,
                schema=schema,
                query_plan=query_plan,
                route_info=route_info,
            )
        graph = build_single_company_graph(self)
        final_state = graph.invoke(
            {
                "company_name": company_name,
                "question": question,
                "schema": schema,
                "query_plan": query_plan,
                "route_info": route_info,
            }
        )
        return dict(final_state.get("final_answer_dict") or {})

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

    def process_question(self, question: str, schema: str, question_meta: Optional[Dict[str, Any]] = None):
        if self._resolved_graph_engine() == "legacy":
            return run_legacy_question(
                self,
                question=question,
                schema=schema,
                question_meta=question_meta,
            )
        return run_finarag_graph(
            self,
            question=question,
            schema=schema,
            question_meta=question_meta,
        )

    @staticmethod
    def _ordered_processed_questions(processed_by_index: Dict[int, Dict]) -> List[Dict]:
        return [processed_by_index[index] for index in sorted(processed_by_index)]

    @staticmethod
    def _is_retryable_resume_entry(question_entry: Dict) -> bool:
        if not isinstance(question_entry, dict):
            return False
        validation_flags = question_entry.get("validation_flags") or []
        return bool(question_entry.get("error")) or "processing_error" in validation_flags

    @staticmethod
    def _get_question_identity(question_entry: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not isinstance(question_entry, dict):
            return None, None, None
        question_id = question_entry.get("question_id")
        question_text = (
            question_entry.get("question_text")
            or question_entry.get("text")
            or question_entry.get("question")
        )
        kind = question_entry.get("kind") or question_entry.get("schema")
        return question_id, question_text, kind

    def _match_resume_question_index(
        self,
        resume_entry: Dict,
        indexes_by_id: Dict[str, List[int]],
        indexes_by_text_kind: Dict[Tuple[Optional[str], Optional[str]], List[int]],
        used_indexes: set[int],
    ) -> Optional[int]:
        question_id, question_text, kind = self._get_question_identity(resume_entry)
        if question_id is not None:
            for index in indexes_by_id.get(str(question_id), []):
                if index not in used_indexes:
                    return index

        for index in indexes_by_text_kind.get((question_text, kind), []):
            if index not in used_indexes:
                return index
        return None

    def _load_resume_progress(
        self,
        questions_with_index: List[Dict],
        resume_from: Union[str, Path],
        retry_errors: bool = True,
    ) -> Tuple[Dict[int, Dict], List[Optional[Dict]], Dict[str, Union[str, int, bool, None]]]:
        output_file = Path(resume_from)
        debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)

        resume_questions: List[Dict] = []
        stored_answer_details: List[Optional[Dict]] = []
        source_path: Optional[Path] = None
        source_type: Optional[str] = None

        if debug_file.exists():
            with open(debug_file, "r", encoding="utf-8") as file:
                payload = json.load(file)
            if isinstance(payload, dict):
                loaded_questions = payload.get("questions")
                loaded_answer_details = payload.get("answer_details")
                if isinstance(loaded_questions, list):
                    resume_questions = loaded_questions
                if isinstance(loaded_answer_details, list):
                    stored_answer_details = loaded_answer_details
            source_path = debug_file
            source_type = "debug"
        elif output_file.exists():
            with open(output_file, "r", encoding="utf-8") as file:
                payload = json.load(file)
            if isinstance(payload, dict):
                loaded_answers = payload.get("answers")
                if isinstance(loaded_answers, list):
                    resume_questions = loaded_answers
            source_path = output_file
            source_type = "answers"

        total_questions = len(questions_with_index)
        answer_details: List[Optional[Dict]] = [None] * total_questions
        if not resume_questions:
            return {}, answer_details, {
                "resume_source": source_type,
                "resume_path": str(source_path) if source_path is not None else None,
                "loaded_count": 0,
                "retry_count": 0,
                "unmatched_count": 0,
            }

        indexes_by_id: Dict[str, List[int]] = defaultdict(list)
        indexes_by_text_kind: Dict[Tuple[Optional[str], Optional[str]], List[int]] = defaultdict(list)
        for question_data in questions_with_index:
            index = question_data["_question_index"]
            question_id = question_data.get("id")
            if question_id is not None:
                indexes_by_id[str(question_id)].append(index)
            indexes_by_text_kind[(question_data.get("text"), question_data.get("kind"))].append(index)

        processed_by_index: Dict[int, Dict] = {}
        used_indexes: set[int] = set()
        retry_count = 0
        unmatched_count = 0

        for resume_entry in resume_questions:
            if not isinstance(resume_entry, dict):
                unmatched_count += 1
                continue

            question_index = self._match_resume_question_index(
                resume_entry,
                indexes_by_id=indexes_by_id,
                indexes_by_text_kind=indexes_by_text_kind,
                used_indexes=used_indexes,
            )
            if question_index is None:
                unmatched_count += 1
                continue

            if retry_errors and self._is_retryable_resume_entry(resume_entry):
                retry_count += 1
                continue

            processed_by_index[question_index] = resume_entry
            used_indexes.add(question_index)
            if question_index < len(stored_answer_details):
                answer_details[question_index] = stored_answer_details[question_index]

        return processed_by_index, answer_details, {
            "resume_source": source_type,
            "resume_path": str(source_path) if source_path is not None else None,
            "loaded_count": len(processed_by_index),
            "retry_count": retry_count,
            "unmatched_count": unmatched_count,
        }

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
                "chart_grounding_result": answer_dict.get("chart_grounding_result"),
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

    def process_questions_list(
        self,
        questions_list: List[Dict],
        output_path: str = None,
        pipeline_details: str = "",
        resume_from: Optional[Union[str, Path]] = None,
        retry_errors: bool = True,
    ) -> Dict:
        total_questions = len(questions_list)
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions
        processed_by_index: Dict[int, Dict] = {}
        parallel_threads = self.parallel_requests

        if resume_from is not None:
            processed_by_index, self.answer_details, resume_meta = self._load_resume_progress(
                questions_with_index,
                resume_from=resume_from,
                retry_errors=retry_errors,
            )
            if resume_meta.get("resume_path"):
                print(
                    f"Resume enabled: loaded {resume_meta['loaded_count']} completed questions from "
                    f"{resume_meta['resume_path']}"
                )
                if resume_meta.get("retry_count"):
                    print(f"Retrying {resume_meta['retry_count']} previously failed questions.")
                if resume_meta.get("unmatched_count"):
                    print(
                        f"Warning: ignored {resume_meta['unmatched_count']} unmatched resume entries from "
                        f"{resume_meta['resume_path']}."
                    )

        remaining_questions = [
            question_data
            for question_data in questions_with_index
            if question_data["_question_index"] not in processed_by_index
        ]

        with tqdm(total=total_questions, initial=len(processed_by_index), desc="Processing questions") as pbar:
            if parallel_threads <= 1:
                for question_data in remaining_questions:
                    processed_question = self._process_single_question(question_data)
                    processed_by_index[question_data["_question_index"]] = processed_question
                    if output_path:
                        self._save_progress(
                            self._ordered_processed_questions(processed_by_index),
                            output_path,
                            pipeline_details=pipeline_details,
                        )
                    pbar.update(1)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                    future_to_question = {
                        executor.submit(self._process_single_question, question_data): question_data
                        for question_data in remaining_questions
                    }
                    for future in concurrent.futures.as_completed(future_to_question):
                        question_data = future_to_question[future]
                        processed_question = future.result()
                        processed_by_index[question_data["_question_index"]] = processed_question

                        if output_path:
                            self._save_progress(
                                self._ordered_processed_questions(processed_by_index),
                                output_path,
                                pipeline_details=pipeline_details,
                            )
                        pbar.update(1)

        processed_questions = self._ordered_processed_questions(processed_by_index)
        if output_path:
            self._save_progress(processed_questions, output_path, pipeline_details=pipeline_details)

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
            question_meta = {
                "question_id": question_data.get("id"),
                "capability": question_data.get("capability"),
                "expected_filters": question_data.get("expected_filters"),
                "metadata": question_data.get("metadata") or {},
            }
            answer_dict = self.process_question(question_text, schema, question_meta=question_meta)

            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
                return {
                    "question_id": question_data.get("id"),
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
                "question_id": question_data.get("id"),
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
            return self._handle_processing_error(
                question_text,
                schema,
                err,
                question_index,
                question_id=question_data.get("id"),
            )

    def _handle_processing_error(
        self,
        question_text: str,
        schema: str,
        err: Exception,
        question_index: int,
        question_id: Optional[str] = None,
    ) -> Dict:
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
            "question_id": question_id,
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
            chart_grounding_result = None
            if answer_details_ref and answer_details_ref.startswith("#/answer_details/"):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                        step_by_step_analysis = self.answer_details[index].get("step_by_step_analysis")
                        table_grounding_result = self.answer_details[index].get("table_grounding_result")
                        chart_grounding_result = self.answer_details[index].get("chart_grounding_result")
                except (ValueError, IndexError):
                    pass

            if value == "N/A":
                references = []
                citations = []
            else:
                normalized_references = []
                for ref in references:
                    pdf_sha1 = ref.get("pdf_sha1") or ref.get("source") or ref.get("doc_id")
                    page = ref.get("page")
                    if isinstance(page, int):
                        page_index = page - 1
                    else:
                        page_index = ref.get("page_index")
                    if not pdf_sha1 or not isinstance(page_index, int):
                        continue
                    normalized_references.append(
                        {
                            "pdf_sha1": pdf_sha1,
                            "page_index": max(0, page_index),
                        }
                    )
                references = normalized_references

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
            if chart_grounding_result:
                submission_answer["chart_grounding_result"] = chart_grounding_result

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
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_file, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        answers = self._post_process_submission_answers(processed_questions)
        result_output = {
            "answers": answers,
            "details": pipeline_details
        }
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(result_output, file, ensure_ascii=False, indent=2)

    def process_all_questions(
        self,
        output_path: str = 'questions_with_answers.json',
        pipeline_details: str = "",
        resume_from: Optional[Union[str, Path]] = None,
        retry_errors: bool = True,
    ) -> Dict:
        return self.process_questions_list(
            self.questions,
            output_path,
            pipeline_details=pipeline_details,
            resume_from=resume_from,
            retry_errors=retry_errors,
        )

    def process_comparative_question(self, question: str, companies: List[str], schema: str) -> Dict:
        structured_answer = self._answer_structured_revenue_comparison(question, companies, schema)
        if structured_answer is not None:
            return structured_answer

        if self._resolved_graph_engine() == "legacy":
            return run_legacy_comparative(
                self,
                question=question,
                companies=companies,
                schema=schema,
            )
        graph = build_comparative_graph(self)
        final_state = graph.invoke(
            {
                "question": question,
                "companies": companies,
                "schema": schema,
            }
        )
        return dict(final_state.get("final_answer_dict") or {})
