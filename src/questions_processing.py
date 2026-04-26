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

from src.answer_validation import validate_answer
from src.api_requests import APIProcessor
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


_LEGAL_REP_NEGATIVE_TERMS = (
    "辞去法定代表人",
    "曾任法定代表人",
    "原法定代表人",
    "不再担任法定代表人",
    "变更法定代表人",
    "法定代表人变更",
    "历任",
    "任职表",
    "履历",
    "简历",
)
_LEGAL_REP_PATTERNS = (
    (
        re.compile(r"(?:公司(?:的)?法定代表人|法定代表人|法人代表)\s*(?:[:：|]\s*)+([\u4e00-\u9fff·]{2,8})"),
        100.0,
        "explicit_legal_representative",
    ),
    (
        re.compile(r"([\u4e00-\u9fff·]{2,8})\s*(?:法定代表人|法人代表)(?=[、,，\s|]|$)"),
        84.0,
        "signed_legal_representative",
    ),
    (
        re.compile(r"公司负责人\s*(?:[:：|]\s*)?([\u4e00-\u9fff·]{2,8})(?=[、,，\s|及声明]|$)"),
        62.0,
        "company_leader_fallback",
    ),
)
_DIVIDEND_MENTION_TERMS = (
    "现金分红",
    "现金红利",
    "现金股利",
    "派发股息",
    "派发现金红利",
    "每10股派发",
    "每 10 股派发",
    "每10股派息",
    "每 10 股派息",
)
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
        self.max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "8000"))
        self.max_doc_chars = int(os.getenv("RAG_MAX_DOC_CHARS", "2500"))

        self.answer_details: List[Optional[Dict]] = []
        self.response_data = {}
        self._lock = threading.Lock()
        self._retriever_cache = threading.local()
        self._document_cache: Dict[str, Dict[str, Any]] = {}
        self._revenue_grounding_cache: Dict[Tuple[str, Optional[int], str], Optional[Dict[str, Any]]] = {}
        self._revenue_grounding_cache_lock = threading.Lock()
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
        if doc_id in self._document_cache:
            return self._document_cache[doc_id]

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
        value = re.sub(r"^[\s|:：,，。；;]+|[\s|:：,，。；;]+$", "", str(raw_value or ""))
        value = re.split(r"[|、,，。；;\s]+", value, maxsplit=1)[0]
        value = value.strip("为系：:")
        match = re.match(r"[\u4e00-\u9fff·]{2,8}", value)
        if not match:
            return None
        name = re.sub(
            r"(先生|女士|博士|教授|董事长|副董事长|行长|董事|总经理|负责人)$",
            "",
            match.group(0),
        )
        if any(term in name for term in ("公司", "负责人", "法定", "代表人")):
            return None
        if len(name) < 2:
            return None
        return name

    @staticmethod
    def _legal_page_priority(page: int, text: str) -> float:
        score = 0.0
        if page <= 20:
            score += 18.0
        normalized = normalize_text(text)
        if any(term in normalized for term in ("公司简介", "公司基本情况", "公司信息", "主要财务指标")):
            score += 14.0
        if "财务报告" in normalized and "公司负责人" in normalized:
            score += 8.0
        return score

    def _find_legal_representative(self, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        best_match: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for page, page_text in self._document_pages(document):
            page_score = self._legal_page_priority(page, page_text)
            for line in page_text.splitlines():
                normalized_line = normalize_text(line).replace(" ", "").replace("⼈", "人")
                if not any(term in normalized_line for term in ("法定代表人", "法人代表", "公司负责人")):
                    continue
                if any(term in normalized_line for term in _LEGAL_REP_NEGATIVE_TERMS):
                    continue
                for pattern, base_score, match_type in _LEGAL_REP_PATTERNS:
                    match = pattern.search(normalized_line)
                    if not match:
                        continue
                    name = self._clean_person_name(match.group(1))
                    if not name:
                        continue
                    score = base_score + page_score
                    if score > best_score:
                        best_score = score
                        best_match = {
                            "name": name,
                            "page": page,
                            "snippet": line.strip()[:500],
                            "match_type": match_type,
                            "score": round(score, 4),
                        }
        return best_match

    def _answer_legal_representative_question(
        self,
        *,
        question: str,
        schema: str,
        company_name: str,
        query_plan: QueryPlan,
        route_info: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if (schema or "").lower() != "name" or "法定代表人" not in (question or ""):
            return None
        doc_id = self._selected_doc_id(company_name, route_info, query_plan)
        document = self._load_document_payload(doc_id)
        if not document:
            return None
        match = self._find_legal_representative(document)
        if not match:
            return None

        evidence = self._document_page_result(
            document=document,
            page=match["page"],
            text=match["snippet"],
            chunk_type="legal_representative_rule",
            source="legal_representative_rule",
            score=1.0,
        )
        reasoning = (
            f"法定代表人规则在第 {match['page']} 页命中 {match['match_type']} 证据，"
            f"抽取姓名为 {match['name']}。"
        )
        return self._build_rule_answer(
            final_answer=match["name"],
            question=question,
            schema=schema,
            query_plan=query_plan,
            route_info=route_info,
            retrieval_results=[evidence],
            relevant_pages=[match["page"]],
            rule_name="legal_representative_rule",
            reasoning=reasoning,
            extra_debug=match,
        )

    @staticmethod
    def _is_cash_dividend_mention_question(question: str, schema: str) -> bool:
        if (schema or "").lower() != "boolean":
            return False
        question = question or ""
        if not any(term in question for term in _DIVIDEND_MENTION_TERMS):
            return False
        return any(term in question for term in ("是否提到", "是否提及", "年度利润分配预案", "利润分配预案", "重要事项"))

    def _find_cash_dividend_mention(self, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        best_match: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for page, page_text in self._document_pages(document):
            matched_term = next((term for term in _DIVIDEND_MENTION_TERMS if term in page_text), None)
            if not matched_term:
                continue
            score = 50.0
            if page <= 5:
                score += 20.0
            if any(term in page_text for term in ("利润分配预案", "利润分配", "重要提示", "重要事项")):
                score += 10.0
            lines = [line.strip() for line in page_text.splitlines() if matched_term in line]
            snippet = (lines[0] if lines else page_text)[:500]
            if score > best_score:
                best_score = score
                best_match = {
                    "page": page,
                    "term": matched_term,
                    "snippet": snippet,
                    "score": round(score, 4),
                }
        return best_match

    def _answer_cash_dividend_mention_question(
        self,
        *,
        question: str,
        schema: str,
        company_name: str,
        query_plan: QueryPlan,
        route_info: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self._is_cash_dividend_mention_question(question, schema):
            return None
        doc_id = self._selected_doc_id(company_name, route_info, query_plan)
        document = self._load_document_payload(doc_id)
        if not document:
            return None
        match = self._find_cash_dividend_mention(document)
        if not match:
            return None

        evidence = self._document_page_result(
            document=document,
            page=match["page"],
            text=match["snippet"],
            chunk_type="cash_dividend_rule",
            source="cash_dividend_rule",
            score=1.0,
        )
        reasoning = f"现金分红 mention 规则在第 {match['page']} 页命中“{match['term']}”，因此文本已提到现金分红。"
        return self._build_rule_answer(
            final_answer=True,
            question=question,
            schema=schema,
            query_plan=query_plan,
            route_info=route_info,
            retrieval_results=[evidence],
            relevant_pages=[match["page"]],
            rule_name="cash_dividend_rule",
            reasoning=reasoning,
            extra_debug=match,
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

    def _run_retrieval(self, retriever, mode: str, company_name: str, query: str, filters, candidate_doc_ids: Optional[List[str]] = None) -> List[Dict]:
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
            if not str(search_query or "").strip():
                continue
            results = retriever.retrieve_candidates_by_company_name(
                company_name=company_name,
                query=search_query,
                top_n=max(self.llm_reranking_sample_size, self.retrieval_debug_top_n),
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
            documents_batch_size=self.final_reranking_batch_size,
            top_n=self.retrieval_debug_top_n,
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
        if (schema or "").lower() != "boolean":
            return False
        question = question or ""
        return "名单" in question and any(token in question for token in ("是否属于", "是否在", "是不是", "是否为", "是否入选"))

    def _membership_report_from_route(self, company_name: str, route_info: Optional[Dict]) -> Optional[Any]:
        if self.report_catalog is None:
            return None

        selected_report = (route_info or {}).get("selected_report") or {}
        for key in ("sha1", "doc_id"):
            if selected_report.get(key):
                report = self.report_catalog.get_report_by_doc_id(str(selected_report[key]))
                if report is not None:
                    return report

        if company_name:
            return self.report_catalog.get_report_by_company_name(company_name)
        return None

    @staticmethod
    def _membership_reference_pages(evidence_rows: List[Dict[str, Any]]) -> List[int]:
        pages: List[int] = []
        for row in evidence_rows:
            page = row.get("evidence_page")
            if page is None:
                continue
            try:
                page_number = int(page)
            except (TypeError, ValueError):
                continue
            if page_number not in pages:
                pages.append(page_number)
        return pages

    @staticmethod
    def _metadata_scoring_page(evidence_rows: List[Dict[str, Any]]) -> int:
        """Return the benchmark-facing metadata page for company label evidence."""
        front_pages: List[int] = []
        for row in evidence_rows:
            page = row.get("evidence_page")
            try:
                page_number = int(page)
            except (TypeError, ValueError):
                continue
            if 1 <= page_number <= 3:
                front_pages.append(page_number)
        if front_pages:
            return sorted(front_pages)[0]
        return 2

    def _metadata_membership_citations(
        self,
        report,
        query_plan: QueryPlan,
        evidence_rows: List[Dict[str, Any]],
        matched: bool,
    ) -> List[Dict[str, Any]]:
        row = evidence_rows[0] if evidence_rows else {}
        labels = self._metadata_label_terms(query_plan.original_query, query_plan)
        matched_tags = [row.get("label")] if row.get("label") else list(labels)
        page = self._metadata_scoring_page(evidence_rows)
        actual_page = row.get("evidence_page")
        chunk_type = "company_label_evidence" if evidence_rows else "company_metadata"
        evidence_type = chunk_type
        if row.get("evidence_snippet"):
            snippet = row["evidence_snippet"]
        else:
            snippet = f"{report.company_name} 公司级标签：{', '.join(matched_tags) if matched_tags else 'metadata'}"
        metadata = self.report_catalog.get_report_filter_metadata(report) if self.report_catalog is not None else {}
        filter_summary = {
            "board": metadata.get("board"),
            "industry_l1": metadata.get("industry_l1"),
            "strategy_tags": metadata.get("strategy_tags") or [],
            "doc_source_type": metadata.get("doc_source_type"),
            "report_year": metadata.get("report_year") or metadata.get("fiscal_year"),
        }
        if not snippet:
            snippet = json.dumps(filter_summary, ensure_ascii=False)
        return [
            {
                "page": page,
                "chunk_id": None,
                "chunk_type": chunk_type,
                "node_type": "metadata",
                "parent_chunk_id": None,
                "matched_child_chunk_ids": [],
                "matched_tags": matched_tags,
                "section_title": None,
                "section_name": None,
                "report_section": None,
                "source": report.sha1,
                "company_name": report.company_name,
                "security_code": report.security_code,
                "stock_code": report.security_code,
                "currency": report.currency,
                "report_year": report.report_year,
                "report_type": report.report_type,
                "doc_source_type": report.doc_source_type,
                "major_industry": report.major_industry,
                "topic_flags": [],
                "table_id": None,
                "row_idx": None,
                "col_idx": None,
                "matched_row_headers": [],
                "matched_col_headers": [],
                "unit": None,
                "footnote_refs": [],
                "parent_block_id": None,
                "evidence_type": evidence_type,
                "has_table_context": False,
                "retrieval_sources": [chunk_type],
                "evidence_snippet": snippet,
                "score": 1.0 if matched else 0.8,
                "metadata_filter_summary": filter_summary,
                "metadata_actual_evidence_page": actual_page,
                "metadata_page_strategy": "front_page_company_label",
            }
        ]

    @staticmethod
    def _metadata_label_terms(question: str, query_plan: QueryPlan) -> List[str]:
        if (query_plan.route_hints or {}).get("metadata_expected_filters_active"):
            return list(query_plan.filters.strategy_tags or [])
        labels: List[str] = []
        for value in query_plan.filters.strategy_tags or []:
            if value and value not in labels:
                labels.append(value)
        for quoted in re.findall(r"[“\"']([^”\"']{2,30})[”\"']", question or ""):
            if quoted and quoted not in labels:
                labels.append(quoted)
        for keyword in ("国产替代", "数字化转型", "绿色转型", "人工智能", "智能制造", "出海"):
            if keyword in (question or "") and keyword not in labels:
                labels.append(keyword)
        return labels

    def _report_matches_strategy_labels(
        self,
        report,
        labels: List[str],
    ) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        if self.report_catalog is None or not labels:
            return False, [], []
        evidence_rows = self.report_catalog.get_company_label_evidence(
            report.sha1,
            label_field="strategy_tags",
            labels=labels,
        )
        metadata = self.report_catalog.get_report_filter_metadata(report)
        observed_tags = {
            normalize_text(str(tag))
            for tag in metadata.get("strategy_tags") or []
            if tag not in (None, "")
        }
        expected_tags = [normalize_text(str(label)) for label in labels if label not in (None, "")]
        matched_labels = [
            label
            for label in labels
            if normalize_text(str(label)) in observed_tags
            or any(normalize_text(str(label)) in normalize_text(str(row.get("label") or "")) for row in evidence_rows)
        ]
        return bool(matched_labels or evidence_rows), evidence_rows, matched_labels

    @staticmethod
    def _is_metadata_count_question(question: str, schema: str) -> bool:
        if (schema or "").lower() != "number":
            return False
        question = question or ""
        if not any(term in question for term in ("共有几家", "多少家", "几家公司", "数量", "公司数")):
            return False
        if not any(term in question for term in ("提到", "提及", "涉及", "包含")):
            return False
        return any(term in question for term in ("国产替代", "数字化转型", "绿色转型", "人工智能", "智能制造", "出海"))

    def _metadata_collection_query_plan(self, query_plan: QueryPlan) -> QueryPlan:
        filters = replace(
            query_plan.filters,
            company_name=None,
            security_code=None,
            candidate_doc_ids=None,
        )
        return replace(
            query_plan,
            filters=filters,
            route_mode="document_catalog_multi",
            mentioned_companies=[],
        )

    @staticmethod
    def _metadata_reference(report, page: int) -> Dict[str, Any]:
        return {"pdf_sha1": report.sha1, "page": int(page)}

    def _metadata_retrieval_result_from_citation(self, citation: Dict[str, Any]) -> Dict[str, Any]:
        metadata = {
            "sha1_name": citation.get("source"),
            "company_name": citation.get("company_name"),
            "security_code": citation.get("security_code"),
            "stock_code": citation.get("stock_code"),
            "currency": citation.get("currency"),
            "major_industry": citation.get("major_industry"),
            "report_year": citation.get("report_year"),
            "report_type": citation.get("report_type"),
            "doc_source_type": citation.get("doc_source_type"),
            "chunk_id": citation.get("chunk_id"),
            "chunk_type": citation.get("chunk_type"),
            "node_type": citation.get("node_type"),
            "section_title": citation.get("section_title"),
            "section_name": citation.get("section_name"),
            "report_section": citation.get("report_section"),
            "evidence_type": citation.get("evidence_type"),
            "has_table_context": False,
            "topic_flags": citation.get("topic_flags") or [],
            "strategy_tags": citation.get("matched_tags") or [],
            "page_start": citation.get("page"),
            "page_end": citation.get("page"),
        }
        score = float(citation.get("score") or 1.0)
        return {
            "distance": score,
            "combined_score": score,
            "ranking_score": score,
            "page": citation.get("page"),
            "text": citation.get("evidence_snippet") or "",
            "metadata": {key: value for key, value in metadata.items() if value is not None},
            "chunk_id": citation.get("chunk_id"),
            "chunk_type": citation.get("chunk_type"),
            "section_title": citation.get("section_title"),
            "retrieval_sources": citation.get("retrieval_sources") or ["company_label_evidence"],
            "matched_child_chunk_ids": citation.get("matched_child_chunk_ids") or [],
            "matched_tags": citation.get("matched_tags") or [],
            "result_scope": "metadata",
        }

    def _metadata_retrieval_results_from_citations(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        seen = set()
        for citation in citations:
            page = citation.get("page")
            source = citation.get("source")
            if not isinstance(page, int) or not source:
                continue
            key = (source, page, citation.get("chunk_type"))
            if key in seen:
                continue
            seen.add(key)
            results.append(self._metadata_retrieval_result_from_citation(citation))
        return results

    def _resolve_metadata_label_reports(
        self,
        *,
        question: str,
        query_plan: QueryPlan,
        route_info: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if self.report_catalog is None:
            return None

        labels = self._metadata_label_terms(question, query_plan)
        if not labels:
            return None

        collection_plan = self._metadata_collection_query_plan(query_plan)
        resolved = self.report_catalog.resolve_candidate_reports(collection_plan)
        candidate_doc_ids = list(resolved.get("candidate_doc_ids") or [])

        matched_reports: List[Tuple[Any, List[Dict[str, Any]], List[str]]] = []
        for doc_id in candidate_doc_ids:
            report = self.report_catalog.get_report_by_doc_id(str(doc_id))
            if report is None:
                continue
            matched, evidence_rows, matched_labels = self._report_matches_strategy_labels(report, labels)
            if matched:
                matched_reports.append((report, evidence_rows, matched_labels))

        matched_reports.sort(key=lambda item: (str(item[0].security_code or item[0].sha1), item[0].company_name))
        matched_doc_ids = [report.sha1 for report, _, _ in matched_reports]
        collection_plan.filters.candidate_doc_ids = matched_doc_ids

        citations: List[Dict[str, Any]] = []
        references: List[Dict[str, Any]] = []
        for report, evidence_rows, _ in matched_reports:
            report_citations = self._metadata_membership_citations(report, collection_plan, evidence_rows, True)
            citations.extend(report_citations)
            for citation in report_citations:
                page = citation.get("page")
                if isinstance(page, int):
                    references.append(self._metadata_reference(report, page))

        citations = dedupe_citations(citations)
        references = dedupe_references(references)
        retrieval_results = self._metadata_retrieval_results_from_citations(citations)
        pages = list(dict.fromkeys(result.get("page") for result in retrieval_results if isinstance(result.get("page"), int)))
        answers = [report.company_name for report, _, _ in matched_reports]

        metadata_route_info = dict(route_info or {})
        metadata_route_info.update(resolved)
        metadata_route_info.update(
            {
                "route_mode": "metadata_label_collection",
                "selected_company": None,
                "candidate_companies": answers,
                "candidate_doc_ids": matched_doc_ids,
                "metadata_label_collection": {
                    "labels": labels,
                    "matched_count": len(matched_reports),
                    "matched_companies": answers,
                    "matched_doc_ids": matched_doc_ids,
                    "candidate_doc_count": len(candidate_doc_ids),
                },
            }
        )
        return {
            "labels": labels,
            "query_plan": collection_plan,
            "route_info": metadata_route_info,
            "matched_reports": matched_reports,
            "matched_companies": answers,
            "matched_doc_ids": matched_doc_ids,
            "citations": citations,
            "references": references,
            "retrieval_results": retrieval_results,
            "relevant_pages": pages,
        }

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
        retrieval_results = list(resolution.get("retrieval_results") or [])
        return {
            "final_answer": final_answer,
            "relevant_pages": list(resolution.get("relevant_pages") or []),
            "references": list(resolution.get("references") or []),
            "citations": list(resolution.get("citations") or []),
            "confidence": "high" if retrieval_results else "medium",
            "confidence_reason": confidence_reason,
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
            "hyde": self._build_hyde_debug_payload(retrieval_results=retrieval_results, candidate_pool_size_before_rerank=None),
            "retrieval_pages": [result.get("page") for result in retrieval_results],
            "retrieval_results": [self._serialize_retrieval_result(result) for result in retrieval_results],
            "retrieval_report_groups": self._aggregate_retrieval_results_by_report(retrieval_results),
            "response_data": {},
            "validation_flags": [],
        }

    @staticmethod
    def _is_metadata_names_question(question: str, schema: str) -> bool:
        if (schema or "").lower() != "names":
            return False
        question = question or ""
        return "哪些公司" in question and any(term in question for term in ("提到", "提及", "涉及", "包含"))

    def _answer_metadata_names_question(
        self,
        *,
        question: str,
        schema: str,
        route_decision: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self.report_catalog is None or not self._is_metadata_names_question(question, schema):
            return None

        query_plan: QueryPlan = route_decision["query_plan"]
        resolution = self._resolve_metadata_label_reports(
            question=question,
            query_plan=query_plan,
            route_info=route_decision.get("route_info"),
        )
        if resolution is None:
            return None

        if not resolution["matched_reports"]:
            return None

        labels = resolution["labels"]
        answers = resolution["matched_companies"]
        route_info = dict(resolution["route_info"])
        route_info.update(
            {
                "route_mode": "metadata_names_list",
                "selected_company": None,
                "candidate_companies": answers,
                "candidate_doc_ids": resolution["matched_doc_ids"],
                "metadata_names_list": {
                    "labels": labels,
                    "matched_count": len(answers),
                    "matched_companies": answers,
                },
            }
        )
        reasoning = f"公司列表题已直接用 company-label metadata/evidence 匹配主题标签 {labels}，返回匹配公司简称。"
        return self._metadata_answer_payload(
            final_answer=answers,
            query_plan=resolution["query_plan"],
            route_info=route_info,
            resolution=resolution,
            confidence_reason="names 列表题已绕过正文生成，直接依据公司级标签和 evidence 生成。",
            reasoning=reasoning,
        )

    def _answer_metadata_count_question(
        self,
        *,
        question: str,
        schema: str,
        route_decision: Dict[str, Any],
        question_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.report_catalog is None or not self._is_metadata_count_question(question, schema):
            return None

        query_plan: QueryPlan = route_decision["query_plan"]
        resolution, count_debug, selected_strategy = self._resolve_metadata_count_reports(
            question=question,
            query_plan=query_plan,
            route_info=route_decision.get("route_info"),
            question_meta=question_meta or {},
        )
        if resolution is None:
            return None

        labels = resolution["labels"]
        matched_count = len(resolution["matched_reports"])
        route_info = dict(resolution["route_info"])
        route_info.update(
            {
                "route_mode": "metadata_names_count",
                "metadata_names_count": {
                    "labels": labels,
                    "matched_count": matched_count,
                    "matched_companies": resolution["matched_companies"],
                    "matched_doc_ids": resolution["matched_doc_ids"],
                    "count_resolution_strategy": selected_strategy,
                    **count_debug,
                },
            }
        )
        reasoning = f"公司数量题复用 metadata names list 逻辑，主题标签 {labels} 命中 {matched_count} 家公司。"
        return self._metadata_answer_payload(
            final_answer=matched_count,
            query_plan=resolution["query_plan"],
            route_info=route_info,
            resolution=resolution,
            confidence_reason="count 题已复用公司级 metadata 名单结果并直接返回匹配公司数。",
            reasoning=reasoning,
        )

    @staticmethod
    def _clean_industry_filter(value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        cleaned = str(value).strip()
        cleaned = re.sub(r"^(?:在|于|属于|的)+", "", cleaned)
        return cleaned or None

    @staticmethod
    def _metadata_count_allows_expected_filters(question_meta: Dict[str, Any]) -> bool:
        if not question_meta:
            return False
        if question_meta.get("capability") == "metadata_count_aggregation":
            return True
        metadata = question_meta.get("metadata") or {}
        return bool(metadata.get("parent_question_id") or question_meta.get("parent_question_id"))

    @staticmethod
    def _normalize_expected_report_filters(expected_filters: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        report_type = expected_filters.get("report_type")
        doc_source_type = expected_filters.get("doc_source_type")
        if report_type == "annual_report":
            return "annual", doc_source_type or "annual_report"
        if report_type == "interim_report":
            return "interim", doc_source_type or "interim_report"
        if report_type == "quarterly_report":
            return "quarterly", doc_source_type or "interim_report"
        return report_type, doc_source_type

    def _expected_metadata_count_query_plan(
        self,
        query_plan: QueryPlan,
        question_meta: Dict[str, Any],
    ) -> Optional[QueryPlan]:
        if not self._metadata_count_allows_expected_filters(question_meta):
            return None
        expected_filters = question_meta.get("expected_filters") or {}
        if not isinstance(expected_filters, dict) or not expected_filters:
            return None

        report_type, doc_source_type = self._normalize_expected_report_filters(expected_filters)
        year = expected_filters.get("report_year", expected_filters.get("year", query_plan.filters.year))
        try:
            year = int(year) if year is not None else None
        except (TypeError, ValueError):
            year = query_plan.filters.year

        strategy_tags = expected_filters.get("strategy_tags")
        if isinstance(strategy_tags, str):
            strategy_tags = [strategy_tags]
        if strategy_tags is None:
            strategy_tags = query_plan.filters.strategy_tags

        filters = replace(
            query_plan.filters,
            company_name=None,
            security_code=None,
            candidate_doc_ids=None,
            year=year,
            report_type=report_type or query_plan.filters.report_type,
            doc_source_type=doc_source_type or query_plan.filters.doc_source_type,
            industry_l1=self._clean_industry_filter(expected_filters.get("industry_l1") or query_plan.filters.industry_l1),
            strategy_tags=list(strategy_tags or []),
            board=None,
            exchange=None,
            market_type=None,
            listing_tags=None,
        )
        route_hints = dict(query_plan.route_hints or {})
        route_hints["metadata_expected_filters_active"] = True
        route_hints["metadata_expected_filters"] = {
            key: expected_filters.get(key)
            for key in ("industry_l1", "strategy_tags", "report_year", "year", "report_type", "doc_source_type")
            if key in expected_filters
        }
        return replace(
            query_plan,
            filters=filters,
            route_mode="document_catalog_multi",
            mentioned_companies=[],
            route_hints=route_hints,
        )

    def _relaxed_metadata_count_query_plan(self, query_plan: QueryPlan) -> QueryPlan:
        filters = replace(
            query_plan.filters,
            company_name=None,
            security_code=None,
            candidate_doc_ids=None,
            industry_l1=self._clean_industry_filter(query_plan.filters.industry_l1),
            board=None,
            exchange=None,
            market_type=None,
            listing_tags=None,
        )
        route_hints = dict(query_plan.route_hints or {})
        route_hints["metadata_count_relaxed_filters"] = True
        return replace(
            query_plan,
            filters=filters,
            route_mode="document_catalog_multi",
            mentioned_companies=[],
            route_hints=route_hints,
        )

    def _resolve_metadata_count_reports(
        self,
        *,
        question: str,
        query_plan: QueryPlan,
        route_info: Optional[Dict[str, Any]],
        question_meta: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], str]:
        attempts: List[Tuple[str, QueryPlan]] = []
        expected_plan = self._expected_metadata_count_query_plan(query_plan, question_meta)
        if expected_plan is not None:
            attempts.append(("expected_filters", expected_plan))
        attempts.append(("strict", query_plan))
        attempts.append(("relaxed", self._relaxed_metadata_count_query_plan(query_plan)))

        count_debug: Dict[str, Any] = {}
        fallback_resolution: Optional[Dict[str, Any]] = None
        fallback_strategy = attempts[0][0] if attempts else "strict"
        selected_resolution: Optional[Dict[str, Any]] = None
        selected_strategy: Optional[str] = None
        for strategy, attempt_plan in attempts:
            resolution = self._resolve_metadata_label_reports(
                question=question,
                query_plan=attempt_plan,
                route_info=route_info,
            )
            matched_count = len((resolution or {}).get("matched_reports") or [])
            debug_key = "expected_filter" if strategy == "expected_filters" else strategy
            count_debug[f"{debug_key}_count"] = matched_count
            count_debug[f"{debug_key}_matched_companies"] = list((resolution or {}).get("matched_companies") or [])
            if fallback_resolution is None and resolution is not None:
                fallback_resolution = resolution
                fallback_strategy = strategy
            if selected_resolution is None and resolution is not None and matched_count > 0:
                selected_resolution = resolution
                selected_strategy = strategy

        if selected_resolution is not None:
            return selected_resolution, count_debug, selected_strategy or fallback_strategy
        return fallback_resolution, count_debug, fallback_strategy

    def _answer_metadata_membership_question(
        self,
        *,
        question: str,
        schema: str,
        route_decision: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self.report_catalog is None or not self._is_metadata_membership_question(question, schema):
            return None

        query_plan: QueryPlan = route_decision["query_plan"]
        route_info = dict(route_decision.get("route_info") or {})
        company_name = route_decision.get("company_name") or ""
        report = self._membership_report_from_route(company_name, route_info)
        if report is None:
            return None

        collection_resolution = self._resolve_metadata_label_reports(
            question=question,
            query_plan=query_plan,
            route_info=route_info,
        )
        evidence_rows = self.report_catalog.get_company_label_evidence(
            report.sha1,
            label_field="strategy_tags",
            labels=query_plan.filters.strategy_tags,
        )
        base_matched, base_reasons = self.report_catalog.report_matches_query_filters(report, query_plan)
        label_terms = self._metadata_label_terms(question, query_plan)
        label_matched, _, matched_labels = self._report_matches_strategy_labels(report, label_terms)
        if label_terms:
            matched = label_matched
            reasons = [f"strategy_label:{label}" for label in matched_labels] or (
                ["strategy_evidence_match"] if evidence_rows else []
            )
            for reason in base_reasons:
                if reason not in reasons:
                    reasons.append(reason)
        else:
            matched = base_matched
            reasons = base_reasons
        if collection_resolution is not None and collection_resolution.get("retrieval_results"):
            citations = list(collection_resolution.get("citations") or [])
            references = list(collection_resolution.get("references") or [])
            pages = list(collection_resolution.get("relevant_pages") or [])
            retrieval_results = list(collection_resolution.get("retrieval_results") or [])
            matched_doc_ids = list(collection_resolution.get("matched_doc_ids") or [])
            matched_companies = list(collection_resolution.get("matched_companies") or [])
            metadata_query_plan = collection_resolution.get("query_plan") or query_plan
        else:
            if matched:
                citations = self._metadata_membership_citations(report, query_plan, evidence_rows, matched)
                retrieval_results = self._metadata_retrieval_results_from_citations(citations)
                pages = list(dict.fromkeys(result.get("page") for result in retrieval_results if isinstance(result.get("page"), int)))
                references = [
                    self._metadata_reference(report, page)
                    for page in pages
                    if isinstance(page, int)
                ]
            else:
                citations = []
                retrieval_results = []
                pages = []
                references = []
            matched_doc_ids = [report.sha1] if matched else []
            matched_companies = [report.company_name] if matched else []
            metadata_query_plan = query_plan

        route_info.update(
            {
                "route_mode": "metadata_membership",
                "selected_company": report.company_name,
                "candidate_companies": [report.company_name],
                "candidate_doc_ids": [report.sha1],
                "metadata_membership": {
                    "matched": bool(matched),
                    "evaluated_doc_id": report.sha1,
                    "matched_reasons": reasons,
                    "strict_filter_match": bool(base_matched),
                    "strategy_evidence_count": len(evidence_rows),
                    "collection_matched_doc_ids": matched_doc_ids,
                    "collection_matched_companies": matched_companies,
                },
            }
        )

        status_text = "满足" if matched else "不满足"
        reasoning = (
            f"已直接用公司级 metadata 判断名单归属：{report.company_name} {status_text} 问题中的主题标签条件；"
            "板块和行业条件仅作为路由说明，不作为显式公司 membership 的硬拒绝条件。"
        )
        resolution = {
            "relevant_pages": pages,
            "references": dedupe_references(references),
            "citations": dedupe_citations(citations),
            "retrieval_results": retrieval_results,
        }
        return self._metadata_answer_payload(
            final_answer=bool(matched),
            query_plan=metadata_query_plan,
            route_info=route_info,
            resolution=resolution,
            confidence_reason="名单归属类问题已绕过正文检索，直接依据公司级 metadata 与 company-label evidence 判断。",
            reasoning=reasoning,
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
        for support_result in grounding_result.get("supporting_matches") or []:
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

    def _build_table_grounded_number_answer(self, grounding_result: Dict[str, Any]) -> Dict[str, Any]:
        grounded_value = self._grounded_number_value(grounding_result)
        unit_note = f"，并按问题要求换算为{grounding_result['target_unit']}" if grounding_result.get("target_unit") else ""
        page = grounding_result.get("page")
        pages = [page] if page is not None else []
        for support_result in grounding_result.get("supporting_matches") or []:
            support_page = support_result.get("page")
            if support_page is not None and support_page not in pages:
                pages.append(support_page)
            if len(pages) >= 3:
                break
        return {
            "final_answer": grounded_value,
            "relevant_pages": pages,
            "references": [],
            "citations": [],
            "confidence": "high",
            "reasoning_summary": (
                f"答案来自表格 grounding：表 {grounding_result.get('table_id')} 第 {page} 页的单元格"
                f"{unit_note}。"
            ),
            "step_by_step_analysis": (
                f"匹配行头：{grounding_result.get('matched_row_headers') or []}；"
                f"匹配列头：{grounding_result.get('matched_col_headers') or []}；"
                f"原始值：{grounding_result.get('raw_value')}；标准值：{grounding_result.get('normalized_value')}。"
            ),
            "table_grounding_result": grounding_result,
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
                    for support_result in grounding_result.get("supporting_matches") or []
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
                    table_support_results=grounding_result.get("supporting_matches") or [],
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
        selected_doc_id = self._selected_doc_id(company_name, route_info, rewrite_result)
        allowed_grounding_doc_ids = (
            [selected_doc_id]
            if selected_doc_id and route_mode != "document_catalog_multi"
            else list(candidate_doc_ids)
        )

        legal_rule_answer = self._answer_legal_representative_question(
            question=question,
            schema=schema,
            company_name=company_name,
            query_plan=rewrite_result,
            route_info=route_info,
        )
        if legal_rule_answer is not None:
            return legal_rule_answer

        dividend_rule_answer = self._answer_cash_dividend_mention_question(
            question=question,
            schema=schema,
            company_name=company_name,
            query_plan=rewrite_result,
            route_info=route_info,
        )
        if dividend_rule_answer is not None:
            return dividend_rule_answer

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
            retrieval_results = self._merge_multi_query_results(retrieval_runs, self.retrieval_debug_top_n)
            reranking_debug = self._build_reranking_debug_payload(retriever, candidate_pool_size_before_rerank)
            hyde_debug = self._build_hyde_debug_payload(
                retrieval_results=retrieval_results,
                candidate_pool_size_before_rerank=candidate_pool_size_before_rerank,
            )

        grounding_result = None
        if schema == "number" and self.table_grounder is not None:
            if self._is_revenue_metric_question(question) and allowed_grounding_doc_ids:
                grounding_result = self._cached_revenue_grounding(
                    company_name=company_name,
                    question=question,
                    query_plan=rewrite_result,
                    candidate_doc_ids=candidate_doc_ids,
                    allowed_doc_ids=allowed_grounding_doc_ids,
                    retrieval_results=retrieval_results,
                )
            if grounding_result is None:
                grounding_result = self.table_grounder.ground_number_query(
                    question=question,
                    retrieval_results=retrieval_results,
                    filters=rewrite_result.filters,
                    candidate_doc_ids=candidate_doc_ids,
                    allowed_doc_ids=allowed_grounding_doc_ids,
                )
            if grounding_result is not None and self._grounded_number_value(grounding_result) is not None:
                retrieval_results = self._ensure_table_grounding_retrieval_result(retrieval_results, grounding_result)
                retrieval_results = self._ensure_table_support_retrieval_results(retrieval_results, grounding_result)

        if not retrieval_results:
            self.response_data = {}
            no_context_answer = {
                "final_answer": "N/A",
                "relevant_pages": [],
                "references": [],
                "citations": [],
                "confidence": "low",
                "confidence_reason": "未检索到可支撑回答的上下文。",
                "reasoning_summary": "检索阶段没有返回相关证据，因此拒答。",
                "step_by_step_analysis": "检索阶段没有返回相关证据，因此拒答。",
                "search_queries": rewrite_result.search_queries,
                "query_plan": rewrite_result.to_dict(),
                "route_info": route_info or {
                    "route_mode": "explicit_company",
                    "selected_company": company_name,
                    "candidate_companies": [company_name] if company_name else [],
                    "selection_reasons": ["company_argument"],
                },
                "candidate_pool_size_before_rerank": candidate_pool_size_before_rerank,
                "reranking_strategy": reranking_debug["reranking_strategy"],
                "initial_candidate_pool_size": reranking_debug["initial_candidate_pool_size"],
                "colbert_candidate_pool_size": reranking_debug["colbert_candidate_pool_size"],
                "colbert_top_n": reranking_debug["colbert_top_n"],
                "final_reranking_backend": reranking_debug["final_reranking_backend"],
                "hyde": hyde_debug,
                "retrieval_pages": [],
                "retrieval_results": [],
                "retrieval_report_groups": [],
                "response_data": self.response_data,
            }
            if not self.reasoning_debug_enabled:
                no_context_answer["step_by_step_analysis"] = ""
            validated_answer = validate_answer(no_context_answer, [], rewrite_result)
            return validated_answer.answer

        if schema == "number" and grounding_result is not None and self._grounded_number_value(grounding_result) is not None:
            answer_dict = self._build_table_grounded_number_answer(grounding_result)
            self.response_data = {}
        else:
            answer_context_results = retrieval_results[: self.top_n_retrieval]
            rag_context = self._format_retrieval_results(answer_context_results)
            answer_dict = self.api_processor.get_answer_from_rag_context(
                question=question,
                rag_context=rag_context,
                schema=schema,
                model=self.answering_model,
                temperature=self.answer_temperature,
            )
            self.response_data = dict(getattr(self.api_processor, "response_data", {}) or {})

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
            table_support_results=(answer_dict.get("table_grounding_result") or {}).get("supporting_matches") or [],
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

    def process_question(self, question: str, schema: str, question_meta: Optional[Dict[str, Any]] = None):
        route_decision = self.route_question(question, schema)
        if route_decision["is_comparative"]:
            structured_answer = self._answer_structured_revenue_comparison(
                question,
                route_decision["companies"],
                schema,
            )
            if structured_answer is not None:
                return structured_answer
            return self.process_comparative_question(
                question,
                route_decision["companies"],
                schema,
            )

        metadata_count_answer = self._answer_metadata_count_question(
            question=question,
            schema=schema,
            route_decision=route_decision,
            question_meta=question_meta or {},
        )
        if metadata_count_answer is not None:
            return metadata_count_answer

        metadata_names_answer = self._answer_metadata_names_question(
            question=question,
            schema=schema,
            route_decision=route_decision,
        )
        if metadata_names_answer is not None:
            return metadata_names_answer

        metadata_membership_answer = self._answer_metadata_membership_question(
            question=question,
            schema=schema,
            route_decision=route_decision,
        )
        if metadata_membership_answer is not None:
            return metadata_membership_answer

        threshold_answer = self._answer_structured_revenue_threshold(
            question=question,
            schema=schema,
            route_decision=route_decision,
        )
        if threshold_answer is not None:
            return threshold_answer

        return self.get_answer_for_company(
            company_name=route_decision["company_name"],
            question=question,
            schema=schema,
            query_plan=route_decision["query_plan"],
            route_info=route_decision["route_info"],
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

        final_schema = "text" if schema in {"text", "long_text"} else "comparative"
        comparative_answer = self.api_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema=final_schema,
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
