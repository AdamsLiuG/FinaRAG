import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.embedding_backend import EmbeddingBackend, BGEM3SparseEmbeddingBackend
from src.retrieval_filters import RetrievalFilters, apply_retrieval_filters, build_result_metadata
from src.reranking import LLMReranker, FlagEmbeddingReranker
from src.text_normalization import tokenize_for_bm25

_log = logging.getLogger(__name__)
_VALID_PARENT_RETRIEVAL_MODES = {"child", "page", "block"}


def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        if max_score > 0:
            return [1.0 for _ in scores]
        return [0.0 for _ in scores]

    score_range = max_score - min_score
    return [round((score - min_score) / score_range, 4) for score in scores]


def _normalize_parent_retrieval_mode(parent_retrieval_mode: Optional[str]) -> str:
    mode = (parent_retrieval_mode or "child").strip().lower()
    if mode not in _VALID_PARENT_RETRIEVAL_MODES:
        raise ValueError(
            f"Unsupported parent retrieval mode '{parent_retrieval_mode}'. "
            f"Expected one of: {sorted(_VALID_PARENT_RETRIEVAL_MODES)}."
        )
    return mode


def _make_result_key(result: Dict) -> Tuple:
    metadata = result.get("metadata") or {}
    source_name = metadata.get("sha1_name")
    result_scope = result.get("result_scope") or metadata.get("node_type") or "child"
    chunk_id = result.get("chunk_id") or metadata.get("chunk_id")

    if result_scope == "page":
        return source_name, result_scope, result.get("page")
    if chunk_id is not None:
        return source_name, result_scope, chunk_id
    return source_name, result_scope, result.get("page"), result.get("text")


def _reciprocal_rank_fusion_score(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


def _dedupe_preserve_order(values: List) -> List:
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _build_retrieval_result(
    document_meta: Dict,
    chunk: Dict,
    text: str,
    page: int,
    score: float,
    source_name: str,
    *,
    result_scope: Optional[str] = None,
) -> Dict:
    metadata = build_result_metadata(document_meta, chunk)
    scope = result_scope or ("parent" if metadata.get("node_type") == "parent" else "child")
    return {
        "distance": round(float(score), 4),
        "page": page,
        "text": text,
        "metadata": metadata,
        "chunk_id": metadata.get("chunk_id"),
        "chunk_type": metadata.get("chunk_type"),
        "section_title": metadata.get("section_title"),
        "table_id": metadata.get("table_id"),
        "retrieval_sources": [source_name],
        "matched_child_chunk_ids": list(chunk.get("matched_child_chunk_ids") or []),
        "result_scope": scope,
    }


class _DocumentBackedRetriever:
    def __init__(self, documents_dir: Path):
        self.documents_dir = documents_dir
        self.documents = self._load_documents()

    def _load_documents(self) -> List[Dict]:
        loaded_documents = []
        for document_path in self.documents_dir.glob("*.json"):
            try:
                with open(document_path, "r", encoding="utf-8") as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue

            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue

            loaded_documents.append(
                {
                    "name": document_path.stem,
                    "path": document_path,
                    "document": document,
                }
            )
        return loaded_documents

    def _get_document_by_company_name(self, company_name: str) -> Dict:
        for entry in self.documents:
            document = entry["document"]
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                return entry

        raise ValueError(f"No report found with '{company_name}' company name.")

    def _candidate_document_entries(self, company_name: str, candidate_doc_ids: Optional[List[str]] = None) -> List[Dict]:
        if candidate_doc_ids:
            candidate_lookup = {str(doc_id) for doc_id in candidate_doc_ids}
            matched = []
            for entry in self.documents:
                metainfo = entry["document"].get("metainfo") or {}
                doc_id = str(metainfo.get("sha1_name") or metainfo.get("doc_id") or entry["name"])
                if doc_id in candidate_lookup or entry["name"] in candidate_lookup:
                    matched.append(entry)
            if matched:
                return matched
        return [self._get_document_by_company_name(company_name)]

    @staticmethod
    def _page_lookup(document: Dict) -> Dict[int, Dict]:
        return {page["page"]: page for page in document.get("content", {}).get("pages", [])}

    def _parent_chunk_lookup(self, document: Dict, company_name: str) -> Dict[int, Dict]:
        content = document.get("content") or {}
        parent_chunks = content.get("parent_chunks")
        if not isinstance(parent_chunks, list) or not parent_chunks:
            raise ValueError(
                f"Block Parent-Child retrieval requested for '{company_name}', but the chunked report does not "
                "contain `content.parent_chunks`. Please re-run `process-reports` to rebuild the dataset."
            )

        lookup = {}
        for parent_chunk in parent_chunks:
            parent_chunk_id = parent_chunk.get("chunk_id", parent_chunk.get("id"))
            if parent_chunk_id is None:
                raise ValueError(
                    f"Block Parent-Child retrieval requested for '{company_name}', but a parent chunk is missing "
                    "`chunk_id`. Please re-run `process-reports` to rebuild the dataset."
                )
            lookup[parent_chunk_id] = parent_chunk
        return lookup

    @staticmethod
    def _child_chunk_id(chunk: Dict) -> Optional[int]:
        return chunk.get("chunk_id", chunk.get("id"))

    def _build_child_results(self, document: Dict, ranked_hits: List[Tuple[float, int]], source_name: str, top_n: int) -> List[Dict]:
        chunks = document.get("content", {}).get("chunks", [])
        results = []
        for score, index in ranked_hits[:top_n]:
            chunk = chunks[index]
            results.append(
                _build_retrieval_result(
                    document["metainfo"],
                    chunk,
                    chunk["text"],
                    chunk["page"],
                    score,
                    source_name,
                    result_scope="child",
                )
            )
        return results

    def _build_page_results(self, document: Dict, ranked_hits: List[Tuple[float, int]], source_name: str, top_n: int) -> List[Dict]:
        chunks = document.get("content", {}).get("chunks", [])
        pages = self._page_lookup(document)
        results = []
        seen_pages = set()

        for score, index in ranked_hits:
            chunk = chunks[index]
            parent_page = pages.get(chunk["page"])
            if parent_page is None:
                raise ValueError(f"Missing page {chunk['page']} in chunked report for source {document['metainfo'].get('sha1_name')}.")
            if parent_page["page"] in seen_pages:
                continue

            seen_pages.add(parent_page["page"])
            results.append(
                _build_retrieval_result(
                    document["metainfo"],
                    chunk,
                    parent_page["text"],
                    parent_page["page"],
                    score,
                    source_name,
                    result_scope="page",
                )
            )
            if len(results) >= top_n:
                break

        return results

    def _build_block_results(
        self,
        document: Dict,
        company_name: str,
        ranked_hits: List[Tuple[float, int]],
        source_name: str,
        top_n: int,
    ) -> List[Dict]:
        chunks = document.get("content", {}).get("chunks", [])
        if any("parent_chunk_id" not in chunk for chunk in chunks):
            raise ValueError(
                f"Block Parent-Child retrieval requested for '{company_name}', but child chunks are missing "
                "`parent_chunk_id`. Please re-run `process-reports` to rebuild the dataset."
            )

        parent_lookup = self._parent_chunk_lookup(document, company_name)
        aggregated: Dict[int, Dict] = {}
        ordered_parent_ids: List[int] = []

        for score, index in ranked_hits:
            child_chunk = chunks[index]
            parent_chunk_id = child_chunk.get("parent_chunk_id")
            parent_chunk = parent_lookup.get(parent_chunk_id)
            if parent_chunk is None:
                raise ValueError(
                    f"Child chunk references missing parent_chunk_id '{parent_chunk_id}' for '{company_name}'. "
                    "Please re-run `process-reports` to rebuild the dataset."
                )

            child_chunk_id = self._child_chunk_id(child_chunk)
            existing = aggregated.get(parent_chunk_id)

            if existing is None:
                if len(ordered_parent_ids) >= top_n:
                    break

                parent_payload = dict(parent_chunk)
                parent_payload["matched_child_chunk_ids"] = [child_chunk_id] if child_chunk_id is not None else []
                aggregated[parent_chunk_id] = _build_retrieval_result(
                    document["metainfo"],
                    parent_payload,
                    parent_chunk["text"],
                    parent_chunk["page"],
                    score,
                    source_name,
                    result_scope="parent",
                )
                ordered_parent_ids.append(parent_chunk_id)
                continue

            if child_chunk_id is not None:
                merged_child_ids = list(existing.get("matched_child_chunk_ids", []))
                if child_chunk_id not in merged_child_ids:
                    merged_child_ids.append(child_chunk_id)
                existing["matched_child_chunk_ids"] = merged_child_ids

            if score > existing["distance"]:
                existing["distance"] = round(float(score), 4)

        return [aggregated[parent_chunk_id] for parent_chunk_id in ordered_parent_ids]

    def _finalize_results(
        self,
        *,
        company_name: str,
        document: Dict,
        ranked_hits: List[Tuple[float, int]],
        source_name: str,
        top_n: int,
        parent_retrieval_mode: Optional[str],
        filters: Optional[RetrievalFilters],
    ) -> List[Dict]:
        mode = _normalize_parent_retrieval_mode(parent_retrieval_mode)

        if mode == "page":
            results = self._build_page_results(document, ranked_hits, source_name, top_n)
        elif mode == "block":
            results = self._build_block_results(document, company_name, ranked_hits, source_name, top_n)
        else:
            results = self._build_child_results(document, ranked_hits, source_name, top_n)

        return apply_retrieval_filters(results, filters)


class BM25Retriever(_DocumentBackedRetriever):
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        super().__init__(documents_dir)

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        top_n: int = 3,
        parent_retrieval_mode: str = "child",
        filters: Optional[RetrievalFilters] = None,
        candidate_doc_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        aggregated_results: List[Dict] = []
        for document_entry in self._candidate_document_entries(company_name, candidate_doc_ids):
            document = document_entry["document"]
            bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1_name']}.pkl"
            if not bm25_path.exists():
                raise ValueError(f"No BM25 index found for '{company_name}' at {bm25_path}.")
            with open(bm25_path, "rb") as f:
                bm25_index = pickle.load(f)

            scores = bm25_index.get_scores(tokenize_for_bm25(query))
            ranked_hits = sorted(
                ((round(float(score), 4), index) for index, score in enumerate(scores)),
                key=lambda item: item[0],
                reverse=True,
            )
            aggregated_results.extend(
                self._finalize_results(
                    company_name=company_name,
                    document=document,
                    ranked_hits=ranked_hits,
                    source_name="bm25",
                    top_n=max(top_n, 3),
                    parent_retrieval_mode=parent_retrieval_mode,
                    filters=filters,
                )
            )

        aggregated_results.sort(key=lambda item: item.get("ranking_score", item.get("distance", 0.0)), reverse=True)
        return aggregated_results[:top_n]


class BGEM3SparseRetriever(_DocumentBackedRetriever):
    def __init__(self, sparse_db_dir: Path, documents_dir: Path):
        self.sparse_db_dir = sparse_db_dir
        self.embedding_backend = BGEM3SparseEmbeddingBackend()
        super().__init__(documents_dir)

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        top_n: int = 3,
        parent_retrieval_mode: str = "child",
        filters: Optional[RetrievalFilters] = None,
        candidate_doc_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        aggregated_results: List[Dict] = []
        query_weights = self.embedding_backend.encode_query(query)
        for document_entry in self._candidate_document_entries(company_name, candidate_doc_ids):
            document = document_entry["document"]
            sparse_path = self.sparse_db_dir / f"{document['metainfo']['sha1_name']}.pkl"
            if not sparse_path.exists():
                raise ValueError(f"No sparse lexical index found for '{company_name}' at {sparse_path}.")

            with open(sparse_path, "rb") as f:
                sparse_index = pickle.load(f)

            lexical_weights = sparse_index.get("lexical_weights")
            if lexical_weights is None:
                raise ValueError(f"Sparse lexical index at {sparse_path} is missing `lexical_weights`.")

            scores = self.embedding_backend.score_query_against_documents(query_weights, lexical_weights)
            ranked_hits = sorted(
                ((round(float(score), 4), index) for index, score in enumerate(scores)),
                key=lambda item: item[0],
                reverse=True,
            )
            aggregated_results.extend(
                self._finalize_results(
                    company_name=company_name,
                    document=document,
                    ranked_hits=ranked_hits,
                    source_name="sparse",
                    top_n=max(top_n, 3),
                    parent_retrieval_mode=parent_retrieval_mode,
                    filters=filters,
                )
            )

        aggregated_results.sort(key=lambda item: item.get("ranking_score", item.get("distance", 0.0)), reverse=True)
        return aggregated_results[:top_n]


class VectorRetriever(_DocumentBackedRetriever):
    def __init__(
        self,
        vector_db_dir: Path,
        documents_dir: Path,
        vector_search_k: Optional[int] = None,
        ivf_nprobe: int = 8,
        hnsw_ef_search: int = 64,
    ):
        self.vector_db_dir = vector_db_dir
        self.vector_search_k = int(vector_search_k) if vector_search_k else None
        self.ivf_nprobe = max(1, int(ivf_nprobe))
        self.hnsw_ef_search = max(1, int(hnsw_ef_search))
        self.embedding_backend = EmbeddingBackend()
        super().__init__(documents_dir)
        self.all_dbs = self._load_dbs()

    def _configure_index(self, vector_db):
        if hasattr(vector_db, "hnsw"):
            vector_db.hnsw.efSearch = self.hnsw_ef_search
        if hasattr(vector_db, "nprobe"):
            nlist = getattr(vector_db, "nlist", None)
            vector_db.nprobe = min(self.ivf_nprobe, nlist) if nlist is not None else self.ivf_nprobe
        return vector_db

    def _resolve_search_k(self, top_n: int, num_chunks: int, parent_retrieval_mode: Optional[str]) -> int:
        if num_chunks <= 0:
            return 0
        requested_top_n = max(top_n, 3)
        if self.vector_search_k is not None:
            return min(max(self.vector_search_k, requested_top_n), num_chunks)
        mode = _normalize_parent_retrieval_mode(parent_retrieval_mode)
        if mode == "child":
            return min(max(requested_top_n * 4, 16), num_chunks)
        return min(max(requested_top_n * 8, 32), num_chunks)

    def _load_dbs(self) -> List[Dict]:
        all_dbs = []
        vector_db_files = {db_path.stem: db_path for db_path in self.vector_db_dir.glob("*.faiss")}

        for entry in self.documents:
            stem = entry["name"]
            if stem not in vector_db_files:
                _log.warning(f"No matching vector DB found for document {stem}.json")
                continue

            try:
                vector_db = self._configure_index(faiss.read_index(str(vector_db_files[stem])))
            except Exception as e:
                _log.error(f"Error reading vector DB for {stem}.json: {e}")
                continue

            all_dbs.append(
                {
                    "name": stem,
                    "vector_db": vector_db,
                    "document": entry["document"],
                }
            )
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        embedding_backend = EmbeddingBackend()
        embedding1, embedding2 = embedding_backend.embed_texts([str1, str2])
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        llm_reranking_sample_size: int = None,
        top_n: int = 3,
        parent_retrieval_mode: str = "child",
        filters: Optional[RetrievalFilters] = None,
        candidate_doc_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        candidate_lookup = {str(doc_id) for doc_id in candidate_doc_ids or []}
        candidate_reports = []
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                _log.error(f"Report '{report.get('name')}' is missing 'metainfo'!")
                raise ValueError(f"Report '{report.get('name')}' is missing 'metainfo'!")
            doc_id = str(metainfo.get("sha1_name") or metainfo.get("doc_id") or report.get("name"))
            if candidate_lookup:
                if doc_id in candidate_lookup or report.get("name") in candidate_lookup:
                    candidate_reports.append(report)
            elif metainfo.get("company_name") == company_name:
                candidate_reports.append(report)

        if not candidate_reports:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")

        embedding_array = self.embedding_backend.embed_query(query).reshape(1, -1)
        aggregated_results: List[Dict] = []
        for target_report in candidate_reports:
            document = target_report["document"]
            chunks = document.get("content", {}).get("chunks", [])
            if not chunks:
                continue

            actual_top_n = max(top_n, 3)
            actual_search_k = self._resolve_search_k(
                top_n=top_n,
                num_chunks=len(chunks),
                parent_retrieval_mode=parent_retrieval_mode,
            )
            distances, indices = target_report["vector_db"].search(x=embedding_array, k=actual_search_k)
            ranked_hits = [
                (round(float(distance), 4), int(index))
                for distance, index in zip(distances[0], indices[0])
                if int(index) >= 0
            ]
            aggregated_results.extend(
                self._finalize_results(
                    company_name=company_name,
                    document=document,
                    ranked_hits=ranked_hits,
                    source_name="vector",
                    top_n=actual_top_n,
                    parent_retrieval_mode=parent_retrieval_mode,
                    filters=filters,
                )
            )

        aggregated_results.sort(key=lambda item: item.get("ranking_score", item.get("distance", 0.0)), reverse=True)
        return aggregated_results[:top_n]

    def retrieve_all(self, company_name: str, filters: Optional[RetrievalFilters] = None, candidate_doc_ids: Optional[List[str]] = None) -> List[Dict]:
        candidate_lookup = {str(doc_id) for doc_id in candidate_doc_ids or []}
        target_reports = []
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            doc_id = str(metainfo.get("sha1_name") or metainfo.get("doc_id") or report.get("name"))
            if candidate_lookup:
                if doc_id in candidate_lookup or report.get("name") in candidate_lookup:
                    target_reports.append(report)
            elif metainfo.get("company_name") == company_name:
                target_reports.append(report)

        if not target_reports:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")

        all_pages = []
        for target_report in target_reports:
            document = target_report["document"]
            for page in sorted(document.get("content", {}).get("pages", []), key=lambda p: p["page"]):
                result = _build_retrieval_result(
                    document["metainfo"],
                    {"page": page["page"], "chunk_type": "page", "node_type": "page"},
                    page["text"],
                    page["page"],
                    0.5,
                    "vector_full_context",
                    result_scope="page",
                )
                all_pages.append(result)

        return apply_retrieval_filters(all_pages, filters)


class HybridRetriever:
    def __init__(
        self,
        documents_dir: Path,
        vector_db_dir: Optional[Path] = None,
        bm25_db_dir: Optional[Path] = None,
        sparse_db_dir: Optional[Path] = None,
        use_vector_dbs: bool = True,
        use_bm25_db: bool = True,
        use_sparse_lexical_db: bool = False,
        vector_search_k: Optional[int] = None,
        vector_ivf_nprobe: int = 8,
        vector_hnsw_ef_search: int = 64,
        provider: str = "qwen",
        model: str = None,
    ):
        if not use_vector_dbs and not use_bm25_db and not use_sparse_lexical_db:
            raise ValueError("At least one retrieval backend must be enabled.")

        self.use_vector_dbs = use_vector_dbs
        self.use_bm25_db = use_bm25_db
        self.use_sparse_lexical_db = use_sparse_lexical_db
        self.fusion_method = os.getenv("HYBRID_RETRIEVAL_FUSION", "rrf").strip().lower()
        self.rrf_k = int(os.getenv("HYBRID_RETRIEVAL_RRF_K", "60"))
        self.vector_retriever = (
            VectorRetriever(
                vector_db_dir,
                documents_dir,
                vector_search_k=vector_search_k,
                ivf_nprobe=vector_ivf_nprobe,
                hnsw_ef_search=vector_hnsw_ef_search,
            )
            if use_vector_dbs
            else None
        )
        self.bm25_retriever = BM25Retriever(bm25_db_dir, documents_dir) if use_bm25_db else None
        self.sparse_retriever = BGEM3SparseRetriever(sparse_db_dir, documents_dir) if use_sparse_lexical_db else None
        backend = os.getenv("RERANKING_BACKEND", "llm_prompt").lower()
        self.reranker = (
            FlagEmbeddingReranker()
            if backend == "flag_embedding"
            else LLMReranker(provider=provider, model=model)
        )

    def _merge_retrieval_results(
        self,
        retrieval_results_by_source: Dict[str, List[Dict]],
        top_n: int,
    ) -> List[Dict]:
        score_fields = {
            "vector": "vector_score",
            "bm25": "bm25_score",
            "sparse": "sparse_score",
        }
        candidates: Dict[Tuple, Dict] = {}
        active_backends = max(1, len(retrieval_results_by_source))

        for source_name, results in retrieval_results_by_source.items():
            normalized_scores = _normalize_scores([result["distance"] for result in results])
            score_field = score_fields[source_name]

            for rank, (result, normalized_score) in enumerate(zip(results, normalized_scores), start=1):
                key = _make_result_key(result)
                existing = candidates.get(key)
                if existing is None:
                    candidates[key] = {
                        **result,
                        "vector_score": 0.0,
                        "bm25_score": 0.0,
                        "sparse_score": 0.0,
                        "rrf_score": 0.0,
                        "retrieval_sources": list(result.get("retrieval_sources", [])),
                        "matched_child_chunk_ids": list(result.get("matched_child_chunk_ids", [])),
                    }
                    existing = candidates[key]

                existing[score_field] = max(existing[score_field], normalized_score)
                existing["rrf_score"] += _reciprocal_rank_fusion_score(rank, self.rrf_k)
                existing["retrieval_sources"] = _dedupe_preserve_order(
                    list(existing.get("retrieval_sources", [])) + list(result.get("retrieval_sources", []))
                )
                existing["matched_child_chunk_ids"] = _dedupe_preserve_order(
                    list(existing.get("matched_child_chunk_ids", [])) + list(result.get("matched_child_chunk_ids", []))
                )

        for item in candidates.values():
            average_score = (
                item["vector_score"] + item["bm25_score"] + item["sparse_score"]
            ) / active_backends
            item["average_score"] = round(float(average_score), 4)

            if self.fusion_method == "average":
                item["distance"] = item["average_score"]
            else:
                item["distance"] = round(float(item["rrf_score"]), 6)
            item["ranking_score"] = item["distance"]

        merged_results = list(candidates.values())
        merged_results.sort(key=lambda item: item["distance"], reverse=True)
        return merged_results[:top_n]

    def retrieve_candidates_by_company_name(
        self,
        company_name: str,
        query: str,
        top_n: int = 28,
        parent_retrieval_mode: str = "child",
        filters: Optional[RetrievalFilters] = None,
        candidate_doc_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        vector_results: List[Dict] = []
        bm25_results: List[Dict] = []
        sparse_results: List[Dict] = []

        if self.vector_retriever is not None:
            vector_results = self.vector_retriever.retrieve_by_company_name(
                company_name=company_name,
                query=query,
                top_n=top_n,
                parent_retrieval_mode=parent_retrieval_mode,
                filters=filters,
                candidate_doc_ids=candidate_doc_ids,
            )

        if self.bm25_retriever is not None:
            bm25_results = self.bm25_retriever.retrieve_by_company_name(
                company_name=company_name,
                query=query,
                top_n=top_n,
                parent_retrieval_mode=parent_retrieval_mode,
                filters=filters,
                candidate_doc_ids=candidate_doc_ids,
            )

        if self.sparse_retriever is not None:
            sparse_results = self.sparse_retriever.retrieve_by_company_name(
                company_name=company_name,
                query=query,
                top_n=top_n,
                parent_retrieval_mode=parent_retrieval_mode,
                filters=filters,
                candidate_doc_ids=candidate_doc_ids,
            )

        active_results = {}
        if vector_results:
            active_results["vector"] = vector_results
        if bm25_results:
            active_results["bm25"] = bm25_results
        if sparse_results:
            active_results["sparse"] = sparse_results

        if len(active_results) == 1:
            return next(iter(active_results.values()))[:top_n]
        if not active_results:
            return []

        return self._merge_retrieval_results(active_results, top_n=top_n)

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        llm_reranking_sample_size: int = 28,
        documents_batch_size: int = 2,
        top_n: int = 6,
        llm_weight: float = 0.7,
        parent_retrieval_mode: str = "child",
        filters: Optional[RetrievalFilters] = None,
        candidate_doc_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        candidate_results = self.retrieve_candidates_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            parent_retrieval_mode=parent_retrieval_mode,
            filters=filters,
            candidate_doc_ids=candidate_doc_ids,
        )

        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=candidate_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight,
        )

        return reranked_results[:top_n]
