import json
import logging
from typing import List, Tuple, Dict, Optional
import pickle
from pathlib import Path
import faiss
import numpy as np
import os

from src.embedding_backend import EmbeddingBackend, BGEM3SparseEmbeddingBackend
from src.reranking import LLMReranker, FlagEmbeddingReranker

_log = logging.getLogger(__name__)


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


def _make_result_key(result: Dict) -> Tuple[int, str]:
    return result["page"], result["text"]


def _reciprocal_rank_fusion_score(rank: int, k: int) -> float:
    return 1.0 / (k + rank)

class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir
        self.documents = self._load_documents()

    def _load_documents(self) -> List[Dict]:
        loaded_documents = []
        for document_path in self.documents_dir.glob("*.json"):
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue

            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue

            loaded_documents.append({
                "name": document_path.stem,
                "path": document_path,
                "document": document,
            })
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
        
    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        document_entry = self._get_document_by_company_name(company_name)
        document = document_entry["document"]
            
        # Load corresponding BM25 index
        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1_name']}.pkl"
        if not bm25_path.exists():
            raise ValueError(f"No BM25 index found for '{company_name}' at {bm25_path}.")
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)
            
        # Get the document content and BM25 index
        document = document
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        # Get BM25 scores for the query
        tokenized_query = query.split()
        scores = bm25_index.get_scores(tokenized_query)
        
        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]
        
        retrieval_results = []
        seen_pages = set()
        
        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": score,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": score,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
        
        return retrieval_results


class BGEM3SparseRetriever:
    def __init__(self, sparse_db_dir: Path, documents_dir: Path):
        self.sparse_db_dir = sparse_db_dir
        self.documents_dir = documents_dir
        self.embedding_backend = BGEM3SparseEmbeddingBackend()
        self.documents = self._load_documents()

    def _load_documents(self) -> List[Dict]:
        loaded_documents = []
        for document_path in self.documents_dir.glob("*.json"):
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue

            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue

            loaded_documents.append({
                "name": document_path.stem,
                "path": document_path,
                "document": document,
            })
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

    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        document_entry = self._get_document_by_company_name(company_name)
        document = document_entry["document"]

        sparse_path = self.sparse_db_dir / f"{document['metainfo']['sha1_name']}.pkl"
        if not sparse_path.exists():
            raise ValueError(f"No sparse lexical index found for '{company_name}' at {sparse_path}.")

        with open(sparse_path, 'rb') as f:
            sparse_index = pickle.load(f)

        lexical_weights = sparse_index.get("lexical_weights")
        if lexical_weights is None:
            raise ValueError(f"Sparse lexical index at {sparse_path} is missing `lexical_weights`.")

        query_weights = self.embedding_backend.encode_query(query)
        scores = self.embedding_backend.score_query_against_documents(query_weights, lexical_weights)

        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]

        retrieval_results = []
        seen_pages = set()

        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])

            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": score,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": score,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)

        return retrieval_results


class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()
        self.embedding_backend = EmbeddingBackend()

    def _load_dbs(self):
        all_dbs = []
        # Get list of JSON document paths
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        vector_db_files = {db_path.stem: db_path for db_path in self.vector_db_dir.glob('*.faiss')}
        
        for document_path in all_documents_paths:
            stem = document_path.stem
            if stem not in vector_db_files:
                _log.warning(f"No matching vector DB found for document {document_path.name}")
                continue
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue
            
            # Validate that the document meets the expected schema
            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
                continue
            
            try:
                vector_db = faiss.read_index(str(vector_db_files[stem]))
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue
                
            report = {
                "name": stem,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        embedding_backend = EmbeddingBackend()
        embedding1, embedding2 = embedding_backend.embed_texts([str1, str2])
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False) -> List[Tuple[str, float]]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                _log.error(f"Report '{report.get('name')}' is missing 'metainfo'!")
                raise ValueError(f"Report '{report.get('name')}' is missing 'metainfo'!")
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        actual_top_n = min(top_n, len(chunks))
        
        embedding_array = self.embedding_backend.embed_query(query).reshape(1, -1)
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
    
        retrieval_results = []
        seen_pages = set()
        
        for distance, index in zip(distances[0], indices[0]):
            distance = round(float(distance), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": distance,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": distance,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
            
        return retrieval_results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        pages = document["content"]["pages"]
        
        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "distance": 0.5,
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)
            
        return all_pages


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
        provider: str = "qwen",
        model: str = None
    ):
        if not use_vector_dbs and not use_bm25_db and not use_sparse_lexical_db:
            raise ValueError("At least one retrieval backend must be enabled.")

        self.use_vector_dbs = use_vector_dbs
        self.use_bm25_db = use_bm25_db
        self.use_sparse_lexical_db = use_sparse_lexical_db
        self.fusion_method = os.getenv("HYBRID_RETRIEVAL_FUSION", "rrf").strip().lower()
        self.rrf_k = int(os.getenv("HYBRID_RETRIEVAL_RRF_K", "60"))
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir) if use_vector_dbs else None
        self.bm25_retriever = BM25Retriever(bm25_db_dir, documents_dir) if use_bm25_db else None
        self.sparse_retriever = BGEM3SparseRetriever(sparse_db_dir, documents_dir) if use_sparse_lexical_db else None
        backend = os.getenv("RERANKING_BACKEND", "llm_prompt").lower()
        # `model` here is the answering LLM model. It should only be forwarded to
        # the LLM-based reranker. The FlagEmbedding backend must keep using its
        # own reranker model from `RERANKING_MODEL`.
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
        candidates: Dict[Tuple[int, str], Dict] = {}
        active_backends = max(1, len(retrieval_results_by_source))

        for source_name, results in retrieval_results_by_source.items():
            normalized_scores = _normalize_scores([result["distance"] for result in results])
            score_field = score_fields[source_name]

            for rank, (result, normalized_score) in enumerate(zip(results, normalized_scores), start=1):
                key = _make_result_key(result)
                item = candidates.setdefault(
                    key,
                    {
                        **result,
                        "vector_score": 0.0,
                        "bm25_score": 0.0,
                        "sparse_score": 0.0,
                        "rrf_score": 0.0,
                        "retrieval_sources": [],
                    },
                )
                item[score_field] = max(item[score_field], normalized_score)
                item["rrf_score"] += _reciprocal_rank_fusion_score(rank, self.rrf_k)
                if source_name not in item["retrieval_sources"]:
                    item["retrieval_sources"].append(source_name)

        for item in candidates.values():
            average_score = (
                item["vector_score"] + item["bm25_score"] + item["sparse_score"]
            ) / active_backends
            item["average_score"] = round(float(average_score), 4)

            if self.fusion_method == "average":
                item["distance"] = item["average_score"]
            else:
                item["distance"] = round(float(item["rrf_score"]), 6)

        merged_results = list(candidates.values())
        merged_results.sort(key=lambda item: item["distance"], reverse=True)
        return merged_results[:top_n]

    def retrieve_candidates_by_company_name(
        self,
        company_name: str,
        query: str,
        top_n: int = 28,
        return_parent_pages: bool = False,
    ) -> List[Dict]:
        vector_results: List[Dict] = []
        bm25_results: List[Dict] = []
        sparse_results: List[Dict] = []

        if self.vector_retriever is not None:
            vector_results = self.vector_retriever.retrieve_by_company_name(
                company_name=company_name,
                query=query,
                top_n=top_n,
                return_parent_pages=return_parent_pages,
            )

        if self.bm25_retriever is not None:
            bm25_results = self.bm25_retriever.retrieve_by_company_name(
                company_name=company_name,
                query=query,
                top_n=top_n,
                return_parent_pages=return_parent_pages,
            )

        if self.sparse_retriever is not None:
            sparse_results = self.sparse_retriever.retrieve_by_company_name(
                company_name=company_name,
                query=query,
                top_n=top_n,
                return_parent_pages=return_parent_pages,
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
        return_parent_pages: bool = False
    ) -> List[Dict]:
        """
        Retrieve and rerank documents using hybrid approach.
        
        Args:
            company_name: Name of the company to search documents for
            query: Search query
            llm_reranking_sample_size: Number of initial results to retrieve from vector DB
            documents_batch_size: Number of documents to analyze in one LLM prompt
            top_n: Number of final results to return after reranking
            llm_weight: Weight given to LLM scores (0-1)
            return_parent_pages: Whether to return full pages instead of chunks
            
        Returns:
            List of reranked document dictionaries with scores
        """
        candidate_results = self.retrieve_candidates_by_company_name(
            company_name=company_name,
            query=query,
            top_n=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages
        )

        # Rerank results using LLM
        reranked_results = self.reranker.rerank_documents(
            query=query,
            documents=candidate_results,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight
        )
        
        return reranked_results[:top_n]
