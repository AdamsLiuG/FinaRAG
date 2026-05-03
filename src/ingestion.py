import json
import pickle
import re
from typing import List
from pathlib import Path
from tqdm import tqdm

from rank_bm25 import BM25Okapi
import faiss
import numpy as np

from src.embedding_backend import EmbeddingBackend, BGEM3SparseEmbeddingBackend
from src.text_normalization import normalize_text, tokenize_for_bm25


_TAG_ARRAY_FIELDS = (
    "business_tags",
    "factor_tags",
    "chain_position_minor",
    "listing_tags",
    "ownership_tags",
    "status_tags",
    "style_tags",
)

_STRATEGY_TAG_ALIASES = {
    "国产替代": ["国产替代", "国产化", "自主可控", "信创"],
    "数字化转型": ["数字化转型", "数字化", "数智化"],
    "出海": ["出海", "海外", "海外市场", "海外业务", "国际化", "境外"],
    "绿色转型": ["绿色转型", "绿色低碳", "双碳", "碳中和"],
    "人工智能": ["人工智能", "AI", "大模型"],
}


def _vector_text_from_chunk(chunk: dict) -> str:
    return chunk.get("embedding_text") or chunk.get("search_text") or chunk.get("text") or ""


def _lexical_text_from_chunk(chunk: dict) -> str:
    return chunk.get("search_text") or chunk.get("embedding_text") or chunk.get("text") or ""


def _strategy_tag_terms(tag: str) -> List[str]:
    tag_text = str(tag or "").strip()
    if not tag_text:
        return []
    terms = [tag_text] + _STRATEGY_TAG_ALIASES.get(tag_text, [])
    deduped: List[str] = []
    seen = set()
    for term in terms:
        marker = normalize_text(term)
        if marker and marker not in seen:
            seen.add(marker)
            deduped.append(term)
    return deduped


def _strategy_tag_has_literal_evidence(text: str, tag: str) -> bool:
    text = text or ""
    for term in _strategy_tag_terms(tag):
        flags = re.IGNORECASE if term.isascii() else 0
        if re.search(re.escape(term), text, flags):
            return True
    return False


def _chunk_tag_values(chunk: dict) -> List[str]:
    values: List[str] = []
    scalar_fields = (
        "section_name",
        "section_title",
        "report_section",
        "exchange",
        "board",
        "market_type",
        "industry_l1",
        "industry_l2",
        "chain_position_major",
    )
    for field in scalar_fields:
        value = chunk.get(field)
        if value not in (None, ""):
            values.append(str(value))
    for field in _TAG_ARRAY_FIELDS:
        values.extend(str(item) for item in chunk.get(field) or [] if item not in (None, ""))
    for item in chunk.get("strategy_tags") or []:
        if item not in (None, "") and _strategy_tag_has_literal_evidence(chunk.get("text") or "", str(item)):
            values.append(str(item))
    deduped: List[str] = []
    seen = set()
    for value in values:
        marker = normalize_text(value)
        if not marker or marker in seen:
            continue
        seen.add(marker)
        deduped.append(value)
    return deduped


class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks."""
        tokenized_chunks = [tokenize_for_bm25(chunk) for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Process all reports and save individual BM25 indices.
        
        Args:
            all_reports_dir (Path): Directory containing the JSON report files
            output_dir (Path): Directory where to save the BM25 indices
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            # Load the report
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # Extract text chunks and create BM25 index
            text_chunks = [_lexical_text_from_chunk(chunk) for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)
            
            # Save BM25 index
            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
                
        print(f"Processed {len(all_report_paths)} reports")


class SparseLexicalIngestor:
    def __init__(self):
        self.embedding_backend = BGEM3SparseEmbeddingBackend()

    def _get_lexical_weights(self, texts: List[str]) -> List[dict]:
        lexical_weights = self.embedding_backend.encode_texts(texts)
        if not lexical_weights:
            raise ValueError("No sparse lexical weights were produced for the provided text chunks.")
        return lexical_weights

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for sparse lexical retrieval"):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            text_chunks = [_lexical_text_from_chunk(chunk) for chunk in report_data['content']['chunks']]
            lexical_weights = self._get_lexical_weights(text_chunks)

            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump({"lexical_weights": lexical_weights}, f)

        print(f"Processed {len(all_report_paths)} reports")

class VectorDBIngestor:
    def __init__(
        self,
        index_type: str = "flat",
        ivf_nlist: int = 32,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
    ):
        self.embedding_backend = EmbeddingBackend()
        self.index_type = self._normalize_index_type(index_type)
        self.ivf_nlist = max(1, int(ivf_nlist))
        self.hnsw_m = max(2, int(hnsw_m))
        self.hnsw_ef_construction = max(1, int(hnsw_ef_construction))

    @staticmethod
    def _normalize_index_type(index_type: str) -> str:
        normalized = (index_type or "flat").strip().lower()
        if normalized not in {"flat", "ivf", "hnsw"}:
            raise ValueError(f"Unsupported vector index type '{index_type}'. Expected one of: ['flat', 'ivf', 'hnsw'].")
        return normalized

    def _resolve_ivf_nlist(self, num_embeddings: int) -> int:
        if num_embeddings <= 0:
            raise ValueError("Cannot create an IVF index without embeddings.")
        return min(self.ivf_nlist, num_embeddings)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_backend.embed_texts(texts)
        if not embeddings:
            raise ValueError("No embeddings were produced for the provided text chunks.")
        return embeddings

    def _create_vector_db(self, embeddings: List[float]):
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        if self.index_type == "ivf":
            actual_nlist = self._resolve_ivf_nlist(len(embeddings_array))
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, actual_nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings_array)
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.hnsw_ef_construction
        else:
            index = faiss.IndexFlatIP(dimension)  # Cosine distance
        index.add(embeddings_array)
        return index
    
    def _process_report(self, report: dict):
        text_chunks = [_vector_text_from_chunk(chunk) for chunk in report['content']['chunks']]
        embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            index = self._process_report(report_data)
            sha1_name = report_data["metainfo"]["sha1_name"]
            faiss_file_path = output_dir / f"{sha1_name}.faiss"
            faiss.write_index(index, str(faiss_file_path))

        print(f"Processed {len(all_report_paths)} reports")


class TagIngestor:
    def __init__(self):
        pass

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for tag retrieval"):
            with open(report_path, "r", encoding="utf-8") as file:
                report_data = json.load(file)

            chunk_terms: List[List[str]] = []
            chunk_tag_values: List[List[str]] = []
            for chunk in report_data.get("content", {}).get("chunks", []):
                tag_values = _chunk_tag_values(chunk)
                chunk_tag_values.append(tag_values)
                chunk_terms.append(tokenize_for_bm25(" ".join(tag_values)))

            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.json"
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "chunk_terms": chunk_terms,
                        "chunk_tag_values": chunk_tag_values,
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

        print(f"Processed {len(all_report_paths)} reports")

        
