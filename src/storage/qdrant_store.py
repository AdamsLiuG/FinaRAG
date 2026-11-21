"""Unified hybrid retrieval engine supporting both embedded disk (FAISS-SQ8+FTS5+Sparse) and Qdrant HTTP."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
from urllib.parse import urlparse

import faiss
import httpx
import numpy as np

from src.retrieval import _build_retrieval_result, _normalize_parent_retrieval_mode, _reciprocal_rank_fusion_score
from src.retrieval_filters import RetrievalFilters, apply_retrieval_filters
from src.storage.base import HybridSearchFilter, HybridSearchHit, StorageError, StorageSecurityError, validate_safe_path
from src.storage.sqlite_doc_store import SqliteDocumentRepository, _BoundedLRU
from src.text_normalization import tokenize_for_bm25

_POSTING_STRUCT = struct.Struct("<If")
_POSTING_DTYPE = np.dtype([("pid", "<u4"), ("weight", "<f4")])
_SAFE_FTS_TOKEN_RE = re.compile(r"[^\w\u4e00-\u9fff.-]+")

QDRANT_PAYLOAD_INDEXES: Tuple[Tuple[str, str], ...] = (
    ("doc_id", "keyword"),
    ("company_name", "keyword"),
    ("report_year", "integer"),
    ("industry_l1", "keyword"),
    ("page", "integer"),
    ("node_type", "keyword"),
    ("section_name", "keyword"),
    ("statement_location", "keyword"),
    ("topic_flags", "keyword"),
    ("strategy_tags", "keyword"),
)


def build_qdrant_collection_payload(
    collection_name: str,
    *,
    dense_dim: int = 1024,
    colbert_dim: int = 1024,
) -> Dict[str, Any]:
    """Construct Qdrant collection schema with dense INT8, sparse, and ColBERT multi-vectors."""
    return {
        "collection_name": collection_name,
        "vectors": {
            "dense": {
                "size": int(dense_dim),
                "distance": "Cosine",
                "on_disk": True,
                "quantization_config": {
                    "scalar": {
                        "type": "int8",
                        "quantile": 0.99,
                        "always_ram": True,
                    }
                },
            },
            "colbert": {
                "size": int(colbert_dim),
                "distance": "Cosine",
                "multivector_config": {"comparator": "max_sim"},
                "hnsw_config": {"m": 0},
            },
        },
        "sparse_vectors": {
            "bge_sparse": {},
            "bm25_sparse": {"modifier": "idf"},
        },
    }


def _sanitize_fts_tokens(query_text: str) -> List[str]:
    tokens = tokenize_for_bm25(query_text or "")
    cleaned: List[str] = []
    seen = set()
    for tok in tokens:
        safe = _SAFE_FTS_TOKEN_RE.sub("", tok).strip()
        if not safe or safe in {"AND", "OR", "NOT", "NEAR"}:
            continue
        if safe not in seen:
            seen.add(safe)
            cleaned.append(safe)
    return cleaned


def _l2_normalize_matrix(arr: np.ndarray) -> np.ndarray:
    mat = np.asarray(arr, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return (mat / norms).astype(np.float32, copy=False)


def _sparse_to_indices_values(weights: Dict[str, float]) -> Dict[str, Any]:
    indices: List[int] = []
    values: List[float] = []
    for k, v in (weights or {}).items():
        val = float(v)
        if val <= 0:
            continue
        try:
            idx = int(k)
        except (TypeError, ValueError):
            idx = abs(hash(str(k))) % 2_000_000_000
        indices.append(idx)
        values.append(val)
    return {"indices": indices, "values": values}


def _build_qdrant_filter(filters: Optional[HybridSearchFilter]) -> Optional[Dict[str, Any]]:
    if filters is None:
        return None
    must: List[Dict[str, Any]] = []
    if filters.doc_ids:
        must.append({"key": "doc_id", "match": {"any": [str(x) for x in filters.doc_ids]}})
    if filters.company_name:
        must.append({"key": "company_name", "match": {"value": str(filters.company_name)}})
    if filters.report_year is not None:
        must.append({"key": "report_year", "match": {"value": int(filters.report_year)}})
    if filters.industry_l1:
        must.append({"key": "industry_l1", "match": {"value": str(filters.industry_l1)}})
    if filters.node_type:
        must.append({"key": "node_type", "match": {"value": str(filters.node_type)}})
    if filters.section_name:
        must.append({"key": "section_name", "match": {"value": str(filters.section_name)}})
    if filters.topic_flags:
        must.append({"key": "topic_flags", "match": {"any": [str(x) for x in filters.topic_flags]}})
    if filters.strategy_tags:
        must.append({"key": "strategy_tags", "match": {"any": [str(x) for x in filters.strategy_tags]}})
    return {"must": must} if must else None


def _maxsim_score(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    q = _l2_normalize_matrix(query_vecs)
    d = _l2_normalize_matrix(doc_vecs)
    if q.size == 0 or d.size == 0:
        return 0.0
    sim = q @ d.T
    return float(np.max(sim, axis=1).sum())


class QdrantHybridStore:
    """Unified Dense + Sparse + BM25 + Multi-Vector store (embedded or Qdrant HTTP)."""

    def __init__(
        self,
        *,
        mode: Literal["embedded", "http"] = "embedded",
        store_dir: Optional[Path | str] = None,
        collection_name: str = "finarag_chunks_v1",
        qdrant_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dense_dim: int = 1024,
        colbert_dim: int = 1024,
        allowed_base_dir: Optional[Path | str] = None,
        http_transport: Optional[httpx.BaseTransport] = None,
        timeout: float = 15.0,
    ):
        self.mode = mode.strip().lower()
        if self.mode not in {"embedded", "http"}:
            raise ValueError("QdrantHybridStore mode must be 'embedded' or 'http'")
        self.collection_name = collection_name
        self.dense_dim = int(dense_dim)
        self.colbert_dim = int(colbert_dim)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._conns: List[sqlite3.Connection] = []
        self._faiss_index = None
        self._sparse_blob_cache = _BoundedLRU(512)
        self._filter_ids_cache = _BoundedLRU(64)
        self._max_pid_cache: Optional[int] = None
        self._client: Optional[httpx.Client] = None
        self.store_dir: Optional[Path] = None

        if self.mode == "http":
            raw_url = (qdrant_url or os.getenv("QDRANT_URL", "")).strip().rstrip("/")
            if not raw_url:
                raise StorageSecurityError("qdrant_url (or QDRANT_URL) is required in 'http' mode")
            parsed = urlparse(raw_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise StorageSecurityError("Invalid Qdrant URL scheme or host")
            self.qdrant_url = raw_url
            resolved_key = api_key if api_key is not None else os.getenv("QDRANT_API_KEY")
            headers = {"api-key": resolved_key} if resolved_key else {}
            self._client = httpx.Client(
                base_url=self.qdrant_url,
                headers=headers,
                timeout=timeout,
                transport=http_transport,
            )
        else:
            if store_dir is None:
                raise ValueError("store_dir is required in 'embedded' mode")
            self.store_dir = validate_safe_path(store_dir, allowed_base_dir)
            self.faiss_path = self.store_dir / "finarag_unified.faiss"
            self.meta_db_path = self.store_dir / "finarag_hybrid_meta.sqlite"

    # ------------------------------------------------------------------
    # HTTP mode (real Qdrant REST API)
    # ------------------------------------------------------------------

    def _http_ensure_collection(self) -> None:
        assert self._client is not None
        schema = build_qdrant_collection_payload(
            self.collection_name,
            dense_dim=self.dense_dim,
            colbert_dim=self.colbert_dim,
        )
        body = {k: v for k, v in schema.items() if k != "collection_name"}
        resp = self._client.put(f"/collections/{self.collection_name}", json=body)
        if resp.status_code not in {200, 201, 409}:
            raise StorageError(f"Qdrant collection creation failed (HTTP {resp.status_code})")
        for field_name, field_schema in QDRANT_PAYLOAD_INDEXES:
            idx_resp = self._client.put(
                f"/collections/{self.collection_name}/index",
                json={"field_name": field_name, "field_schema": field_schema},
            )
            if idx_resp.status_code not in {200, 201, 409}:
                raise StorageError(f"Qdrant payload index creation failed (HTTP {idx_resp.status_code})")

    def _http_upsert_points(self, points: Sequence[Dict[str, Any]], batch_size: int = 256) -> int:
        assert self._client is not None
        total = 0
        for start in range(0, len(points), max(1, int(batch_size))):
            batch = points[start : start + max(1, int(batch_size))]
            serialized = []
            for p in batch:
                vectors: Dict[str, Any] = {}
                if p.get("dense_vector") is not None:
                    vectors["dense"] = _l2_normalize_matrix(np.asarray(p["dense_vector"], dtype=np.float32))[0].tolist()
                if p.get("bge_sparse"):
                    vectors["bge_sparse"] = _sparse_to_indices_values(p["bge_sparse"])
                if p.get("bm25_sparse"):
                    vectors["bm25_sparse"] = _sparse_to_indices_values(p["bm25_sparse"])
                if p.get("colbert_vectors") is not None:
                    vectors["colbert"] = _l2_normalize_matrix(np.asarray(p["colbert_vectors"], dtype=np.float32)).tolist()
                payload = {
                    "doc_id": str(p.get("doc_id") or ""),
                    "chunk_idx": int(p.get("chunk_idx") or 0),
                    "company_name": str(p.get("company_name") or ""),
                    "report_year": p.get("report_year"),
                    "industry_l1": str(p.get("industry_l1") or ""),
                    "page": int(p.get("page") or 0),
                    "node_type": str(p.get("node_type") or "child"),
                    "section_name": str(p.get("section_name") or ""),
                    "statement_location": str(p.get("statement_location") or ""),
                    "topic_flags": list(p.get("topic_flags") or []),
                    "strategy_tags": list(p.get("strategy_tags") or []),
                    "text": str(p.get("text") or ""),
                }
                serialized.append({"id": int(p["point_id"]), "vector": vectors, "payload": payload})
            resp = self._client.put(
                f"/collections/{self.collection_name}/points?wait=true",
                json={"points": serialized},
            )
            if resp.status_code not in {200, 201}:
                raise StorageError(f"Qdrant upsert failed (HTTP {resp.status_code})")
            total += len(batch)
        return total

    def _http_search_hybrid(
        self,
        *,
        dense_vector: Any,
        query_text: str,
        bge_sparse: Optional[Dict[str, float]],
        colbert_vectors: Any,
        filters: Optional[HybridSearchFilter],
        top_k: int,
        prefetch_limit: int,
    ) -> List[HybridSearchHit]:
        assert self._client is not None
        q_filter = _build_qdrant_filter(filters)
        prefetch: List[Dict[str, Any]] = []
        if dense_vector is not None:
            vec = _l2_normalize_matrix(np.asarray(dense_vector, dtype=np.float32))[0].tolist()
            entry: Dict[str, Any] = {"query": vec, "using": "dense", "limit": int(prefetch_limit)}
            if q_filter:
                entry["filter"] = q_filter
            prefetch.append(entry)
        if bge_sparse:
            entry = {"query": _sparse_to_indices_values(bge_sparse), "using": "bge_sparse", "limit": int(prefetch_limit)}
            if q_filter:
                entry["filter"] = q_filter
            prefetch.append(entry)
        bm25_tokens = _sanitize_fts_tokens(query_text)
        if bm25_tokens:
            bm25_weights = {tok: 1.0 for tok in bm25_tokens}
            entry = {"query": _sparse_to_indices_values(bm25_weights), "using": "bm25_sparse", "limit": int(prefetch_limit)}
            if q_filter:
                entry["filter"] = q_filter
            prefetch.append(entry)

        if not prefetch:
            return []

        if colbert_vectors is not None:
            cb_list = _l2_normalize_matrix(np.asarray(colbert_vectors, dtype=np.float32)).tolist()
            body: Dict[str, Any] = {
                "prefetch": {"prefetch": prefetch, "fusion": "rrf", "limit": int(prefetch_limit)},
                "query": cb_list,
                "using": "colbert",
                "limit": int(top_k),
                "with_payload": True,
            }
        else:
            body = {
                "prefetch": prefetch,
                "query": {"fusion": "rrf"},
                "limit": int(top_k),
                "with_payload": True,
            }
        if q_filter:
            body["filter"] = q_filter

        resp = self._client.post(f"/collections/{self.collection_name}/points/query", json=body)
        if resp.status_code != 200:
            raise StorageError(f"Qdrant hybrid query failed (HTTP {resp.status_code})")
        data = resp.json()
        raw_points = (data.get("result") or {}).get("points") if isinstance(data.get("result"), dict) else data.get("result")
        hits: List[HybridSearchHit] = []
        for item in raw_points or []:
            payload = item.get("payload") or {}
            hits.append(
                HybridSearchHit(
                    point_id=int(item["id"]),
                    doc_id=str(payload.get("doc_id") or ""),
                    chunk_idx=int(payload.get("chunk_idx") or 0),
                    score=float(item.get("score") or 0.0),
                    page=int(payload.get("page") or 0),
                    text=str(payload.get("text") or ""),
                    metadata=dict(payload),
                    retrieval_sources=["qdrant_hybrid"],
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Embedded unified disk mode (FAISS SQ8 + SQLite WAL + FTS5 + Sparse)
    # ------------------------------------------------------------------

    def _get_embedded_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        if not self.meta_db_path.exists():
            raise StorageError("Embedded hybrid metadata database does not exist")
        try:
            uri = f"{self.meta_db_path.as_uri()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
        except sqlite3.Error as exc:
            raise StorageError("Failed to open embedded hybrid metadata database") from exc
        self._local.conn = conn
        with self._conns_lock_or_create():
            self._conns.append(conn)
        return conn

    def _conns_lock_or_create(self) -> threading.Lock:
        return self._lock

    def _get_faiss_index(self):
        if self._faiss_index is not None:
            return self._faiss_index
        with self._lock:
            if self._faiss_index is None:
                if not self.faiss_path.exists():
                    raise StorageError("Unified FAISS index does not exist")
                self._faiss_index = faiss.read_index(str(self.faiss_path))
        return self._faiss_index

    @staticmethod
    def _create_embedded_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS store_meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS chunks_meta (
                point_id           INTEGER PRIMARY KEY,
                doc_id             TEXT NOT NULL,
                chunk_idx          INTEGER NOT NULL,
                company_name       TEXT,
                report_year        INTEGER,
                industry_l1        TEXT,
                page               INTEGER,
                node_type          TEXT,
                section_name       TEXT,
                statement_location TEXT,
                topic_flags_json   TEXT,
                strategy_tags_json TEXT,
                text               TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                doc_key,
                content,
                tokenize="unicode61 tokenchars '_-'"
            );
            CREATE TABLE IF NOT EXISTS sparse_postings (
                term_id  TEXT PRIMARY KEY,
                df       INTEGER NOT NULL,
                postings BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS colbert_vectors (
                point_id INTEGER PRIMARY KEY,
                n_tokens INTEGER NOT NULL,
                dim      INTEGER NOT NULL,
                vec_blob BLOB NOT NULL
            );
            """
        )

    @staticmethod
    def _create_embedded_indexes(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_meta_company_year ON chunks_meta(company_name, report_year);
            CREATE INDEX IF NOT EXISTS idx_meta_doc_chunk ON chunks_meta(doc_id, chunk_idx);
            CREATE INDEX IF NOT EXISTS idx_meta_industry ON chunks_meta(industry_l1);
            CREATE INDEX IF NOT EXISTS idx_meta_node_type ON chunks_meta(node_type);
            """
        )

    def ensure_collection(self) -> None:
        if self.mode == "http":
            self._http_ensure_collection()
            return
        assert self.store_dir is not None
        self.store_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.meta_db_path))
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            self._create_embedded_schema(conn)
            self._create_embedded_indexes(conn)
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()

    def build_embedded_from_batches(
        self,
        batches_iter: Iterable[Tuple[List[Dict[str, Any]], np.ndarray]],
        *,
        quantize_sq8: bool = True,
    ) -> Dict[str, Any]:
        """Stream all documents cleanly into a single unified FAISS (SQ8) + SQLite WAL/FTS5/Sparse store."""
        assert self.store_dir is not None
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.close()

        tmp_db = self.meta_db_path.with_name(f".{self.meta_db_path.name}.tmp")
        tmp_faiss = self.faiss_path.with_name(f".{self.faiss_path.name}.tmp")
        for p in (tmp_db, Path(f"{tmp_db}-wal"), Path(f"{tmp_db}-shm"), tmp_faiss):
            if p.exists():
                p.unlink()

        conn = sqlite3.connect(str(tmp_db))
        vec_chunks: List[np.ndarray] = []
        sparse_accum: Dict[str, bytearray] = {}
        sparse_df: Dict[str, int] = {}
        next_point_id = 0
        inferred_dim = self.dense_dim
        colbert_count = 0

        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._create_embedded_schema(conn)

            for records, vec_matrix in batches_iter:
                if not records:
                    continue
                if vec_matrix is not None and len(vec_matrix) > 0:
                    normed = _l2_normalize_matrix(vec_matrix)
                    inferred_dim = int(normed.shape[1])
                    vec_chunks.append(normed)

                meta_rows = []
                fts_rows = []
                colbert_rows = []
                for idx, rec in enumerate(records):
                    pid = int(rec["point_id"]) if rec.get("point_id") is not None else next_point_id
                    next_point_id = max(next_point_id + 1, pid + 1)
                    doc_id = str(rec.get("doc_id") or "")
                    text = str(rec.get("text") or "")
                    bm25_tokens = rec.get("bm25_tokens")
                    if bm25_tokens is None:
                        bm25_tokens = _sanitize_fts_tokens(text)
                    fts_content = " ".join(bm25_tokens) if bm25_tokens else ""

                    meta_rows.append(
                        (
                            pid,
                            doc_id,
                            int(rec.get("chunk_idx") if rec.get("chunk_idx") is not None else idx),
                            str(rec.get("company_name") or ""),
                            int(rec["report_year"]) if rec.get("report_year") is not None else None,
                            str(rec.get("industry_l1") or ""),
                            int(rec.get("page") or 0),
                            str(rec.get("node_type") or "child"),
                            str(rec.get("section_name") or ""),
                            str(rec.get("statement_location") or ""),
                            json.dumps(list(rec.get("topic_flags") or []), ensure_ascii=False),
                            json.dumps(list(rec.get("strategy_tags") or []), ensure_ascii=False),
                            text,
                        )
                    )
                    fts_rows.append((pid, doc_id, fts_content))

                    bge_sparse = rec.get("bge_sparse")
                    if isinstance(bge_sparse, dict) and bge_sparse:
                        top_terms = sorted(bge_sparse.items(), key=lambda kv: float(kv[1]), reverse=True)[:64]
                        for term, weight in top_terms:
                            w = float(weight)
                            if w <= 0:
                                continue
                            t_key = str(term)
                            buf = sparse_accum.get(t_key)
                            if buf is None:
                                buf = bytearray()
                                sparse_accum[t_key] = buf
                                sparse_df[t_key] = 0
                            buf.extend(_POSTING_STRUCT.pack(pid, w))
                            sparse_df[t_key] += 1

                    cb = rec.get("colbert_vectors")
                    if cb is not None:
                        cb_arr = _l2_normalize_matrix(np.asarray(cb, dtype=np.float32))
                        colbert_rows.append((pid, int(cb_arr.shape[0]), int(cb_arr.shape[1]), cb_arr.tobytes()))
                        colbert_count += 1

                conn.executemany(
                    """
                    INSERT OR REPLACE INTO chunks_meta (
                        point_id, doc_id, chunk_idx, company_name, report_year, industry_l1,
                        page, node_type, section_name, statement_location,
                        topic_flags_json, strategy_tags_json, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    meta_rows,
                )
                conn.executemany(
                    "INSERT INTO chunks_fts(rowid, doc_key, content) VALUES (?, ?, ?)",
                    fts_rows,
                )
                if colbert_rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO colbert_vectors (point_id, n_tokens, dim, vec_blob) VALUES (?, ?, ?, ?)",
                        colbert_rows,
                    )

            posting_rows = [(t_id, sparse_df[t_id], bytes(buf)) for t_id, buf in sparse_accum.items()]
            sparse_accum.clear()
            for start in range(0, len(posting_rows), 20_000):
                conn.executemany(
                    "INSERT OR REPLACE INTO sparse_postings (term_id, df, postings) VALUES (?, ?, ?)",
                    posting_rows[start : start + 20_000],
                )

            all_vecs = np.vstack(vec_chunks) if vec_chunks else np.zeros((0, inferred_dim), dtype=np.float32)
            total_points = next_point_id
            vec_count = int(all_vecs.shape[0])
            if quantize_sq8 and vec_count >= 4096:
                nlist = min(512, max(64, int(math.sqrt(vec_count))))
                quantizer = faiss.IndexFlatIP(inferred_dim)
                index = faiss.IndexIVFScalarQuantizer(
                    quantizer,
                    inferred_dim,
                    nlist,
                    faiss.ScalarQuantizer.QT_8bit,
                    faiss.METRIC_INNER_PRODUCT,
                )
                sample_n = min(vec_count, 65_536)
                step = max(1, vec_count // sample_n)
                train_slice = np.ascontiguousarray(all_vecs[::step][:sample_n], dtype=np.float32)
                index.train(train_slice)
                index.add(np.ascontiguousarray(all_vecs, dtype=np.float32))
                index.nprobe = min(24, nlist)
                dense_quant = "ivf_sq8_int8"
            elif quantize_sq8 and vec_count >= 256:
                index = faiss.IndexScalarQuantizer(inferred_dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
                sample_n = min(vec_count, 65_536)
                step = max(1, vec_count // sample_n)
                train_slice = np.ascontiguousarray(all_vecs[::step][:sample_n], dtype=np.float32)
                index.train(train_slice)
                index.add(np.ascontiguousarray(all_vecs, dtype=np.float32))
                dense_quant = "sq8_int8"
            else:
                index = faiss.IndexFlatIP(inferred_dim)
                if vec_count > 0:
                    index.add(np.ascontiguousarray(all_vecs, dtype=np.float32))
                dense_quant = "float32_flat_ip"

            faiss.write_index(index, str(tmp_faiss))
            self._create_embedded_indexes(conn)
            conn.executemany(
                "INSERT OR REPLACE INTO store_meta (key, value) VALUES (?, ?)",
                [
                    ("collection_name", self.collection_name),
                    ("dense_dim", str(inferred_dim)),
                    ("dense_quantization", dense_quant),
                    ("vectors_count", str(vec_count)),
                    ("sparse_terms_count", str(len(posting_rows))),
                    ("colbert_count", str(colbert_count)),
                ],
            )
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()

        os.replace(tmp_faiss, self.faiss_path)
        os.replace(tmp_db, self.meta_db_path)
        for p in (Path(f"{tmp_db}-wal"), Path(f"{tmp_db}-shm")):
            if p.exists():
                p.unlink()

        return {
            "store_dir": str(self.store_dir),
            "collection_name": self.collection_name,
            "vectors_count": vec_count,
            "dense_dim": inferred_dim,
            "dense_quantization": dense_quant,
            "sparse_terms_count": len(posting_rows),
            "colbert_count": colbert_count,
        }

    def upsert_points(self, points: Sequence[Dict[str, Any]], batch_size: int = 256) -> int:
        if self.mode == "http":
            return self._http_upsert_points(points, batch_size=batch_size)
        if not points:
            return 0
        # For embedded incremental usage (tests/small updates), merge existing rows cleanly.
        assert self.store_dir is not None
        existing_vecs: Optional[np.ndarray] = None
        existing_records: Dict[int, Dict[str, Any]] = {}
        existing_sparse: Dict[str, Dict[int, float]] = {}

        if self.meta_db_path.exists() and self.faiss_path.exists():
            conn = sqlite3.connect(str(self.meta_db_path))
            conn.row_factory = sqlite3.Row
            try:
                old_index = faiss.read_index(str(self.faiss_path))
                if old_index.ntotal > 0:
                    existing_vecs = old_index.reconstruct_n(0, old_index.ntotal)
                for r in conn.execute("SELECT * FROM chunks_meta ORDER BY point_id ASC"):
                    existing_records[int(r["point_id"])] = {
                        "point_id": int(r["point_id"]),
                        "doc_id": r["doc_id"],
                        "chunk_idx": int(r["chunk_idx"]),
                        "company_name": r["company_name"],
                        "report_year": r["report_year"],
                        "industry_l1": r["industry_l1"],
                        "page": int(r["page"] or 0),
                        "node_type": r["node_type"],
                        "section_name": r["section_name"],
                        "statement_location": r["statement_location"],
                        "topic_flags": json.loads(r["topic_flags_json"] or "[]"),
                        "strategy_tags": json.loads(r["strategy_tags_json"] or "[]"),
                        "text": r["text"],
                    }
                for r in conn.execute("SELECT term_id, postings FROM sparse_postings"):
                    arr = np.frombuffer(r["postings"], dtype=_POSTING_DTYPE)
                    for row in arr:
                        existing_sparse.setdefault(int(row["pid"]), {})[str(r["term_id"])] = float(row["weight"])
                for r in conn.execute("SELECT point_id, n_tokens, dim, vec_blob FROM colbert_vectors"):
                    pid = int(r["point_id"])
                    if pid in existing_records:
                        existing_records[pid]["colbert_vectors"] = np.frombuffer(
                            r["vec_blob"], dtype=np.float32
                        ).reshape(int(r["n_tokens"]), int(r["dim"])).copy()
            finally:
                conn.close()

        for pid, sparse_dict in existing_sparse.items():
            if pid in existing_records:
                existing_records[pid]["bge_sparse"] = sparse_dict

        next_pid = (max(existing_records.keys()) + 1) if existing_records else 0
        for p in points:
            rec = dict(p)
            if rec.get("point_id") is None:
                rec["point_id"] = next_pid
                next_pid += 1
            pid = int(rec["point_id"])
            if rec.get("dense_vector") is None and existing_vecs is not None and pid < len(existing_vecs):
                rec["dense_vector"] = existing_vecs[pid]
            existing_records[pid] = rec

        ordered_pids = sorted(existing_records.keys())
        ordered_recs: List[Dict[str, Any]] = []
        vec_list: List[np.ndarray] = []
        for idx, pid in enumerate(ordered_pids):
            rec = dict(existing_records[pid])
            rec["point_id"] = idx
            vec = rec.get("dense_vector")
            if vec is None:
                if existing_vecs is not None and pid < len(existing_vecs):
                    vec = existing_vecs[pid]
                else:
                    vec = np.zeros(self.dense_dim, dtype=np.float32)
            vec_list.append(np.asarray(vec, dtype=np.float32))
            ordered_recs.append(rec)

        mat = np.vstack(vec_list) if vec_list else np.zeros((0, self.dense_dim), dtype=np.float32)
        self.build_embedded_from_batches([(ordered_recs, mat)], quantize_sq8=len(ordered_recs) >= 256)
        return len(points)

    def _embedded_allowed_ids(self, filters: Optional[HybridSearchFilter]) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        if filters is None:
            return None, None
        filter_key = (
            tuple(str(x) for x in (filters.doc_ids or ())),
            filters.company_name,
            filters.report_year,
            filters.industry_l1,
            filters.node_type,
            filters.section_name,
            tuple(str(x) for x in (filters.topic_flags or ())),
            tuple(str(x) for x in (filters.strategy_tags or ())),
        )
        cached = self._filter_ids_cache.get(filter_key, default=None)
        if cached is not None:
            return cached

        clauses: List[str] = []
        params: List[Any] = []
        if filters.doc_ids:
            placeholders = ",".join("?" for _ in filters.doc_ids)
            clauses.append(f"doc_id IN ({placeholders})")
            params.extend([str(d) for d in filters.doc_ids])
        if filters.company_name:
            clauses.append("company_name = ?")
            params.append(str(filters.company_name))
        if filters.report_year is not None:
            clauses.append("report_year = ?")
            params.append(int(filters.report_year))
        if filters.industry_l1:
            clauses.append("industry_l1 = ?")
            params.append(str(filters.industry_l1))
        if filters.node_type:
            clauses.append("node_type = ?")
            params.append(str(filters.node_type))
        if filters.section_name:
            clauses.append("section_name = ?")
            params.append(str(filters.section_name))

        if not clauses and not filters.topic_flags and not filters.strategy_tags:
            return None, None

        conn = self._get_embedded_conn()
        sql = "SELECT point_id, doc_id, topic_flags_json, strategy_tags_json FROM chunks_meta"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        try:
            rows = conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.Error as exc:
            raise StorageError("Hybrid metadata filter query failed") from exc

        want_topics = set(filters.topic_flags or [])
        want_strategies = set(filters.strategy_tags or [])
        allowed_ids: List[int] = []
        doc_ids_seen: List[str] = []
        doc_set = set()
        for r in rows:
            if want_topics:
                row_topics = set(json.loads(r["topic_flags_json"] or "[]"))
                if not (want_topics & row_topics):
                    continue
            if want_strategies:
                row_strats = set(json.loads(r["strategy_tags_json"] or "[]"))
                if not (want_strategies & row_strats):
                    continue
            allowed_ids.append(int(r["point_id"]))
            d = str(r["doc_id"])
            if d not in doc_set:
                doc_set.add(d)
                doc_ids_seen.append(d)

        result_pair = (np.asarray(allowed_ids, dtype=np.int64), doc_ids_seen)
        self._filter_ids_cache.put(filter_key, result_pair)
        return result_pair

    def _embedded_search_hybrid(
        self,
        *,
        dense_vector: Any,
        query_text: str,
        bge_sparse: Optional[Dict[str, float]],
        colbert_vectors: Any,
        filters: Optional[HybridSearchFilter],
        top_k: int,
        prefetch_limit: int,
        rrf_k: int,
    ) -> List[HybridSearchHit]:
        conn = self._get_embedded_conn()
        allowed_ids, scoped_doc_ids = self._embedded_allowed_ids(filters)
        if allowed_ids is not None and len(allowed_ids) == 0:
            return []

        branch_ranks: Dict[str, List[int]] = {}
        allowed_set = set(int(x) for x in allowed_ids.tolist()) if allowed_ids is not None else None

        # 1. Dense FAISS (FlatIP, SQ8, or IVF-SQ8) with IDSelectorBatch filter
        if dense_vector is not None:
            index = self._get_faiss_index()
            if index.ntotal > 0:
                q_vec = _l2_normalize_matrix(np.asarray(dense_vector, dtype=np.float32))
                if q_vec.shape[1] == index.d:
                    k_req = min(int(prefetch_limit), int(index.ntotal))
                    is_ivf = isinstance(index, faiss.IndexIVF)
                    if allowed_ids is not None:
                        sel = faiss.IDSelectorBatch(allowed_ids)
                        if is_ivf:
                            params = faiss.SearchParametersIVF(nprobe=min(32, index.nlist), sel=sel)
                        else:
                            params = faiss.SearchParameters(sel=sel)
                        _, indices = index.search(q_vec, min(k_req, len(allowed_ids)), params=params)
                    else:
                        if is_ivf:
                            index.nprobe = min(24, index.nlist)
                        _, indices = index.search(q_vec, k_req)
                    dense_pids = [int(pid) for pid in indices[0] if int(pid) >= 0]
                    if dense_pids:
                        branch_ranks["dense"] = dense_pids

        # 2. BM25 FTS5 prefetch (AND-first to avoid slow OR union on ubiquitous tokens like years)
        safe_tokens = _sanitize_fts_tokens(query_text)
        if safe_tokens:
            non_year_tokens = [tok for tok in safe_tokens if not re.fullmatch(r"(?:19|20)\d{2}", tok)]
            and_tokens = (non_year_tokens or safe_tokens)[:4]
            or_tokens = (non_year_tokens or safe_tokens)[:6]
            doc_prefix = ""
            if scoped_doc_ids and len(scoped_doc_ids) <= 32:
                doc_expr = " OR ".join(f'"{d}"' for d in scoped_doc_ids)
                doc_prefix = f"doc_key:({doc_expr}) AND "

            bm25_pids: List[int] = []
            seen_bm25 = set()
            fetch_cap = int(prefetch_limit) * 3 if allowed_set is not None else int(prefetch_limit)

            def _run_fts_expr(expr: str) -> None:
                try:
                    rows = conn.execute(
                        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) ASC LIMIT ?",
                        (f"{doc_prefix}content:({expr})", fetch_cap),
                    ).fetchall()
                except sqlite3.Error as exc:
                    raise StorageError("BM25 FTS5 query failed") from exc
                for r in rows:
                    pid = int(r[0])
                    if pid in seen_bm25:
                        continue
                    if allowed_set is None or pid in allowed_set:
                        seen_bm25.add(pid)
                        bm25_pids.append(pid)
                        if len(bm25_pids) >= int(prefetch_limit):
                            break

            if len(and_tokens) >= 2:
                _run_fts_expr(" AND ".join(f'"{tok}"' for tok in and_tokens))
            if len(bm25_pids) < min(10, int(prefetch_limit)):
                _run_fts_expr(" OR ".join(f'"{tok}"' for tok in or_tokens))
            if bm25_pids:
                branch_ranks["bm25"] = bm25_pids

        # 3. BGE-M3 Sparse lexical prefetch (Top-20 terms + bounded LRU blob cache + vectorized NumPy accumulation)
        if bge_sparse:
            terms = sorted(
                ((str(k), float(v)) for k, v in bge_sparse.items() if float(v) > 0),
                key=lambda kv: kv[1],
                reverse=True,
            )[:20]
            if terms:
                q_map = {k: v for k, v in terms}
                blobs_by_term: Dict[str, bytes] = {}
                missing_terms: List[str] = []
                for term_id, _ in terms:
                    cached_blob = self._sparse_blob_cache.get(term_id, default=None)
                    if cached_blob is not None:
                        if cached_blob:
                            blobs_by_term[term_id] = cached_blob
                    else:
                        missing_terms.append(term_id)

                if missing_terms:
                    placeholders = ",".join("?" for _ in missing_terms)
                    try:
                        rows = conn.execute(
                            f"SELECT term_id, postings FROM sparse_postings WHERE term_id IN ({placeholders})",
                            missing_terms,
                        ).fetchall()
                    except sqlite3.Error as exc:
                        raise StorageError("Sparse postings lookup failed") from exc
                    found_terms = set()
                    for r in rows:
                        tid = str(r["term_id"])
                        blob = bytes(r["postings"])
                        self._sparse_blob_cache.put(tid, blob)
                        blobs_by_term[tid] = blob
                        found_terms.add(tid)
                    for tid in missing_terms:
                        if tid not in found_terms:
                            self._sparse_blob_cache.put(tid, b"")

                if self._max_pid_cache is None:
                    row_max = conn.execute("SELECT MAX(point_id) FROM chunks_meta").fetchone()[0]
                    self._max_pid_cache = int(row_max) + 1 if row_max is not None else 0
                max_pid = self._max_pid_cache

                if blobs_by_term and max_pid > 0:
                    scores_arr = np.zeros(max_pid, dtype=np.float32)
                    allowed_mask = None
                    if allowed_ids is not None:
                        allowed_mask = np.zeros(max_pid, dtype=bool)
                        valid_allowed = allowed_ids[allowed_ids < max_pid]
                        allowed_mask[valid_allowed] = True
                    for tid, blob in blobs_by_term.items():
                        q_w = np.float32(q_map.get(tid, 0.0))
                        arr = np.frombuffer(blob, dtype=_POSTING_DTYPE)
                        if arr.size == 0:
                            continue
                        pids = arr["pid"]
                        weights = arr["weight"]
                        if allowed_mask is not None:
                            valid = (pids < max_pid) & allowed_mask[pids]
                        else:
                            valid = pids < max_pid
                        if np.any(valid):
                            np.add.at(scores_arr, pids[valid], weights[valid] * q_w)
                    nonzero = np.flatnonzero(scores_arr > 0)
                    if nonzero.size > 0:
                        nz_scores = scores_arr[nonzero]
                        k = min(int(prefetch_limit), int(nonzero.size))
                        if nonzero.size > k:
                            top_idx = np.argpartition(nz_scores, -k)[-k:]
                            top_idx = top_idx[np.argsort(nz_scores[top_idx])[::-1]]
                        else:
                            top_idx = np.argsort(nz_scores)[::-1]
                        branch_ranks["bge_sparse"] = [int(x) for x in nonzero[top_idx]]

        if not branch_ranks:
            return []

        # 4. RRF fusion across active branches
        rrf_scores: Dict[int, float] = {}
        pid_sources: Dict[int, List[str]] = {}
        for branch_name, pids in branch_ranks.items():
            for rank, pid in enumerate(pids, start=1):
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + _reciprocal_rank_fusion_score(rank, rrf_k)
                sources = pid_sources.setdefault(pid, [])
                if branch_name not in sources:
                    sources.append(branch_name)

        candidate_limit = max(int(top_k), int(prefetch_limit)) if colbert_vectors is not None else int(top_k)
        fused_pids = [
            pid for pid, _ in sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:candidate_limit]
        ]
        if not fused_pids:
            return []

        # 5. Optional ColBERT multi-vector MaxSim stage-2 reranking
        final_scores: Dict[int, float] = {pid: rrf_scores[pid] for pid in fused_pids}
        if colbert_vectors is not None:
            q_cb = _l2_normalize_matrix(np.asarray(colbert_vectors, dtype=np.float32))
            placeholders = ",".join("?" for _ in fused_pids)
            cb_rows = conn.execute(
                f"SELECT point_id, n_tokens, dim, vec_blob FROM colbert_vectors WHERE point_id IN ({placeholders})",
                fused_pids,
            ).fetchall()
            cb_map = {
                int(r["point_id"]): np.frombuffer(r["vec_blob"], dtype=np.float32).reshape(int(r["n_tokens"]), int(r["dim"]))
                for r in cb_rows
            }
            if cb_map:
                for pid in fused_pids:
                    if pid in cb_map:
                        final_scores[pid] = _maxsim_score(q_cb, cb_map[pid])
                        if "colbert" not in pid_sources[pid]:
                            pid_sources[pid].append("colbert")

        ranked_pids = [
            pid for pid, _ in sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)[: int(top_k)]
        ]
        placeholders = ",".join("?" for _ in ranked_pids)
        meta_rows = conn.execute(
            f"SELECT * FROM chunks_meta WHERE point_id IN ({placeholders})",
            ranked_pids,
        ).fetchall()
        meta_by_pid = {int(r["point_id"]): r for r in meta_rows}

        hits: List[HybridSearchHit] = []
        for pid in ranked_pids:
            r = meta_by_pid.get(pid)
            if r is None:
                continue
            meta = {
                "doc_id": r["doc_id"],
                "sha1_name": r["doc_id"],
                "chunk_idx": int(r["chunk_idx"]),
                "company_name": r["company_name"],
                "report_year": r["report_year"],
                "industry_l1": r["industry_l1"],
                "page": int(r["page"] or 0),
                "node_type": r["node_type"],
                "section_name": r["section_name"],
                "statement_location": r["statement_location"],
                "topic_flags": json.loads(r["topic_flags_json"] or "[]"),
                "strategy_tags": json.loads(r["strategy_tags_json"] or "[]"),
            }
            hits.append(
                HybridSearchHit(
                    point_id=pid,
                    doc_id=str(r["doc_id"]),
                    chunk_idx=int(r["chunk_idx"]),
                    score=round(float(final_scores[pid]), 6),
                    page=int(r["page"] or 0),
                    text=str(r["text"]),
                    metadata=meta,
                    retrieval_sources=list(pid_sources.get(pid, ["unified_hybrid"])),
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Public unified API
    # ------------------------------------------------------------------

    def search_hybrid(
        self,
        *,
        dense_vector: Any = None,
        query_text: str = "",
        bge_sparse: Optional[Dict[str, float]] = None,
        colbert_vectors: Any = None,
        filters: Optional[HybridSearchFilter] = None,
        top_k: int = 10,
        prefetch_limit: int = 60,
        rrf_k: int = 60,
    ) -> List[HybridSearchHit]:
        if self.mode == "http":
            return self._http_search_hybrid(
                dense_vector=dense_vector,
                query_text=query_text,
                bge_sparse=bge_sparse,
                colbert_vectors=colbert_vectors,
                filters=filters,
                top_k=top_k,
                prefetch_limit=prefetch_limit,
            )
        return self._embedded_search_hybrid(
            dense_vector=dense_vector,
            query_text=query_text,
            bge_sparse=bge_sparse,
            colbert_vectors=colbert_vectors,
            filters=filters,
            top_k=top_k,
            prefetch_limit=prefetch_limit,
            rrf_k=rrf_k,
        )

    def stats(self) -> Dict[str, Any]:
        if self.mode == "http":
            assert self._client is not None
            try:
                resp = self._client.get(f"/collections/{self.collection_name}")
                if resp.status_code == 200:
                    return {"mode": "http", "collection": self.collection_name, "remote": resp.json().get("result")}
            except Exception:
                pass
            return {"mode": "http", "collection": self.collection_name}

        conn = self._get_embedded_conn()
        rows = conn.execute("SELECT key, value FROM store_meta").fetchall()
        meta = {str(r[0]): str(r[1]) for r in rows}
        return {
            "mode": "embedded",
            "store_dir": str(self.store_dir),
            "collection_name": self.collection_name,
            "vectors_count": int(meta.get("vectors_count") or 0),
            "dense_dim": int(meta.get("dense_dim") or self.dense_dim),
            "dense_quantization": meta.get("dense_quantization", "unknown"),
            "sparse_terms_count": int(meta.get("sparse_terms_count") or 0),
            "colbert_count": int(meta.get("colbert_count") or 0),
        }

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        with self._lock:
            for conn in self._conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._conns.clear()
        if hasattr(self._local, "conn"):
            delattr(self._local, "conn")
        self._faiss_index = None
        self._sparse_blob_cache.clear()
        self._filter_ids_cache.clear()
        self._max_pid_cache = None


class UnifiedHybridRetriever:
    """Drop-in FinaRAG retriever backed by QdrantHybridStore + SqliteDocumentRepository."""

    def __init__(
        self,
        store: QdrantHybridStore,
        doc_repo: SqliteDocumentRepository,
        *,
        dense_encoder: Optional[Callable[[str], np.ndarray]] = None,
        sparse_encoder: Optional[Callable[[str], Dict[str, float]]] = None,
        colbert_encoder: Optional[Callable[[str], np.ndarray]] = None,
        reranker: Any = None,
        reranking_strategy: str = "single",
        final_reranking_backend: str = "llm_prompt",
        lazy_load_encoders: bool = False,
    ):
        self.store = store
        self.doc_repo = doc_repo
        self._dense_encoder = dense_encoder
        self._sparse_encoder = sparse_encoder
        self._colbert_encoder = colbert_encoder
        self._lazy_load_encoders = lazy_load_encoders
        self.reranker = reranker
        self.reranking_strategy = reranking_strategy
        self.final_reranking_backend = final_reranking_backend
        self.colbert_top_n = 10
        self.last_rerank_debug: Dict[str, Any] = {}
        self._embedding_backend = None
        self._sparse_backend = None

    def _encode_query(self, query: str) -> Tuple[Any, Optional[Dict[str, float]], Any]:
        dense_vec = self._dense_encoder(query) if self._dense_encoder is not None else None
        sparse_vec = self._sparse_encoder(query) if self._sparse_encoder is not None else None
        colbert_vec = self._colbert_encoder(query) if self._colbert_encoder is not None else None
        if dense_vec is None and self._lazy_load_encoders:
            try:
                if self._embedding_backend is None:
                    from src.embedding_backend import EmbeddingBackend

                    self._embedding_backend = EmbeddingBackend()
                dense_vec = self._embedding_backend.embed_query(query)
            except Exception:
                dense_vec = None
        if sparse_vec is None and self._lazy_load_encoders:
            try:
                if self._sparse_backend is None:
                    from src.embedding_backend import BGEM3SparseEmbeddingBackend

                    self._sparse_backend = BGEM3SparseEmbeddingBackend()
                encoded = self._sparse_backend.encode_texts([query])
                sparse_vec = encoded[0] if encoded else None
            except Exception:
                sparse_vec = None
        return dense_vec, sparse_vec, colbert_vec

    @staticmethod
    def _to_hybrid_filter(
        company_name: Optional[str],
        filters: Optional[RetrievalFilters],
        candidate_doc_ids: Optional[List[str]],
    ) -> Optional[HybridSearchFilter]:
        doc_ids = list(candidate_doc_ids or [])
        if filters and getattr(filters, "candidate_doc_ids", None) and not doc_ids:
            doc_ids = list(filters.candidate_doc_ids)
        comp = company_name or (filters.company_name if filters else None)
        report_year = getattr(filters, "report_year", getattr(filters, "year", None)) if filters else None
        industry_l1 = filters.industry_l1 if filters else None
        section_name = filters.section_name if filters else None
        topic_flags = getattr(filters, "required_topic_flags", getattr(filters, "topic_flags", None)) if filters else None
        strategy_tags = filters.strategy_tags if filters else None
        if not any([doc_ids, comp, report_year, industry_l1, section_name, topic_flags, strategy_tags]):
            return None
        return HybridSearchFilter(
            doc_ids=doc_ids or None,
            company_name=comp or None,
            report_year=report_year,
            industry_l1=industry_l1,
            section_name=section_name,
            topic_flags=topic_flags,
            strategy_tags=strategy_tags,
        )

    def retrieve_candidates_by_company_name(
        self,
        company_name: str,
        query: str,
        top_n: int = 10,
        parent_retrieval_mode: str = "child",
        filters: Optional[RetrievalFilters] = None,
        candidate_doc_ids: Optional[List[str]] = None,
        backend_scope: str = "all",
    ) -> List[Dict]:
        if not str(query or "").strip():
            return []
        mode = _normalize_parent_retrieval_mode(parent_retrieval_mode)
        hybrid_filter = self._to_hybrid_filter(company_name, filters, candidate_doc_ids)
        dense_vec, sparse_vec, colbert_vec = self._encode_query(query)
        if backend_scope == "vector_only":
            sparse_vec = None
        hits = self.store.search_hybrid(
            dense_vector=dense_vec,
            query_text="" if backend_scope == "vector_only" else query,
            bge_sparse=sparse_vec,
            colbert_vectors=colbert_vec,
            filters=hybrid_filter,
            top_k=max(top_n * 3, 20) if mode in {"page", "block"} else top_n,
            prefetch_limit=max(top_n * 4, 40),
        )
        return self._expand_hits(hits, mode=mode, top_n=top_n, filters=filters)

    def _expand_hits(
        self,
        hits: List[HybridSearchHit],
        *,
        mode: str,
        top_n: int,
        filters: Optional[RetrievalFilters],
    ) -> List[Dict]:
        results: List[Dict[str, Any]] = []
        seen_pages = set()
        aggregated_parents: Dict[Tuple[str, int], Dict[str, Any]] = {}
        ordered_parent_keys: List[Tuple[str, int]] = []

        for hit in hits:
            metainfo = self.doc_repo.get_metainfo(hit.doc_id) or {
                "sha1_name": hit.doc_id,
                "doc_id": hit.doc_id,
                "company_name": hit.metadata.get("company_name"),
                "report_year": hit.metadata.get("report_year"),
            }
            chunk = self.doc_repo.get_chunk_by_index(hit.doc_id, hit.chunk_idx) or {
                "page": hit.page,
                "text": hit.text,
                "chunk_id": hit.chunk_idx,
                "node_type": hit.metadata.get("node_type", "child"),
                "section_name": hit.metadata.get("section_name"),
            }

            if mode == "page":
                page_key = (hit.doc_id, int(hit.page))
                if page_key in seen_pages:
                    continue
                seen_pages.add(page_key)
                page_entry = self.doc_repo.get_page(hit.doc_id, int(hit.page))
                page_text = page_entry["text"] if page_entry is not None else hit.text
                res = _build_retrieval_result(
                    metainfo,
                    chunk,
                    page_text,
                    int(hit.page),
                    hit.score,
                    "unified_hybrid",
                    result_scope="page",
                )
                res["combined_score"] = round(float(hit.score), 4)
                res["retrieval_sources"] = list(hit.retrieval_sources)
                results.append(res)
                if len(results) >= top_n:
                    break
            elif mode == "block":
                parent_id = chunk.get("parent_chunk_id")
                parent_chunk = (
                    self.doc_repo.get_parent_chunk(hit.doc_id, int(parent_id))
                    if parent_id is not None
                    else None
                )
                if parent_chunk is None:
                    res = _build_retrieval_result(
                        metainfo,
                        chunk,
                        hit.text,
                        int(hit.page),
                        hit.score,
                        "unified_hybrid",
                        result_scope="child",
                    )
                    res["combined_score"] = round(float(hit.score), 4)
                    res["retrieval_sources"] = list(hit.retrieval_sources)
                    results.append(res)
                    if len(results) >= top_n:
                        break
                    continue
                pkey = (hit.doc_id, int(parent_id))
                child_cid = chunk.get("chunk_id", chunk.get("id"))
                if pkey not in aggregated_parents:
                    if len(ordered_parent_keys) >= top_n:
                        continue
                    parent_payload = dict(parent_chunk)
                    parent_payload["matched_child_chunk_ids"] = [child_cid] if child_cid is not None else []
                    res = _build_retrieval_result(
                        metainfo,
                        parent_payload,
                        parent_chunk["text"],
                        int(parent_chunk["page"]),
                        hit.score,
                        "unified_hybrid",
                        result_scope="parent",
                    )
                    res["combined_score"] = round(float(hit.score), 4)
                    res["retrieval_sources"] = list(hit.retrieval_sources)
                    aggregated_parents[pkey] = res
                    ordered_parent_keys.append(pkey)
                elif child_cid is not None:
                    ids = aggregated_parents[pkey].setdefault("matched_child_chunk_ids", [])
                    if child_cid not in ids:
                        ids.append(child_cid)
            else:
                res = _build_retrieval_result(
                    metainfo,
                    chunk,
                    chunk.get("text") or hit.text,
                    int(chunk.get("page") or hit.page),
                    hit.score,
                    "unified_hybrid",
                    result_scope="child",
                )
                res["combined_score"] = round(float(hit.score), 4)
                res["retrieval_sources"] = list(hit.retrieval_sources)
                results.append(res)
                if len(results) >= top_n:
                    break

        if mode == "block" and ordered_parent_keys:
            results = [aggregated_parents[k] for k in ordered_parent_keys[:top_n]]
        return apply_retrieval_filters(results, filters)

    def rerank_candidates(
        self,
        query: str,
        candidate_results: List[Dict],
        documents_batch_size: int = 2,
        top_n: int = 6,
        llm_weight: float = 0.7,
        task_type: Optional[str] = None,
    ) -> List[Dict]:
        if not candidate_results:
            self.last_rerank_debug = {
                "reranking_strategy": self.reranking_strategy,
                "initial_candidate_pool_size": 0,
                "colbert_candidate_pool_size": None,
                "colbert_top_n": None,
                "final_reranking_backend": self.final_reranking_backend,
            }
            return []
        if self.reranker is not None:
            reranked = self.reranker.rerank_documents(
                query=query,
                documents=candidate_results,
                documents_batch_size=documents_batch_size,
                llm_weight=llm_weight,
            )
            self.last_rerank_debug = dict(getattr(self.reranker, "last_debug", {}) or {})
            return reranked[:top_n]
        self.last_rerank_debug = {
            "reranking_strategy": self.reranking_strategy,
            "initial_candidate_pool_size": len(candidate_results),
            "colbert_candidate_pool_size": None,
            "colbert_top_n": None,
            "final_reranking_backend": "unified_rrf",
        }
        return candidate_results[:top_n]

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
        task_type: Optional[str] = None,
    ) -> List[Dict]:
        sample_n = max(int(llm_reranking_sample_size or top_n), top_n)
        candidates = self.retrieve_candidates_by_company_name(
            company_name=company_name,
            query=query,
            top_n=sample_n,
            parent_retrieval_mode=parent_retrieval_mode,
            filters=filters,
            candidate_doc_ids=candidate_doc_ids,
        )
        return self.rerank_candidates(
            query=query,
            candidate_results=candidates,
            documents_batch_size=documents_batch_size,
            top_n=top_n,
            llm_weight=llm_weight,
            task_type=task_type,
        )

    def retrieve_all(
        self,
        company_name: str,
        filters: Optional[RetrievalFilters] = None,
        candidate_doc_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        target_ids = list(candidate_doc_ids or [])
        if not target_ids:
            for meta in self.doc_repo.list_metainfo():
                if meta.get("company_name") == company_name:
                    did = str(meta.get("sha1_name") or meta.get("doc_id") or "")
                    if did:
                        target_ids.append(did)
        all_pages: List[Dict[str, Any]] = []
        for doc_id in target_ids:
            metainfo = self.doc_repo.get_metainfo(doc_id)
            if metainfo is None:
                continue
            for page in self.doc_repo.get_pages(doc_id):
                all_pages.append(
                    _build_retrieval_result(
                        metainfo,
                        {"page": page["page"], "chunk_type": "page", "node_type": "page"},
                        page["text"],
                        int(page["page"]),
                        0.5,
                        "unified_full_context",
                        result_scope="page",
                    )
                )
        return apply_retrieval_filters(all_pages, filters)
