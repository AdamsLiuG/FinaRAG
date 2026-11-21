"""On-demand SQLite WAL document repository replacing full-memory JSON loading."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from src.storage.base import StorageError, validate_safe_path

_MISSING = object()


class _BoundedLRU:
    """Thread-safe fixed-capacity LRU cache with hit/miss/eviction counters."""

    def __init__(self, max_size: int):
        self.max_size = max(1, int(max_size))
        self._data: "OrderedDict[Any, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: Any, default: Any = _MISSING) -> Any:
        with self._lock:
            val = self._data.get(key, _MISSING)
            if val is _MISSING:
                self.misses += 1
                return default
            self._data.move_to_end(key)
            self.hits += 1
            return val

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = value
                return
            self._data[key] = value
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)
                self.evictions += 1

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "size": len(self._data),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class SqliteDocumentRepository:
    """Thread-safe read-only SQLite document repository with bounded LRU caches."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        allowed_base_dir: Optional[Path | str] = None,
        page_cache_size: int = 256,
        chunk_cache_size: int = 512,
        doc_cache_size: int = 8,
    ):
        self.db_path = validate_safe_path(db_path, allowed_base_dir)
        self._page_cache = _BoundedLRU(page_cache_size)
        self._chunk_cache = _BoundedLRU(chunk_cache_size)
        self._doc_cache = _BoundedLRU(doc_cache_size)
        self._metainfo_lock = threading.Lock()
        self._metainfo_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._stem_map_cache: Optional[Dict[str, str]] = None
        self._local = threading.local()
        self._conns: List[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        if not self.db_path.exists():
            raise StorageError("Document SQLite database does not exist")
        try:
            uri = f"{self.db_path.as_uri()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
        except sqlite3.Error as exc:
            raise StorageError("Failed to open document SQLite repository") from exc
        self._local.conn = conn
        with self._conns_lock:
            self._conns.append(conn)
        return conn

    def _query_all(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        try:
            cur = self._get_conn().execute(sql, tuple(params))
            return cur.fetchall()
        except sqlite3.Error as exc:
            raise StorageError("Document repository query failed") from exc

    def _query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        try:
            cur = self._get_conn().execute(sql, tuple(params))
            return cur.fetchone()
        except sqlite3.Error as exc:
            raise StorageError("Document repository query failed") from exc

    def close(self) -> None:
        with self._conns_lock:
            for conn in self._conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._conns.clear()
        if hasattr(self._local, "conn"):
            delattr(self._local, "conn")

    def _ensure_metainfo_loaded(self) -> None:
        if self._metainfo_cache is not None and self._stem_map_cache is not None:
            return
        with self._metainfo_lock:
            if self._metainfo_cache is not None and self._stem_map_cache is not None:
                return
            rows = self._query_all("SELECT doc_id, stem, metainfo_json FROM documents ORDER BY stem ASC")
            meta_by_id: Dict[str, Dict[str, Any]] = {}
            stem_to_id: Dict[str, str] = {}
            for row in rows:
                doc_id = str(row["doc_id"])
                stem = str(row["stem"] or doc_id)
                meta_by_id[doc_id] = json.loads(row["metainfo_json"])
                stem_to_id[stem] = doc_id
            self._metainfo_cache = meta_by_id
            self._stem_map_cache = stem_to_id

    def resolve_doc_id(self, doc_or_stem: str) -> str:
        key = str(doc_or_stem or "").strip()
        if key.endswith(".json"):
            key = key[:-5]
        self._ensure_metainfo_loaded()
        assert self._metainfo_cache is not None and self._stem_map_cache is not None
        if key in self._metainfo_cache:
            return key
        return self._stem_map_cache.get(key, key)

    def list_documents_index(self) -> List[Tuple[str, str]]:
        self._ensure_metainfo_loaded()
        assert self._stem_map_cache is not None
        return sorted((stem, doc_id) for stem, doc_id in self._stem_map_cache.items())

    def list_metainfo(self) -> List[Dict[str, Any]]:
        self._ensure_metainfo_loaded()
        assert self._metainfo_cache is not None
        return [dict(v) for v in self._metainfo_cache.values()]

    def get_metainfo_map(self) -> Dict[str, Dict[str, Any]]:
        self._ensure_metainfo_loaded()
        assert self._metainfo_cache is not None
        return {k: dict(v) for k, v in self._metainfo_cache.items()}

    def get_metainfo(self, doc_id: str) -> Optional[Dict[str, Any]]:
        canonical = self.resolve_doc_id(doc_id)
        self._ensure_metainfo_loaded()
        assert self._metainfo_cache is not None
        meta = self._metainfo_cache.get(canonical)
        return dict(meta) if meta is not None else None

    def get_page(self, doc_id: str, page: int) -> Optional[Dict[str, Any]]:
        canonical = self.resolve_doc_id(doc_id)
        cache_key = (canonical, int(page))
        cached = self._page_cache.get(cache_key)
        if cached is not _MISSING:
            return dict(cached) if cached is not None else None
        row = self._query_one(
            "SELECT page, text FROM pages WHERE doc_id = ? AND page = ?",
            (canonical, int(page)),
        )
        result = {"page": int(row["page"]), "text": row["text"]} if row is not None else None
        self._page_cache.put(cache_key, result)
        return dict(result) if result is not None else None

    def get_pages(self, doc_id: str) -> List[Dict[str, Any]]:
        canonical = self.resolve_doc_id(doc_id)
        rows = self._query_all(
            "SELECT page, text FROM pages WHERE doc_id = ? ORDER BY page ASC",
            (canonical,),
        )
        pages = [{"page": int(r["page"]), "text": r["text"]} for r in rows]
        for p in pages:
            self._page_cache.put((canonical, p["page"]), p)
        return pages

    def get_chunk_by_index(self, doc_id: str, chunk_idx: int) -> Optional[Dict[str, Any]]:
        canonical = self.resolve_doc_id(doc_id)
        cache_key = (canonical, int(chunk_idx))
        cached = self._chunk_cache.get(cache_key)
        if cached is not _MISSING:
            return dict(cached) if cached is not None else None
        row = self._query_one(
            "SELECT chunk_json FROM chunks WHERE doc_id = ? AND chunk_idx = ?",
            (canonical, int(chunk_idx)),
        )
        result = json.loads(row["chunk_json"]) if row is not None else None
        self._chunk_cache.put(cache_key, result)
        return dict(result) if result is not None else None

    def get_chunks_by_indices(self, doc_id: str, indices: Sequence[int]) -> List[Dict[str, Any]]:
        canonical = self.resolve_doc_id(doc_id)
        clean_indices = [int(i) for i in indices]
        if not clean_indices:
            return []
        found: Dict[int, Dict[str, Any]] = {}
        missing: List[int] = []
        for idx in clean_indices:
            cached = self._chunk_cache.get((canonical, idx))
            if cached is not _MISSING and cached is not None:
                found[idx] = dict(cached)
            else:
                missing.append(idx)
        if missing:
            placeholders = ",".join("?" for _ in missing)
            rows = self._query_all(
                f"SELECT chunk_idx, chunk_json FROM chunks WHERE doc_id = ? AND chunk_idx IN ({placeholders})",
                [canonical, *missing],
            )
            for row in rows:
                idx = int(row["chunk_idx"])
                payload = json.loads(row["chunk_json"])
                self._chunk_cache.put((canonical, idx), payload)
                found[idx] = dict(payload)
        return [found[idx] for idx in clean_indices if idx in found]

    def get_all_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        canonical = self.resolve_doc_id(doc_id)
        rows = self._query_all(
            "SELECT chunk_idx, chunk_json FROM chunks WHERE doc_id = ? ORDER BY chunk_idx ASC",
            (canonical,),
        )
        return [json.loads(r["chunk_json"]) for r in rows]

    def get_parent_chunk(self, doc_id: str, parent_chunk_id: int) -> Optional[Dict[str, Any]]:
        canonical = self.resolve_doc_id(doc_id)
        row = self._query_one(
            "SELECT chunk_json FROM parent_chunks WHERE doc_id = ? AND parent_chunk_id = ?",
            (canonical, int(parent_chunk_id)),
        )
        return json.loads(row["chunk_json"]) if row is not None else None

    def get_parent_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        canonical = self.resolve_doc_id(doc_id)
        rows = self._query_all(
            "SELECT chunk_json FROM parent_chunks WHERE doc_id = ? ORDER BY parent_chunk_id ASC",
            (canonical,),
        )
        return [json.loads(r["chunk_json"]) for r in rows]

    def get_tables(self, doc_id: str, page: Optional[int] = None) -> List[Dict[str, Any]]:
        canonical = self.resolve_doc_id(doc_id)
        if page is not None:
            rows = self._query_all(
                "SELECT table_json FROM structured_tables WHERE doc_id = ? AND page = ? ORDER BY table_idx ASC",
                (canonical, int(page)),
            )
        else:
            rows = self._query_all(
                "SELECT table_json FROM structured_tables WHERE doc_id = ? ORDER BY table_idx ASC",
                (canonical,),
            )
        return [json.loads(r["table_json"]) for r in rows]

    def get_document_payload(self, doc_id: str) -> Optional[Dict[str, Any]]:
        canonical = self.resolve_doc_id(doc_id)
        cached = self._doc_cache.get(canonical)
        if cached is not _MISSING:
            return cached
        metainfo = self.get_metainfo(canonical)
        if metainfo is None:
            return None
        payload = {
            "metainfo": metainfo,
            "content": {
                "pages": self.get_pages(canonical),
                "chunks": self.get_all_chunks(canonical),
                "parent_chunks": self.get_parent_chunks(canonical),
                "structured_tables": self.get_tables(canonical),
            },
        }
        self._doc_cache.put(canonical, payload)
        return payload

    def get_corpus_version(self) -> str:
        row = self._query_one("SELECT value FROM store_meta WHERE key = ?", ("corpus_version",))
        return str(row["value"]) if row is not None else "unknown"

    def stats(self) -> Dict[str, Any]:
        meta_rows = self._query_all("SELECT key, value FROM store_meta")
        meta = {str(r["key"]): str(r["value"]) for r in meta_rows}
        return {
            "db_path": str(self.db_path),
            "corpus_version": meta.get("corpus_version", "unknown"),
            "reports_count": int(meta.get("reports_count") or 0),
            "pages_count": int(meta.get("pages_count") or 0),
            "chunks_count": int(meta.get("chunks_count") or 0),
            "tables_count": int(meta.get("tables_count") or 0),
            "page_cache": self._page_cache.stats(),
            "chunk_cache": self._chunk_cache.stats(),
            "doc_cache": self._doc_cache.stats(),
        }

    @staticmethod
    def build_from_chunked_dir(
        chunked_reports_dir: Path | str,
        db_path: Path | str,
        *,
        limit: Optional[int] = None,
        allowed_base_dir: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        reports_dir = Path(chunked_reports_dir).expanduser().resolve(strict=False)
        out_path = validate_safe_path(db_path, allowed_base_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f".{out_path.name}.tmp")
        for suffix in ("", "-wal", "-shm"):
            p = Path(f"{tmp_path}{suffix}")
            if p.exists():
                p.unlink()

        files = sorted(reports_dir.glob("*.json"))
        if limit is not None:
            files = files[: max(0, int(limit))]

        conn = sqlite3.connect(str(tmp_path))
        reports_count = 0
        pages_count = 0
        chunks_count = 0
        parent_chunks_count = 0
        tables_count = 0
        version_parts: List[str] = []

        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS store_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id        TEXT PRIMARY KEY,
                    stem          TEXT,
                    company_name  TEXT,
                    stock_code    TEXT,
                    report_year   INTEGER,
                    currency      TEXT,
                    industry_l1   TEXT,
                    pages_count   INTEGER,
                    chunks_count  INTEGER,
                    metainfo_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pages (
                    doc_id TEXT NOT NULL,
                    page   INTEGER NOT NULL,
                    text   TEXT NOT NULL,
                    PRIMARY KEY (doc_id, page)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    doc_id          TEXT NOT NULL,
                    chunk_idx       INTEGER NOT NULL,
                    chunk_id        INTEGER,
                    page            INTEGER,
                    chunk_type      TEXT,
                    node_type       TEXT,
                    parent_chunk_id INTEGER,
                    section_name    TEXT,
                    table_id        TEXT,
                    text            TEXT,
                    chunk_json      TEXT NOT NULL,
                    PRIMARY KEY (doc_id, chunk_idx)
                );
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    doc_id          TEXT NOT NULL,
                    parent_chunk_id INTEGER NOT NULL,
                    page            INTEGER,
                    text            TEXT,
                    chunk_json      TEXT NOT NULL,
                    PRIMARY KEY (doc_id, parent_chunk_id)
                );
                CREATE TABLE IF NOT EXISTS structured_tables (
                    doc_id     TEXT NOT NULL,
                    table_idx  INTEGER NOT NULL,
                    table_id   TEXT,
                    page       INTEGER,
                    table_json TEXT NOT NULL,
                    PRIMARY KEY (doc_id, table_idx)
                );
                """
            )

            for file_path in files:
                with open(file_path, "r", encoding="utf-8") as fh:
                    document = json.load(fh)
                if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                    continue
                metainfo = document.get("metainfo") or {}
                content = document.get("content") or {}
                doc_id = str(metainfo.get("sha1_name") or metainfo.get("doc_id") or file_path.stem)
                stem = file_path.stem

                pages = list(content.get("pages") or [])
                chunks = list(content.get("chunks") or [])
                parents = list(content.get("parent_chunks") or [])
                tables = list(content.get("structured_tables") or [])

                raw_year = metainfo.get("report_year") or metainfo.get("fiscal_year")
                try:
                    report_year = int(raw_year) if raw_year is not None else None
                except (TypeError, ValueError):
                    report_year = None

                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents (
                        doc_id, stem, company_name, stock_code, report_year,
                        currency, industry_l1, pages_count, chunks_count, metainfo_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        stem,
                        str(metainfo.get("company_name") or ""),
                        str(metainfo.get("stock_code") or metainfo.get("security_code") or ""),
                        report_year,
                        str(metainfo.get("currency") or ""),
                        str(metainfo.get("industry_l1") or metainfo.get("major_industry") or ""),
                        len(pages),
                        len(chunks),
                        json.dumps(metainfo, ensure_ascii=False),
                    ),
                )

                page_rows = [
                    (doc_id, int(p["page"]), str(p.get("text") or ""))
                    for p in pages
                    if isinstance(p, dict) and p.get("page") is not None
                ]
                conn.executemany(
                    "INSERT OR REPLACE INTO pages (doc_id, page, text) VALUES (?, ?, ?)",
                    page_rows,
                )

                chunk_rows = []
                for idx, chunk in enumerate(chunks):
                    if not isinstance(chunk, dict):
                        continue
                    cid = chunk.get("chunk_id", chunk.get("id"))
                    parent_cid = chunk.get("parent_chunk_id")
                    page_val = chunk.get("page")
                    chunk_rows.append(
                        (
                            doc_id,
                            idx,
                            int(cid) if isinstance(cid, int) else None,
                            int(page_val) if isinstance(page_val, int) else None,
                            str(chunk.get("chunk_type") or ""),
                            str(chunk.get("node_type") or "child"),
                            int(parent_cid) if isinstance(parent_cid, int) else None,
                            str(chunk.get("section_name") or chunk.get("section_title") or ""),
                            str(chunk.get("table_id") or "") if chunk.get("table_id") is not None else None,
                            str(chunk.get("text") or ""),
                            json.dumps(chunk, ensure_ascii=False),
                        )
                    )
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO chunks (
                        doc_id, chunk_idx, chunk_id, page, chunk_type, node_type,
                        parent_chunk_id, section_name, table_id, text, chunk_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    chunk_rows,
                )

                parent_rows = []
                for p_idx, p_chunk in enumerate(parents):
                    if not isinstance(p_chunk, dict):
                        continue
                    pcid = p_chunk.get("chunk_id", p_chunk.get("id", p_idx))
                    page_val = p_chunk.get("page")
                    parent_rows.append(
                        (
                            doc_id,
                            int(pcid),
                            int(page_val) if isinstance(page_val, int) else None,
                            str(p_chunk.get("text") or ""),
                            json.dumps(p_chunk, ensure_ascii=False),
                        )
                    )
                if parent_rows:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO parent_chunks (
                            doc_id, parent_chunk_id, page, text, chunk_json
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        parent_rows,
                    )

                table_rows = []
                for t_idx, table in enumerate(tables):
                    if not isinstance(table, dict):
                        continue
                    page_val = table.get("page")
                    table_rows.append(
                        (
                            doc_id,
                            t_idx,
                            str(table.get("table_id") or "") if table.get("table_id") is not None else None,
                            int(page_val) if isinstance(page_val, int) else None,
                            json.dumps(table, ensure_ascii=False),
                        )
                    )
                if table_rows:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO structured_tables (
                            doc_id, table_idx, table_id, page, table_json
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        table_rows,
                    )

                reports_count += 1
                pages_count += len(page_rows)
                chunks_count += len(chunk_rows)
                parent_chunks_count += len(parent_rows)
                tables_count += len(table_rows)
                version_parts.append(f"{doc_id}:{len(page_rows)}:{len(chunk_rows)}")

            corpus_version = hashlib.sha256("\n".join(sorted(version_parts)).encode("utf-8")).hexdigest()[:16]
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_docs_company ON documents(company_name, report_year);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks(doc_id, page);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_chunkid ON chunks(doc_id, chunk_id);
                CREATE INDEX IF NOT EXISTS idx_tables_doc_page ON structured_tables(doc_id, page);
                """
            )
            conn.executemany(
                "INSERT OR REPLACE INTO store_meta (key, value) VALUES (?, ?)",
                [
                    ("corpus_version", corpus_version),
                    ("reports_count", str(reports_count)),
                    ("pages_count", str(pages_count)),
                    ("chunks_count", str(chunks_count)),
                    ("parent_chunks_count", str(parent_chunks_count)),
                    ("tables_count", str(tables_count)),
                ],
            )
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()

        os.replace(tmp_path, out_path)
        for suffix in ("-wal", "-shm"):
            leftover = Path(f"{tmp_path}{suffix}")
            if leftover.exists():
                leftover.unlink()

        return {
            "db_path": str(out_path),
            "corpus_version": corpus_version,
            "reports_count": reports_count,
            "pages_count": pages_count,
            "chunks_count": chunks_count,
            "parent_chunks_count": parent_chunks_count,
            "tables_count": tables_count,
        }


class LazySqliteContentProxy(Mapping):
    """Mapping view over a document's content backed by on-demand SQLite queries."""

    _KEYS = ("pages", "chunks", "parent_chunks", "structured_tables")

    def __init__(self, repo: SqliteDocumentRepository, doc_id: str):
        self._repo = repo
        self._doc_id = doc_id

    def __getitem__(self, key: str) -> Any:
        if key == "pages":
            return self._repo.get_pages(self._doc_id)
        if key == "chunks":
            return self._repo.get_all_chunks(self._doc_id)
        if key == "parent_chunks":
            return self._repo.get_parent_chunks(self._doc_id)
        if key == "structured_tables":
            return self._repo.get_tables(self._doc_id)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)


class LazySqliteDocumentProxy(Mapping):
    """Mapping proxy compatible with {'metainfo': ..., 'content': ...} document entries."""

    def __init__(self, repo: SqliteDocumentRepository, doc_id: str, metainfo: Dict[str, Any]):
        self._repo = repo
        self._doc_id = doc_id
        self._metainfo = metainfo
        self._content: Optional[LazySqliteContentProxy] = None

    def __getitem__(self, key: str) -> Any:
        if key == "metainfo":
            return self._metainfo
        if key == "content":
            if self._content is None:
                self._content = LazySqliteContentProxy(self._repo, self._doc_id)
            return self._content
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("metainfo", "content"))

    def __len__(self) -> int:
        return 2


class SqliteChunkedDocumentStore:
    """Drop-in replacement for ChunkedDocumentStore using on-demand SQLite reads."""

    def __init__(self, documents_dir: Path, repo: SqliteDocumentRepository):
        self.documents_dir = documents_dir
        self.repo = repo
        self.metainfo_by_doc_id = repo.get_metainfo_map()
        entries = []
        for stem, doc_id in repo.list_documents_index():
            meta = self.metainfo_by_doc_id.get(doc_id, {})
            entries.append(
                {
                    "name": stem,
                    "path": documents_dir / f"{stem}.json",
                    "document": LazySqliteDocumentProxy(repo, doc_id, meta),
                }
            )
        self.documents = tuple(entries)
