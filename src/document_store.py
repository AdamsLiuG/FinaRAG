from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_log = logging.getLogger(__name__)


DocumentEntry = Dict[str, Any]


@dataclass(frozen=True)
class ChunkedDocumentStore:
    documents_dir: Path
    documents: Tuple[DocumentEntry, ...]
    metainfo_by_doc_id: Dict[str, Dict[str, Any]]


def _normalize_documents_dir(documents_dir: Path | str) -> Path:
    return Path(documents_dir).expanduser().resolve(strict=False)


def resolve_doc_sqlite_path(documents_dir: Path | str) -> Optional[Path]:
    explicit = os.getenv("FINARAG_DOC_SQLITE_PATH", "").strip()
    if explicit:
        p = Path(explicit).expanduser().resolve(strict=False)
        return p if p.exists() else None
    norm = _normalize_documents_dir(documents_dir)
    candidate = norm.parent / "finarag_docs.sqlite"
    if candidate.exists():
        return candidate
    return None


@lru_cache(maxsize=4)
def _load_sqlite_repo_cached(db_path_str: str, _mtime_ns: int):
    from src.storage.sqlite_doc_store import SqliteDocumentRepository

    return SqliteDocumentRepository(Path(db_path_str))


def get_sqlite_doc_repository(documents_dir: Path | str):
    backend = os.getenv("DOCUMENT_STORE_BACKEND", "memory").strip().lower()
    if backend != "sqlite":
        return None
    db_path = resolve_doc_sqlite_path(documents_dir)
    if db_path is None:
        return None
    try:
        mtime_ns = db_path.stat().st_mtime_ns
    except OSError:
        return None
    return _load_sqlite_repo_cached(str(db_path), mtime_ns)


@lru_cache(maxsize=4)
def _load_sqlite_document_store(documents_dir_str: str, db_path_str: str, _mtime_ns: int):
    from src.storage.sqlite_doc_store import SqliteChunkedDocumentStore

    repo = _load_sqlite_repo_cached(db_path_str, _mtime_ns)
    return SqliteChunkedDocumentStore(Path(documents_dir_str), repo)


def _document_store_cache_key(documents_dir: Path | str) -> Tuple[str, Tuple[str, ...], int, int]:
    normalized_dir = _normalize_documents_dir(documents_dir)
    if not normalized_dir.exists():
        return str(normalized_dir), tuple(), 0, 0

    names = []
    latest_mtime_ns = 0
    total_size = 0
    for document_path in normalized_dir.glob("*.json"):
        names.append(document_path.name)
        try:
            stat_result = document_path.stat()
        except OSError:
            continue
        latest_mtime_ns = max(latest_mtime_ns, stat_result.st_mtime_ns)
        total_size += stat_result.st_size
    return str(normalized_dir), tuple(sorted(names)), latest_mtime_ns, total_size


def get_document_store(documents_dir: Path | str):
    if os.getenv("DOCUMENT_STORE_BACKEND", "memory").strip().lower() == "sqlite":
        db_path = resolve_doc_sqlite_path(documents_dir)
        if db_path is not None:
            try:
                mtime_ns = db_path.stat().st_mtime_ns
                return _load_sqlite_document_store(
                    str(_normalize_documents_dir(documents_dir)),
                    str(db_path),
                    mtime_ns,
                )
            except Exception as exc:
                _log.warning("Falling back to memory document store: %s", exc)
    return _load_document_store(*_document_store_cache_key(documents_dir))


def clear_document_store_cache() -> None:
    _load_document_store.cache_clear()
    _load_sqlite_document_store.cache_clear()
    _load_sqlite_repo_cached.cache_clear()


@lru_cache(maxsize=2)
def _load_document_store(
    normalized_dir: str,
    _document_names: Tuple[str, ...],
    _latest_mtime_ns: int,
    _total_size: int,
) -> ChunkedDocumentStore:
    documents_dir = Path(normalized_dir)
    if not documents_dir.exists():
        return ChunkedDocumentStore(documents_dir=documents_dir, documents=tuple(), metainfo_by_doc_id={})

    loaded_documents = []
    metainfo_by_doc_id: Dict[str, Dict[str, Any]] = {}

    for document_path in sorted(documents_dir.glob("*.json")):
        try:
            with open(document_path, "r", encoding="utf-8") as file:
                document = json.load(file)
        except Exception as exc:
            _log.error(f"Error loading JSON from {document_path.name}: {exc}")
            continue

        if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
            _log.warning(f"Skipping {document_path.name}: does not match the expected schema.")
            continue

        entry = {
            "name": document_path.stem,
            "path": document_path,
            "document": document,
        }
        loaded_documents.append(entry)

        metainfo = document.get("metainfo") or {}
        doc_id = metainfo.get("sha1_name") or metainfo.get("doc_id") or document_path.stem
        if doc_id:
            metainfo_by_doc_id[str(doc_id)] = metainfo

    return ChunkedDocumentStore(
        documents_dir=documents_dir,
        documents=tuple(loaded_documents),
        metainfo_by_doc_id=metainfo_by_doc_id,
    )
