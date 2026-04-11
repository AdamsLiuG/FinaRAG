from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

_log = logging.getLogger(__name__)


DocumentEntry = Dict[str, Any]


@dataclass(frozen=True)
class ChunkedDocumentStore:
    documents_dir: Path
    documents: Tuple[DocumentEntry, ...]
    metainfo_by_doc_id: Dict[str, Dict[str, Any]]


def _normalize_documents_dir(documents_dir: Path | str) -> Path:
    return Path(documents_dir).expanduser().resolve(strict=False)


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


def get_document_store(documents_dir: Path | str) -> ChunkedDocumentStore:
    return _load_document_store(*_document_store_cache_key(documents_dir))


def clear_document_store_cache() -> None:
    _load_document_store.cache_clear()


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
