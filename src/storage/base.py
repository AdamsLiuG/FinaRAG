"""Abstract storage protocols, shared dataclasses, and security path validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable


class StorageSecurityError(ValueError):
    """Raised when a storage operation violates path or access security constraints."""


class StorageError(RuntimeError):
    """Raised when a storage operation fails without leaking internal storage details."""


def validate_safe_path(
    target_path: Path | str,
    allowed_base_dir: Optional[Path | str],
) -> Path:
    """Resolve a path and enforce strict directory boundary isolation."""
    raw = Path(target_path)
    if ".." in raw.parts:
        raise StorageSecurityError("Path traversal sequence '..' is not allowed")
    resolved = raw.expanduser().resolve(strict=False)
    if allowed_base_dir is not None:
        base_resolved = Path(allowed_base_dir).expanduser().resolve(strict=False)
        if not str(resolved).startswith(str(base_resolved) + os.sep):
            raise StorageSecurityError("Target path is outside allowed_base_dir")
    return resolved


@dataclass(frozen=True)
class HybridSearchFilter:
    """Metadata payload filter applied during unified hybrid search."""

    doc_ids: Optional[Sequence[str]] = None
    company_name: Optional[str] = None
    report_year: Optional[int] = None
    industry_l1: Optional[str] = None
    node_type: Optional[str] = None
    section_name: Optional[str] = None
    topic_flags: Optional[Sequence[str]] = None
    strategy_tags: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class HybridSearchHit:
    """Single ranked hit returned from a unified hybrid retrieval store."""

    point_id: int
    doc_id: str
    chunk_idx: int
    score: float
    page: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_sources: List[str] = field(default_factory=list)


@runtime_checkable
class DocumentRepository(Protocol):
    """Protocol for on-demand document metadata, page, chunk, and table lookup."""

    def list_metainfo(self) -> List[Dict[str, Any]]:
        ...

    def get_metainfo(self, doc_id: str) -> Optional[Dict[str, Any]]:
        ...

    def get_page(self, doc_id: str, page: int) -> Optional[Dict[str, Any]]:
        ...

    def get_pages(self, doc_id: str) -> List[Dict[str, Any]]:
        ...

    def get_chunk_by_index(self, doc_id: str, chunk_idx: int) -> Optional[Dict[str, Any]]:
        ...

    def get_chunks_by_indices(self, doc_id: str, indices: Sequence[int]) -> List[Dict[str, Any]]:
        ...

    def get_parent_chunk(self, doc_id: str, parent_chunk_id: int) -> Optional[Dict[str, Any]]:
        ...

    def get_tables(self, doc_id: str, page: Optional[int] = None) -> List[Dict[str, Any]]:
        ...

    def get_corpus_version(self) -> str:
        ...

    def stats(self) -> Dict[str, Any]:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class RetrievalStore(Protocol):
    """Protocol for unified Dense + Sparse + Multi-Vector + Metadata hybrid search stores."""

    def ensure_collection(self) -> None:
        ...

    def upsert_points(self, points: Sequence[Dict[str, Any]], batch_size: int = 256) -> int:
        ...

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
        ...

    def stats(self) -> Dict[str, Any]:
        ...

    def close(self) -> None:
        ...
