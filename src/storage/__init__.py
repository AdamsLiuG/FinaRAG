"""Unified hybrid retrieval and on-demand document storage abstractions for FinaRAG P1."""

from src.storage.base import (
    DocumentRepository,
    HybridSearchFilter,
    HybridSearchHit,
    RetrievalStore,
    StorageError,
    StorageSecurityError,
    validate_safe_path,
)

__all__ = [
    "DocumentRepository",
    "HybridSearchFilter",
    "HybridSearchHit",
    "RetrievalStore",
    "StorageError",
    "StorageSecurityError",
    "validate_safe_path",
]
