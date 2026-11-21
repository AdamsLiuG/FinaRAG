"""Targeted lightweight Financial Knowledge Graph & LightRAG+PPR retriever for FinaRAG."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "EntityNode": ("src.knowledge_graph.kg_builder", "EntityNode"),
    "FinancialKGBuilder": ("src.knowledge_graph.kg_builder", "FinancialKGBuilder"),
    "FinancialKGStore": ("src.knowledge_graph.kg_builder", "FinancialKGStore"),
    "LightRAGRetriever": ("src.knowledge_graph.lightrag_retriever", "LightRAGRetriever"),
    "RelationEdge": ("src.knowledge_graph.kg_builder", "RelationEdge"),
    "canonicalize_entity_name": ("src.knowledge_graph.kg_builder", "canonicalize_entity_name"),
    "fuse_kg_and_hybrid_results": ("src.knowledge_graph.lightrag_retriever", "fuse_kg_and_hybrid_results"),
    "should_trigger_kg_retrieval": ("src.knowledge_graph.lightrag_retriever", "should_trigger_kg_retrieval"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str):
    if name in _EXPORTS:
        mod_name, attr_name = _EXPORTS[name]
        module = importlib.import_module(mod_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
