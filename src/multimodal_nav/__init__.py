"""P2 Multimodal & Hierarchical Document Navigation modules for FinaRAG."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ContextualEnricher": ("src.multimodal_nav.contextual_enricher", "ContextualEnricher"),
    "PageVisualScore": ("src.multimodal_nav.visual_index", "PageVisualScore"),
    "ReportTocTree": ("src.multimodal_nav.toc_tree_builder", "ReportTocTree"),
    "TocNode": ("src.multimodal_nav.toc_tree_builder", "TocNode"),
    "TocTreeBuilder": ("src.multimodal_nav.toc_tree_builder", "TocTreeBuilder"),
    "TocTreeNavigator": ("src.multimodal_nav.tree_navigator", "TocTreeNavigator"),
    "TocTreeStore": ("src.multimodal_nav.toc_tree_builder", "TocTreeStore"),
    "VLMExtractionResult": ("src.multimodal_nav.vlm_page_reader", "VLMExtractionResult"),
    "VLMPageReader": ("src.multimodal_nav.vlm_page_reader", "VLMPageReader"),
    "VLMVerificationResult": ("src.multimodal_nav.vlm_page_reader", "VLMVerificationResult"),
    "VisualColBERTIndex": ("src.multimodal_nav.visual_index", "VisualColBERTIndex"),
    "fuse_tree_and_hybrid_results": ("src.multimodal_nav.tree_navigator", "fuse_tree_and_hybrid_results"),
    "render_pdf_pages": ("src.multimodal_nav.visual_index", "render_pdf_pages"),
    "score_page_visual_complexity": ("src.multimodal_nav.visual_index", "score_page_visual_complexity"),
    "select_visual_pages": ("src.multimodal_nav.visual_index", "select_visual_pages"),
    "should_trigger_tree_navigation": ("src.multimodal_nav.tree_navigator", "should_trigger_tree_navigation"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str):
    if name in _EXPORTS:
        mod_name, attr_name = _EXPORTS[name]
        module = importlib.import_module(mod_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
