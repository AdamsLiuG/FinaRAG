"""Contextual Retrieval prefix enricher injecting company, section, scope, unit, and page headers."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

from src.multimodal_nav.toc_tree_builder import ReportTocTree
from src.structured.pot_sandbox import normalize_monetary_unit

_CONTEXT_HEADER_TAG = "[上下文定位]"
_UNIT_RE = re.compile(r"单位[：:\s]*(人民币(?:亿元|千万元|百万元|万元|千元|元)|亿元|千万元|百万元|万元|千元|元)")
_GENERIC_SECTION_RE = re.compile(r"^(?:Page\s*\d+|页\s*\d+|第\s*\d+\s*页|附注)?$", re.IGNORECASE)


def _strip_existing_prefix(text: str) -> str:
    raw = str(text or "")
    lines = raw.splitlines()
    if lines and lines[0].lstrip().startswith(_CONTEXT_HEADER_TAG):
        return "\n".join(lines[1:]).lstrip("\n")
    return raw


class ContextualEnricher:
    """Prepends deterministic (+ optional LLM) contextual locators to chunks for Dense & Sparse retrieval."""

    @staticmethod
    def detect_scope_label(text: str, section_path: str = "") -> str:
        combined = f"{section_path}\n{text}"
        if any(tok in combined for tok in ("母公司", "母公司资产负债表", "母公司利润表", "母公司现金流量表")):
            return "母公司口径"
        if any(tok in combined for tok in ("合并", "归属于上市公司股东", "归属于母公司")):
            return "合并口径"
        return ""

    @staticmethod
    def detect_unit_label(chunk: Dict[str, Any], page_text: str = "") -> str:
        text = str(chunk.get("text") or "")
        for source in (text[:500], str(page_text or "")[:1200]):
            m = _UNIT_RE.search(source)
            if m:
                return m.group(1).strip()
        hint = str(chunk.get("unit_hint") or "").strip()
        if hint:
            return normalize_monetary_unit(hint) or hint
        return ""

    def build_context_prefix(
        self,
        chunk: Dict[str, Any],
        report_meta: Dict[str, Any],
        *,
        toc_tree: Optional[ReportTocTree] = None,
        page_text: str = "",
        llm_summary_fn: Optional[Callable[[str, Dict[str, Any]], str]] = None,
    ) -> str:
        company = str(chunk.get("company_name") or report_meta.get("company_name") or "").strip()
        year = chunk.get("report_year") or report_meta.get("report_year") or report_meta.get("fiscal_year")
        doc_source = str(chunk.get("doc_source_type") or report_meta.get("doc_source_type") or "annual_report")
        doc_label = "年度报告" if "annual" in doc_source else "报告"
        header_entity = " ".join(part for part in (company, f"{year}年{doc_label}" if year else "") if part).strip()

        page = chunk.get("page")
        sec_path = str(
            chunk.get("section_path")
            or chunk.get("report_section")
            or chunk.get("section_name")
            or chunk.get("section_title")
            or ""
        ).strip()
        if (not sec_path or _GENERIC_SECTION_RE.match(sec_path)) and toc_tree is not None and page is not None:
            resolved = toc_tree.find_section_path(int(page))
            if resolved:
                sec_path = resolved

        scope = self.detect_scope_label(f"{chunk.get('text') or ''}\n{page_text[:500]}", sec_path)
        unit = self.detect_unit_label(chunk, page_text=page_text)

        parts: list[str] = []
        if header_entity:
            parts.append(header_entity)
        if sec_path and not _GENERIC_SECTION_RE.match(sec_path):
            parts.append(sec_path)
        if scope:
            parts.append(scope)
        if unit:
            parts.append(f"单位：{unit}")
        if page is not None:
            parts.append(f"第{page}页")

        is_weak_heading = (not sec_path) or bool(_GENERIC_SECTION_RE.match(sec_path)) or ("附注" in sec_path)
        if llm_summary_fn is not None and is_weak_heading:
            try:
                note_sum = str(llm_summary_fn(str(chunk.get("text") or ""), report_meta) or "").strip()
                if note_sum:
                    parts.append(note_sum[:80])
            except Exception:
                pass

        return f"{_CONTEXT_HEADER_TAG} " + " | ".join(parts)

    def enrich_chunk(
        self,
        chunk: Dict[str, Any],
        report_meta: Dict[str, Any],
        *,
        toc_tree: Optional[ReportTocTree] = None,
        page_text: str = "",
        llm_summary_fn: Optional[Callable[[str, Dict[str, Any]], str]] = None,
    ) -> Dict[str, Any]:
        enriched = dict(chunk)
        prefix = self.build_context_prefix(
            enriched,
            report_meta,
            toc_tree=toc_tree,
            page_text=page_text,
            llm_summary_fn=llm_summary_fn,
        )
        raw_text = str(enriched.get("text") or "")
        base_emb = _strip_existing_prefix(str(enriched.get("embedding_text") or raw_text))
        base_search = _strip_existing_prefix(str(enriched.get("search_text") or raw_text))

        enriched["contextual_prefix"] = prefix
        enriched["embedding_text"] = f"{prefix}\n{base_emb}".strip()
        enriched["search_text"] = f"{prefix}\n{base_search}".strip()
        return enriched

    def enrich_report(
        self,
        report_dict: Dict[str, Any],
        *,
        toc_tree: Optional[ReportTocTree] = None,
        llm_summary_fn: Optional[Callable[[str, Dict[str, Any]], str]] = None,
    ) -> Dict[str, Any]:
        out = dict(report_dict)
        metainfo = out.get("metainfo") or {}
        content = dict(out.get("content") or {})
        pages_map = {
            int(p["page"]): str(p.get("text") or "")
            for p in content.get("pages") or []
            if isinstance(p, dict) and p.get("page") is not None
        }
        enriched_chunks = []
        for chunk in content.get("chunks") or []:
            if not isinstance(chunk, dict):
                continue
            pno = chunk.get("page")
            page_txt = pages_map.get(int(pno), "") if pno is not None else ""
            enriched_chunks.append(
                self.enrich_chunk(
                    chunk,
                    metainfo,
                    toc_tree=toc_tree,
                    page_text=page_txt,
                    llm_summary_fn=llm_summary_fn,
                )
            )
        content["chunks"] = enriched_chunks
        out["content"] = content
        return out
