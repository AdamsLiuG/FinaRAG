import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Backward compatibility for older LangChain releases.
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.pdfcrawl_metadata import (
    load_company_label_snapshot_lookup,
    load_report_page_lookup,
    write_jsonl,
)
from src.text_normalization import contains_cjk, dedupe_preserve_order, normalize_period_token, parse_numeric_value


_COMPANY_LABEL_EVIDENCE_ALIASES = {
    "国产替代": ["国产替代", "国产化", "自主可控", "信创"],
    "数字化转型": ["数字化转型", "数字化", "数智化"],
    "出海": ["出海", "海外", "海外市场", "海外业务", "国际化", "境外"],
    "绿色转型": ["绿色转型", "绿色低碳", "双碳", "碳中和"],
    "人工智能": ["人工智能", "AI", "大模型"],
}


class TextSplitter():
    def __init__(self, child_chunk_size: int = 320, child_chunk_overlap: int = 50):
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self._metadata_tag_fields = (
            "business_tags",
            "strategy_tags",
            "factor_tags",
            "chain_position_minor",
            "listing_tags",
            "ownership_tags",
            "status_tags",
            "style_tags",
        )
        self._company_label_evidence_fields = ("strategy_tags",)

    def _infer_report_year(self, pages: List[Dict[str, any]]) -> Optional[int]:
        year_counter: Counter[int] = Counter()
        for page in pages[:5]:
            for match in re.finditer(r"\b(19|20)\d{2}\b", page.get("text", "")):
                year = int(match.group(0))
                if 1990 <= year <= 2100:
                    year_counter[year] += 1
        if not year_counter:
            return None
        return year_counter.most_common(1)[0][0]

    def _extract_structural_blocks(self, page: Dict[str, any]) -> List[Dict[str, any]]:
        lines = [line.rstrip() for line in page.get("text", "").splitlines()]
        blocks: List[Dict[str, any]] = []
        current_title = None
        current_lines: List[str] = []

        def flush_block():
            if not current_lines:
                return
            block_text = "\n".join(line for line in current_lines if line.strip()).strip()
            if not block_text:
                return
            blocks.append(
                {
                    "page": page["page"],
                    "section_title": current_title or f"Page {page['page']}",
                    "text": block_text,
                    "chunk_type": "content",
                    "table_id": None,
                }
            )

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                current_lines.append("")
                continue

            if self._is_heading_line(line):
                flush_block()
                current_title = line.lstrip("#").strip()
                current_lines = [line]
                continue

            current_lines.append(raw_line)

        flush_block()
        if not blocks and page.get("text", "").strip():
            blocks.append(
                {
                    "page": page["page"],
                    "section_title": f"Page {page['page']}",
                    "text": page["text"].strip(),
                    "chunk_type": "content",
                    "table_id": None,
                }
            )
        return blocks

    @staticmethod
    def _is_heading_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.startswith(("# ", "## ", "### ")):
            return True
        if len(stripped) > 36:
            return False
        if any(token in stripped for token in ("。", "；", ";", ":", "：", "，", ",")):
            return False
        return stripped.endswith(("报告", "概况", "概览", "目录", "说明", "分析", "风险提示", "情况"))

    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """Group serialized tables by page number"""
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
                
            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []
            
            table_text = "\n".join(
                block["information_block"] 
                for block in table["serialized"]["information_blocks"]
            )
            
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self.count_tokens(table_text)
            })
            
        return tables_by_page

    def _report_topic_flags(self, report_meta: Dict[str, any]) -> List[str]:
        return sorted(
            key
            for key, value in report_meta.items()
            if key.startswith(("has_", "mentions_")) and str(value).strip().lower() == "true"
        )

    def _coerce_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return dedupe_preserve_order([item for item in value if item not in (None, "")])
        return [value]

    def _merge_report_snapshot(self, report_meta: Dict[str, Any], report_snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not report_snapshot:
            return report_meta

        merged = dict(report_meta)
        scalar_fields = (
            "stock_code",
            "exchange",
            "board",
            "market_type",
            "industry_l1",
            "industry_l2",
            "chain_position_major",
        )
        for field in scalar_fields:
            if not merged.get(field) and report_snapshot.get(field):
                merged[field] = report_snapshot.get(field)

        merged["stock_code"] = merged.get("stock_code") or merged.get("security_code")
        for field in self._metadata_tag_fields:
            merged[field] = self._coerce_list(merged.get(field) or report_snapshot.get(field))
        return merged

    def _apply_page_metadata(self, block: Dict[str, Any], page_metadata: Optional[Dict[str, Any]], report_meta: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(block)
        page_number = enriched.get("page")
        enriched["stock_code"] = (page_metadata or {}).get("stock_code") or report_meta.get("stock_code") or report_meta.get("security_code")
        enriched["exchange"] = (page_metadata or {}).get("exchange") or report_meta.get("exchange")
        enriched["board"] = (page_metadata or {}).get("board") or report_meta.get("board")
        enriched["market_type"] = (page_metadata or {}).get("market_type") or report_meta.get("market_type")
        enriched["industry_l1"] = (page_metadata or {}).get("industry_l1") or report_meta.get("industry_l1")
        enriched["industry_l2"] = (page_metadata or {}).get("industry_l2") or report_meta.get("industry_l2")
        enriched["page_start"] = (page_metadata or {}).get("page_start") or page_number
        enriched["page_end"] = (page_metadata or {}).get("page_end") or page_number
        enriched["section_name"] = (
            (page_metadata or {}).get("section_name")
            or enriched.get("section_name")
            or enriched.get("report_section")
            or enriched.get("section_title")
            or f"Page {page_number}"
        )
        enriched["section_l1"] = (page_metadata or {}).get("section_l1") or enriched.get("section_l1")
        enriched["section_l2"] = (page_metadata or {}).get("section_l2") or enriched.get("section_l2")
        enriched["section_l3"] = (page_metadata or {}).get("section_l3") or enriched.get("section_l3")
        enriched["section_path"] = (
            (page_metadata or {}).get("section_path")
            or enriched.get("section_path")
            or enriched.get("report_section")
            or enriched.get("section_name")
        )
        enriched["section_leaf"] = (
            (page_metadata or {}).get("section_leaf")
            or enriched.get("section_leaf")
            or enriched.get("section_name")
        )
        enriched["page_role"] = (page_metadata or {}).get("page_role") or enriched.get("page_role")
        enriched["chunk_metadata_id"] = (page_metadata or {}).get("chunk_id")
        enriched["chain_position_major"] = (page_metadata or {}).get("chain_position_major") or report_meta.get("chain_position_major")
        for field in self._metadata_tag_fields:
            enriched[field] = self._coerce_list((page_metadata or {}).get(field) or report_meta.get(field))
        return enriched

    @staticmethod
    def _normalize_report_type_label(doc_source_type: Optional[str], report_year: Any) -> Optional[str]:
        year_text = str(report_year).strip() if report_year not in (None, "") else ""
        suffix = {
            "annual_report": "年报",
            "interim_report": "中期报告",
            "research_report": "研报",
        }.get(str(doc_source_type or "").strip(), "文档")
        if not year_text:
            return suffix
        return f"{year_text}年{suffix}"

    def _build_embedding_text(self, payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        company_name = payload.get("company_name")
        stock_code = payload.get("stock_code") or payload.get("security_code")
        report_label = self._normalize_report_type_label(payload.get("doc_source_type"), payload.get("report_year"))
        market_tokens = [token for token in (payload.get("market_type"), payload.get("board")) if token]
        industry_tokens = [token for token in (payload.get("industry_l1"), payload.get("industry_l2")) if token]

        if company_name:
            lines.append(f"公司：{company_name}")
        if stock_code:
            lines.append(f"代码：{stock_code}")
        if report_label:
            lines.append(f"年份：{report_label}")
        if market_tokens:
            lines.append(f"市场：{'、'.join(dedupe_preserve_order(market_tokens))}")
        if industry_tokens:
            lines.append(f"行业：{'-'.join(dedupe_preserve_order(industry_tokens))}")
        if payload.get("section_name"):
            lines.append(f"章节：{payload['section_name']}")
        if payload.get("section_path"):
            lines.append(f"章节路径：{payload['section_path']}")
        if payload.get("evidence_type") == "chart":
            if payload.get("chart_id"):
                lines.append(f"图表ID：{payload['chart_id']}")
            if payload.get("series_name"):
                lines.append(f"图表指标：{payload['series_name']}")
            if payload.get("x_label"):
                lines.append(f"图表横轴：{payload['x_label']}")
        if payload.get("business_tags"):
            lines.append(f"业务主题：{'、'.join(payload['business_tags'])}")
        if payload.get("factor_tags"):
            lines.append(f"因子主题：{'、'.join(payload['factor_tags'])}")
        chain_minor = payload.get("chain_position_minor") or []
        if payload.get("chain_position_major") or chain_minor:
            lines.append(
                "产业链："
                + "/".join(
                    token
                    for token in [
                        payload.get("chain_position_major"),
                        "、".join(chain_minor) if chain_minor else None,
                    ]
                    if token
                )
            )
        lines.append("")
        lines.append("正文：")
        lines.append(payload.get("text") or "")
        return "\n".join(lines).strip()

    def _build_search_text(self, payload: Dict[str, Any]) -> str:
        parts: List[str] = []
        for field in (
            "company_name",
            "stock_code",
            "report_year",
            "doc_source_type",
            "exchange",
            "board",
            "market_type",
            "industry_l1",
            "industry_l2",
            "section_name",
            "section_path",
            "section_leaf",
        ):
            value = payload.get(field)
            if value not in (None, ""):
                parts.append(str(value))
        for field in self._metadata_tag_fields:
            if field == "strategy_tags":
                continue
            parts.extend(str(item) for item in payload.get(field) or [] if item)
        parts.append(payload.get("text") or "")
        return "\n".join(parts).strip()

    @staticmethod
    def _label_evidence_terms(label: Any) -> List[str]:
        label_text = str(label or "").strip()
        if not label_text:
            return []
        return dedupe_preserve_order([label_text] + _COMPANY_LABEL_EVIDENCE_ALIASES.get(label_text, []))

    @staticmethod
    def _snippet_around_match(text: str, start: int, end: int, before: int = 80, after: int = 140) -> str:
        snippet_start = max(0, start - before)
        snippet_end = min(len(text), end + after)
        snippet = text[snippet_start:snippet_end]
        return " ".join(snippet.split())

    def _find_company_label_evidence(self, pages: List[Dict[str, Any]], label: Any) -> Optional[Dict[str, Any]]:
        for term in self._label_evidence_terms(label):
            flags = re.IGNORECASE if term.isascii() else 0
            pattern = re.compile(re.escape(term), flags)
            for page in pages:
                text = str(page.get("text") or "")
                match = pattern.search(text)
                if not match:
                    continue
                return {
                    "evidence_page": page.get("page"),
                    "evidence_snippet": self._snippet_around_match(text, match.start(), match.end()),
                    "match_term": term,
                    "has_literal_evidence": True,
                }
        return None

    def _build_company_label_evidence_rows(
        self,
        report_id: str,
        file_content: Dict[str, Any],
        report_snapshot: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        report_meta = self._merge_report_snapshot(file_content.get("metainfo", {}), report_snapshot)
        pages = file_content.get("content", {}).get("pages", [])
        label_source = (report_snapshot or {}).get("label_source") or "company_label_snapshot"
        rows: List[Dict[str, Any]] = []

        for field in self._company_label_evidence_fields:
            for label in self._coerce_list(report_meta.get(field)):
                evidence = self._find_company_label_evidence(pages, label) or {
                    "evidence_page": None,
                    "evidence_snippet": "",
                    "match_term": None,
                    "has_literal_evidence": False,
                }
                rows.append(
                    {
                        "report_id": report_id,
                        "doc_id": report_id,
                        "company_name": report_meta.get("company_name"),
                        "stock_code": report_meta.get("stock_code") or report_meta.get("security_code"),
                        "report_year": report_meta.get("report_year") or report_meta.get("fiscal_year"),
                        "doc_source_type": report_meta.get("doc_source_type"),
                        "label_field": field,
                        "label": label,
                        "source": label_source,
                        "evidence_source": "annual_report_text" if evidence["has_literal_evidence"] else label_source,
                        **evidence,
                    }
                )

        return rows

    def _build_chunk_payload(
        self,
        text: str,
        source_chunk: Dict[str, Any],
        report_meta: Dict[str, Any],
        *,
        chunk_id: int,
        node_type: str,
        parent_chunk_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        section_title = source_chunk.get("section_title")
        section_name = source_chunk.get("section_name") or source_chunk.get("report_section") or section_title or f"Page {source_chunk['page']}"
        section_path = source_chunk.get("section_path") or source_chunk.get("report_section") or section_name
        section_leaf = source_chunk.get("section_leaf") or section_name
        payload = {
            "page": source_chunk["page"],
            "page_start": source_chunk.get("page_start") or source_chunk["page"],
            "page_end": source_chunk.get("page_end") or source_chunk["page"],
            "length_tokens": self.count_tokens(text),
            "text": text,
            "id": chunk_id,
            "chunk_id": chunk_id,
            "type": source_chunk.get("chunk_type", source_chunk.get("type", "content")),
            "chunk_type": source_chunk.get("chunk_type", source_chunk.get("type", "content")),
            "node_type": node_type,
            "section_title": section_title,
            "section_name": section_name,
            "section_l1": source_chunk.get("section_l1"),
            "section_l2": source_chunk.get("section_l2"),
            "section_l3": source_chunk.get("section_l3"),
            "section_path": section_path,
            "section_leaf": section_leaf,
            "table_id": source_chunk.get("table_id"),
            "chart_id": source_chunk.get("chart_id"),
            "picture_id": source_chunk.get("picture_id"),
            "chart_type": source_chunk.get("chart_type"),
            "series_name": source_chunk.get("series_name"),
            "x_label": source_chunk.get("x_label"),
            "chart_confidence": source_chunk.get("chart_confidence"),
            "has_chart_context": bool(source_chunk.get("has_chart_context")),
            "bbox": source_chunk.get("bbox"),
            "report_year": report_meta.get("report_year"),
            "currency": report_meta.get("currency"),
            "company_name": report_meta.get("company_name"),
            "major_industry": report_meta.get("major_industry"),
            "report_type": report_meta.get("report_type"),
            "doc_source_type": report_meta.get("doc_source_type"),
            "company_aliases": list(report_meta.get("company_aliases") or []),
            "security_code": report_meta.get("security_code"),
            "stock_code": source_chunk.get("stock_code") or report_meta.get("stock_code") or report_meta.get("security_code"),
            "broker_name": report_meta.get("broker_name"),
            "report_date": report_meta.get("report_date"),
            "fiscal_year": report_meta.get("fiscal_year"),
            "language": report_meta.get("language"),
            "topic_flags": self._report_topic_flags(report_meta),
            "parent_block_id": source_chunk.get("parent_block_id"),
            "parent_chunk_id": parent_chunk_id,
            "report_section": source_chunk.get("report_section", section_name),
            "evidence_type": source_chunk.get("evidence_type", "narrative"),
            "has_table_context": bool(source_chunk.get("has_table_context")),
            "page_role": source_chunk.get("page_role"),
            "period": source_chunk.get("period") or normalize_period_token(source_chunk.get("text", "")),
            "unit_hint": source_chunk.get("unit_hint") or self._extract_unit_hint(source_chunk.get("text", "")),
            "exchange": source_chunk.get("exchange") or report_meta.get("exchange"),
            "board": source_chunk.get("board") or report_meta.get("board"),
            "market_type": source_chunk.get("market_type") or report_meta.get("market_type"),
            "industry_l1": source_chunk.get("industry_l1") or report_meta.get("industry_l1"),
            "industry_l2": source_chunk.get("industry_l2") or report_meta.get("industry_l2"),
            "chain_position_major": source_chunk.get("chain_position_major") or report_meta.get("chain_position_major"),
            "chunk_metadata_id": source_chunk.get("chunk_metadata_id"),
        }
        for field in self._metadata_tag_fields:
            payload[field] = self._coerce_list(source_chunk.get(field) or report_meta.get(field))
        payload["embedding_text"] = self._build_embedding_text(payload)
        payload["search_text"] = self._build_search_text(payload)
        if node_type == "parent":
            payload["child_chunk_ids"] = []
        return payload

    def _build_parent_chunk(self, block: Dict[str, Any], report_meta: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
        return self._build_chunk_payload(
            block["text"],
            block,
            report_meta,
            chunk_id=chunk_id,
            node_type="parent",
        )

    def _split_report(
        self,
        file_content: Dict[str, any],
        serialized_tables_report_path: Optional[Path] = None,
        report_page_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
        report_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, any]:
        """Split report into parent/child chunks, preserving structure-aware metadata."""
        child_chunks: List[Dict[str, Any]] = []
        parent_chunks: List[Dict[str, Any]] = []
        parent_chunk_id = 0
        child_chunk_id = 0
        report_meta = self._merge_report_snapshot(file_content.get("metainfo", {}), report_snapshot)
        pages = file_content["content"]["pages"]
        report_year = self._infer_report_year(pages)
        if report_year is not None:
            report_meta["report_year"] = report_year
        
        tables_by_page = {}
        structured_tables: List[Dict[str, Any]] = []
        if serialized_tables_report_path is not None:
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))
            structured_tables = self._extract_structured_tables(parsed_report.get('tables', []))

        charts = list(file_content.get("content", {}).get("charts") or [])
        charts_by_page = self._get_charts_by_page(charts)
        chart_records = self._extract_chart_records(charts)
        
        for page in pages:
            page_metadata = (report_page_metadata or {}).get(page["page"])
            page_has_table_context = bool(tables_by_page and page['page'] in tables_by_page)
            page_blocks = self._split_page(
                page,
                report_meta,
                page_metadata=page_metadata,
                has_table_context=page_has_table_context,
            )
            for block in page_blocks:
                parent_chunk = self._build_parent_chunk(block, report_meta, parent_chunk_id)
                child_nodes, child_chunk_id = self._split_parent_chunk(parent_chunk, report_meta, child_chunk_id)
                parent_chunks.append(parent_chunk)
                child_chunks.extend(child_nodes)
                parent_chunk_id += 1
            
            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['section_title'] = self._get_page_section_title(page)
                    table['type'] = 'serialized_table'
                    table['chunk_type'] = 'serialized_table'
                    table['parent_block_id'] = f"page{page['page']}_table{table['table_id']}"
                    table['report_section'] = table.get('section_name') or table['section_title']
                    table['evidence_type'] = 'table'
                    table['has_table_context'] = True
                    table = self._apply_page_metadata(table, page_metadata, report_meta)
                    parent_chunk = self._build_parent_chunk(table, report_meta, parent_chunk_id)
                    child_nodes, child_chunk_id = self._split_parent_chunk(parent_chunk, report_meta, child_chunk_id)
                    parent_chunks.append(parent_chunk)
                    child_chunks.extend(child_nodes)
                    parent_chunk_id += 1

            if charts_by_page and page['page'] in charts_by_page:
                for chart in charts_by_page[page['page']]:
                    chart_chunk = self._build_chart_chunk(chart, page)
                    chart_chunk = self._apply_page_metadata(chart_chunk, page_metadata, report_meta)
                    parent_chunk = self._build_parent_chunk(chart_chunk, report_meta, parent_chunk_id)
                    child_nodes, child_chunk_id = self._split_parent_chunk(parent_chunk, report_meta, child_chunk_id)
                    parent_chunks.append(parent_chunk)
                    child_chunks.extend(child_nodes)
                    parent_chunk_id += 1
        
        file_content['content']['parent_chunks'] = parent_chunks
        file_content['content']['chunks'] = child_chunks
        file_content['content']['structured_tables'] = structured_tables
        file_content['content']['chart_records'] = chart_records
        file_content['metainfo'] = report_meta
        return file_content

    def _get_page_section_title(self, page: Dict[str, any]) -> str:
        for line in page.get("text", "").splitlines():
            stripped = line.strip()
            if stripped.startswith(("# ", "## ", "### ")):
                return stripped.lstrip("#").strip()
        return f"Page {page['page']}"

    def count_tokens(self, string: str, encoding_name="o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name)

        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count

    def _split_parent_chunk(
        self,
        parent_chunk: Dict[str, Any],
        report_meta: Dict[str, Any],
        child_chunk_id_start: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        separators = ["\n### ", "\n## ", "\n# ", "\n\n", "\n", ". ", "; ", ", "]
        if contains_cjk(parent_chunk.get("text", "")) or report_meta.get("language") in {"zh", "bilingual"}:
            separators = ["\n### ", "\n## ", "\n# ", "\n\n", "\n", "。", "；", "：", "，", " "]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
            separators=separators,
        )
        split_texts = text_splitter.split_text(parent_chunk['text'])
        if not split_texts and parent_chunk['text'].strip():
            split_texts = [parent_chunk['text']]

        child_chunks: List[Dict[str, Any]] = []
        for chunk_text in split_texts:
            child_chunk = self._build_chunk_payload(
                chunk_text,
                parent_chunk,
                report_meta,
                chunk_id=child_chunk_id_start,
                node_type="child",
                parent_chunk_id=parent_chunk["chunk_id"],
            )
            child_chunks.append(child_chunk)
            parent_chunk["child_chunk_ids"].append(child_chunk_id_start)
            child_chunk_id_start += 1

        return child_chunks, child_chunk_id_start

    def _split_page(
        self,
        page: Dict[str, any],
        report_meta: Dict[str, any],
        page_metadata: Optional[Dict[str, Any]] = None,
        has_table_context: bool = False,
    ) -> List[Dict[str, any]]:
        """Split page text into structure-aware parent blocks."""
        all_chunks: List[Dict[str, any]] = []
        for index, block in enumerate(self._extract_structural_blocks(page)):
            block["parent_block_id"] = f"page{page['page']}_block{index}"
            block = self._apply_page_metadata(block, page_metadata, report_meta)
            block["report_section"] = block.get("section_name") or block.get("section_title")
            block["evidence_type"] = "narrative"
            block["has_table_context"] = has_table_context
            block["period"] = normalize_period_token(block.get("text", ""))
            block["unit_hint"] = self._extract_unit_hint(block.get("text", ""))
            all_chunks.append(block)
        return all_chunks

    @staticmethod
    def _extract_unit_hint(text: str) -> Optional[str]:
        if not text:
            return None
        for unit in ("亿元", "万元", "亿股", "万股", "百万元", "千万元", "%", "百分点"):
            if unit in text:
                return unit
        return None

    def _get_charts_by_page(self, charts: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        charts_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for chart in charts:
            if chart.get("status") != "ok":
                continue
            page = chart.get("page")
            if page is None:
                continue
            charts_by_page.setdefault(int(page), []).append(chart)
        return charts_by_page

    def _build_chart_chunk(self, chart: Dict[str, Any], page: Dict[str, Any]) -> Dict[str, Any]:
        text = self._render_chart_text(chart)
        records = chart.get("records") or []
        series_names = [str(record.get("series_name")) for record in records if record.get("series_name")]
        x_labels = [str(record.get("x_label")) for record in records if record.get("x_label")]
        return {
            "page": chart.get("page"),
            "text": text,
            "length_tokens": self.count_tokens(text),
            "table_id": None,
            "chart_id": chart.get("chart_id"),
            "picture_id": chart.get("picture_id"),
            "chart_type": chart.get("chart_type"),
            "series_name": "、".join(dedupe_preserve_order(series_names)) if series_names else None,
            "x_label": "、".join(dedupe_preserve_order(x_labels)) if x_labels else None,
            "chart_confidence": chart.get("confidence"),
            "bbox": chart.get("bbox"),
            "section_title": self._get_page_section_title(page),
            "type": "chart_to_table",
            "chunk_type": "chart_to_table",
            "parent_block_id": f"page{chart.get('page')}_chart{chart.get('picture_id')}",
            "report_section": self._get_page_section_title(page),
            "evidence_type": "chart",
            "has_table_context": False,
            "has_chart_context": bool(chart.get("context_text")),
            "period": normalize_period_token(text),
            "unit_hint": chart.get("unit_hint") or self._extract_unit_hint(text),
        }

    @staticmethod
    def _render_chart_text(chart: Dict[str, Any]) -> str:
        lines = [
            "[Chart Evidence]",
            f"图表ID：{chart.get('chart_id')}",
            f"页码：{chart.get('page')}",
        ]
        if chart.get("picture_id") is not None:
            lines.append(f"图片ID：{chart.get('picture_id')}")
        table_text = str(chart.get("table_markdown") or chart.get("raw_output") or "").strip()
        if table_text:
            lines.extend(["DePlot表格：", table_text])
        context_text = str(chart.get("context_text") or "").strip()
        if context_text:
            lines.extend(["周边说明：", context_text])
        return "\n".join(lines)

    def _extract_chart_records(self, charts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chart_records: List[Dict[str, Any]] = []
        for chart in charts:
            if chart.get("status") != "ok":
                continue
            for record in chart.get("records") or []:
                enriched = dict(record)
                enriched.setdefault("chart_id", chart.get("chart_id"))
                enriched.setdefault("page", chart.get("page"))
                enriched.setdefault("picture_id", chart.get("picture_id"))
                enriched.setdefault("bbox", chart.get("bbox"))
                enriched.setdefault("context_text", chart.get("context_text"))
                if not enriched.get("unit"):
                    enriched["unit"] = chart.get("unit_hint")
                if enriched.get("confidence") is None:
                    enriched["confidence"] = chart.get("confidence")
                enriched.setdefault("table_markdown", chart.get("table_markdown"))
                chart_records.append(enriched)
        return chart_records

    def _extract_structured_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        structured_tables: List[Dict[str, Any]] = []
        for table in tables:
            table_json = table.get("json") or {}
            table_data = table_json.get("data") or {}
            cells = table_data.get("table_cells") or []
            if not cells:
                continue

            row_headers_by_row: Dict[int, List[str]] = {}
            col_headers_by_col: Dict[int, List[str]] = {}
            for cell in cells:
                text = str(cell.get("text") or "").strip()
                if not text:
                    continue
                start_row = int(cell.get("start_row_offset_idx", 0))
                end_row = int(cell.get("end_row_offset_idx", start_row + 1))
                start_col = int(cell.get("start_col_offset_idx", 0))
                end_col = int(cell.get("end_col_offset_idx", start_col + 1))
                if cell.get("column_header"):
                    for col_idx in range(start_col, end_col):
                        col_headers_by_col.setdefault(col_idx, []).append(text)
                if cell.get("row_header") or (start_col == 0 and not cell.get("column_header")):
                    for row_idx in range(start_row, end_row):
                        row_headers_by_row.setdefault(row_idx, []).append(text)

            cell_records: List[Dict[str, Any]] = []
            for cell in cells:
                text = str(cell.get("text") or "").strip()
                if not text:
                    continue
                if cell.get("column_header") or cell.get("row_header"):
                    continue
                numeric_value = parse_numeric_value(text, unit_hint=self._extract_unit_hint(table.get("markdown", "")))
                if numeric_value is None:
                    continue
                row_idx = int(cell.get("start_row_offset_idx", 0))
                col_idx = int(cell.get("start_col_offset_idx", 0))
                cell_records.append(
                    {
                        "table_id": table.get("table_id"),
                        "page": table.get("page"),
                        "row_idx": row_idx,
                        "col_idx": col_idx,
                        "raw_value": text,
                        "normalized_value": numeric_value,
                        "matched_row_headers": list(dict.fromkeys(row_headers_by_row.get(row_idx, []))),
                        "matched_col_headers": list(dict.fromkeys(col_headers_by_col.get(col_idx, []))),
                        "unit_hint": self._extract_unit_hint(table.get("markdown", "")),
                        "period": normalize_period_token(table.get("markdown", "")),
                        "footnote_refs": list(table_json.get("footnotes") or []),
                    }
                )

            structured_tables.append(
                {
                    "table_id": table.get("table_id"),
                    "page": table.get("page"),
                    "markdown": table.get("markdown", ""),
                    "html": table.get("html", ""),
                    "cell_records": cell_records,
                }
            )
        return structured_tables

    def _build_chunk_metadata_rows(self, report_id: str, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        metainfo = report.get("metainfo", {})
        for chunk in list(report.get("content", {}).get("parent_chunks", [])) + list(report.get("content", {}).get("chunks", [])):
            rows.append(
                {
                    "doc_id": report_id,
                    "report_id": report_id,
                    "stock_code": chunk.get("stock_code") or metainfo.get("stock_code") or metainfo.get("security_code"),
                    "company_name": chunk.get("company_name") or metainfo.get("company_name"),
                    "report_year": chunk.get("report_year") or metainfo.get("report_year"),
                    "chunk_id": chunk.get("chunk_id"),
                    "node_type": chunk.get("node_type"),
                    "chunk_type": chunk.get("chunk_type"),
                    "parent_chunk_id": chunk.get("parent_chunk_id"),
                    "page": chunk.get("page"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "section_name": chunk.get("section_name"),
                    "section_l1": chunk.get("section_l1"),
                    "section_l2": chunk.get("section_l2"),
                    "section_l3": chunk.get("section_l3"),
                    "section_path": chunk.get("section_path"),
                    "section_leaf": chunk.get("section_leaf"),
                    "section_title": chunk.get("section_title"),
                    "report_section": chunk.get("report_section"),
                    "table_id": chunk.get("table_id"),
                    "chart_id": chunk.get("chart_id"),
                    "picture_id": chunk.get("picture_id"),
                    "chart_type": chunk.get("chart_type"),
                    "series_name": chunk.get("series_name"),
                    "x_label": chunk.get("x_label"),
                    "chart_confidence": chunk.get("chart_confidence"),
                    "has_chart_context": chunk.get("has_chart_context"),
                    "evidence_type": chunk.get("evidence_type"),
                    "has_table_context": chunk.get("has_table_context"),
                    "exchange": chunk.get("exchange"),
                    "board": chunk.get("board"),
                    "market_type": chunk.get("market_type"),
                    "industry_l1": chunk.get("industry_l1"),
                    "industry_l2": chunk.get("industry_l2"),
                    "business_tags": chunk.get("business_tags", []),
                    "strategy_tags": chunk.get("strategy_tags", []),
                    "factor_tags": chunk.get("factor_tags", []),
                    "chain_position_major": chunk.get("chain_position_major"),
                    "chain_position_minor": chunk.get("chain_position_minor", []),
                    "listing_tags": chunk.get("listing_tags", []),
                    "ownership_tags": chunk.get("ownership_tags", []),
                    "status_tags": chunk.get("status_tags", []),
                    "style_tags": chunk.get("style_tags", []),
                    "embedding_text": chunk.get("embedding_text"),
                    "search_text": chunk.get("search_text"),
                    "chunk_metadata_id": chunk.get("chunk_metadata_id"),
                }
            )
        return rows

    def split_all_reports(
        self,
        all_report_dir: Path,
        output_dir: Path,
        serialized_tables_dir: Optional[Path] = None,
        metadata_store_dir: Optional[Path] = None,
    ):

        all_report_paths = list(all_report_dir.glob("*.json"))
        report_page_lookup = load_report_page_lookup(metadata_store_dir) if metadata_store_dir else {}
        company_snapshot_lookup = load_company_label_snapshot_lookup(metadata_store_dir) if metadata_store_dir else {}
        chunk_metadata_rows: List[Dict[str, Any]] = []
        company_label_evidence_rows: List[Dict[str, Any]] = []
        
        for report_path in all_report_paths:
            serialized_tables_path = None
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"Warning: Could not find serialized tables report for {report_path.name}")
                
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)

            report_id = str((report_data.get("metainfo") or {}).get("sha1_name") or report_path.stem)
            report_snapshot = company_snapshot_lookup.get(report_id)
            company_label_evidence_rows.extend(
                self._build_company_label_evidence_rows(
                    report_id,
                    report_data,
                    report_snapshot,
                )
            )
            updated_report = self._split_report(
                report_data,
                serialized_tables_path,
                report_page_metadata=report_page_lookup.get(report_id),
                report_snapshot=report_snapshot,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)
            chunk_metadata_rows.extend(self._build_chunk_metadata_rows(report_id, updated_report))

        if metadata_store_dir is not None:
            metadata_store_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(metadata_store_dir / "chunk_metadata.jsonl", chunk_metadata_rows)
            write_jsonl(metadata_store_dir / "report_chunk.jsonl", chunk_metadata_rows)
            write_jsonl(metadata_store_dir / "company_label_evidence.jsonl", company_label_evidence_rows)
                
        print(f"Split {len(all_report_paths)} files")
