import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.text_normalization import contains_cjk, normalize_period_token, parse_numeric_value

class TextSplitter():
    def __init__(self, child_chunk_size: int = 320, child_chunk_overlap: int = 50):
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap

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
        payload = {
            "page": source_chunk["page"],
            "length_tokens": self.count_tokens(text),
            "text": text,
            "id": chunk_id,
            "chunk_id": chunk_id,
            "type": source_chunk.get("chunk_type", source_chunk.get("type", "content")),
            "chunk_type": source_chunk.get("chunk_type", source_chunk.get("type", "content")),
            "node_type": node_type,
            "section_title": source_chunk.get("section_title"),
            "table_id": source_chunk.get("table_id"),
            "report_year": report_meta.get("report_year"),
            "currency": report_meta.get("currency"),
            "company_name": report_meta.get("company_name"),
            "major_industry": report_meta.get("major_industry"),
            "report_type": report_meta.get("report_type"),
            "doc_source_type": report_meta.get("doc_source_type"),
            "company_aliases": list(report_meta.get("company_aliases") or []),
            "security_code": report_meta.get("security_code"),
            "broker_name": report_meta.get("broker_name"),
            "report_date": report_meta.get("report_date"),
            "fiscal_year": report_meta.get("fiscal_year"),
            "language": report_meta.get("language"),
            "topic_flags": self._report_topic_flags(report_meta),
            "parent_block_id": source_chunk.get("parent_block_id"),
            "parent_chunk_id": parent_chunk_id,
            "report_section": source_chunk.get("report_section", source_chunk.get("section_title")),
            "evidence_type": source_chunk.get("evidence_type", "narrative"),
            "has_table_context": bool(source_chunk.get("has_table_context")),
            "period": source_chunk.get("period") or normalize_period_token(source_chunk.get("text", "")),
            "unit_hint": source_chunk.get("unit_hint") or self._extract_unit_hint(source_chunk.get("text", "")),
        }
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

    def _split_report(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """Split report into parent/child chunks, preserving structure-aware metadata."""
        child_chunks: List[Dict[str, Any]] = []
        parent_chunks: List[Dict[str, Any]] = []
        parent_chunk_id = 0
        child_chunk_id = 0
        report_meta = file_content.get("metainfo", {})
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
        
        for page in pages:
            page_has_table_context = bool(tables_by_page and page['page'] in tables_by_page)
            page_blocks = self._split_page(page, report_meta, has_table_context=page_has_table_context)
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
                    table['report_section'] = table['section_title']
                    table['evidence_type'] = 'table'
                    table['has_table_context'] = True
                    parent_chunk = self._build_parent_chunk(table, report_meta, parent_chunk_id)
                    child_nodes, child_chunk_id = self._split_parent_chunk(parent_chunk, report_meta, child_chunk_id)
                    parent_chunks.append(parent_chunk)
                    child_chunks.extend(child_nodes)
                    parent_chunk_id += 1
        
        file_content['content']['parent_chunks'] = parent_chunks
        file_content['content']['chunks'] = child_chunks
        file_content['content']['structured_tables'] = structured_tables
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

    def _split_page(self, page: Dict[str, any], report_meta: Dict[str, any], has_table_context: bool = False) -> List[Dict[str, any]]:
        """Split page text into structure-aware parent blocks."""
        all_chunks: List[Dict[str, any]] = []
        for index, block in enumerate(self._extract_structural_blocks(page)):
            block["parent_block_id"] = f"page{page['page']}_block{index}"
            block["report_section"] = block.get("section_title")
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

    def split_all_reports(self, all_report_dir: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None):

        all_report_paths = list(all_report_dir.glob("*.json"))
        
        for report_path in all_report_paths:
            serialized_tables_path = None
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"Warning: Could not find serialized tables report for {report_path.name}")
                
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
                
            updated_report = self._split_report(report_data, serialized_tables_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)
                
        print(f"Split {len(all_report_paths)} files")
