import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplitter():
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

            if line.startswith(("# ", "## ", "### ")):
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

    def _split_report(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """Split report into chunks, preserving markdown tables in content and optionally including serialized tables."""
        chunks = []
        chunk_id = 0
        report_meta = file_content.get("metainfo", {})
        pages = file_content["content"]["pages"]
        report_year = self._infer_report_year(pages)
        if report_year is not None:
            report_meta["report_year"] = report_year
        
        tables_by_page = {}
        if serialized_tables_report_path is not None:
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))
        
        for page in pages:
            page_has_table_context = bool(tables_by_page and page['page'] in tables_by_page)
            page_chunks = self._split_page(page, report_meta, has_table_context=page_has_table_context)
            for chunk in page_chunks:
                chunk['id'] = chunk_id
                chunk['chunk_id'] = chunk_id
                chunk['type'] = 'content'
                chunk_id += 1
                chunks.append(chunk)
            
            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['id'] = chunk_id
                    table['chunk_id'] = chunk_id
                    table['type'] = 'serialized_table'
                    table['chunk_type'] = 'serialized_table'
                    table['section_title'] = self._get_page_section_title(page)
                    table['report_year'] = report_meta.get("report_year")
                    table['currency'] = report_meta.get("currency")
                    table['company_name'] = report_meta.get("company_name")
                    table['major_industry'] = report_meta.get("major_industry")
                    table['report_type'] = report_meta.get("report_type")
                    table['topic_flags'] = self._report_topic_flags(report_meta)
                    table['parent_block_id'] = f"page{page['page']}_table{table['table_id']}"
                    table['report_section'] = table['section_title']
                    table['evidence_type'] = 'table'
                    table['has_table_context'] = True
                    chunk_id += 1
                    chunks.append(table)
        
        file_content['content']['chunks'] = chunks
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

    def _split_structural_block(
        self,
        block: Dict[str, any],
        report_meta: Dict[str, any],
        chunk_size: int = 320,
        chunk_overlap: int = 50,
    ) -> List[Dict[str, any]]:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(block['text'])
        chunks_with_meta = []
        for chunk in chunks:
            chunks_with_meta.append({
                "page": block['page'],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk,
                "chunk_type": block.get("chunk_type", "content"),
                "section_title": block.get("section_title"),
                "table_id": block.get("table_id"),
                "report_year": report_meta.get("report_year"),
                "currency": report_meta.get("currency"),
                "company_name": report_meta.get("company_name"),
                "major_industry": report_meta.get("major_industry"),
                "report_type": report_meta.get("report_type"),
                "topic_flags": self._report_topic_flags(report_meta),
                "parent_block_id": block.get("parent_block_id"),
                "report_section": block.get("report_section", block.get("section_title")),
                "evidence_type": block.get("evidence_type", "narrative"),
                "has_table_context": bool(block.get("has_table_context")),
            })
        return chunks_with_meta

    def _split_page(self, page: Dict[str, any], report_meta: Dict[str, any], has_table_context: bool = False) -> List[Dict[str, any]]:
        """Split page text into structure-aware chunks."""
        all_chunks: List[Dict[str, any]] = []
        for index, block in enumerate(self._extract_structural_blocks(page)):
            block["parent_block_id"] = f"page{page['page']}_block{index}"
            block["report_section"] = block.get("section_title")
            block["evidence_type"] = "narrative"
            block["has_table_context"] = has_table_context
            all_chunks.extend(self._split_structural_block(block, report_meta))
        return all_chunks

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
