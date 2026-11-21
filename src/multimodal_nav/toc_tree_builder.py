"""PageIndex-style hierarchical TOC tree builder and read-only SQLite store for annual reports."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.storage.base import StorageError, validate_safe_path
from src.storage.sqlite_doc_store import _BoundedLRU

_DEFAULT_DOC_SQLITE = Path("data/top10_industries_2024_20each/databases_ser_tab/finarag_docs.sqlite")
_DEFAULT_TOC_SQLITE = Path("data/top10_industries_2024_20each/databases_ser_tab/finarag_toc_trees.sqlite")

_OCR_SEC_DASH_RE = re.compile(r"第\s*[-－—]\s*([节章部分])")
_OCR_ENUM_DASH_RE = re.compile(r"^([#*\s]*)[-－—]\s*([、.])")
_L1_RE = re.compile(r"^(第[一二三四五六七八九十百零0-9]+[节章部分])\s*(.{1,48})$")
_L1_SPECIAL_RE = re.compile(
    r"^(?:董事长致辞|行长致辞|总裁致辞|总经理致辞|审计报告|财务报告|公司治理报告|内部控制评价报告)$"
)
_L2_NUM_RE = re.compile(r"^(\d+\.\d+)\s+([^\n|]{2,42})$")
_L2_CN_RE = re.compile(r"^([一二三四五六七八九十]+[、.])\s*([^\n|]{2,42})$")
_L3_PAREN_RE = re.compile(r"^([（(][一二三四五六七八九十]+[）)])\s*([^\n|]{2,44})$")
_L3_NUM_RE = re.compile(r"^(\d+\.\d+\.\d+)\s+([^\n|]{2,44})$")
_MD_HASH_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_TOC_PAGE_HINT_RE = re.compile(r"(?:目\s*录|contents)", re.IGNORECASE)
_TOC_TABLE_ROW_RE = re.compile(r"^\|\s*([^|]*?)\s*\|\s*(?:([^|]*?)\s*\|\s*)?(\d{1,4})\s*\|?$")
_IGNORE_TITLES_RE = re.compile(
    r"^(?:\d{1,4}|20\d{2}\s*年年度报告|年度报告|Annual\s*Report|目\s*录|目录|重要提示|释义|"
    r"单位[：:].*|适用.*不适用.*|[√□].*|备查文件目录|[^a-zA-Z\u4e00-\u9fff]+)$",
    re.IGNORECASE,
)


def _normalize_heading_line(line: str) -> str:
    nfkc = unicodedata.normalize("NFKC", line or "")
    s = " ".join(nfkc.strip().split())
    s = _OCR_SEC_DASH_RE.sub(r"第一\1", s)
    s = _OCR_ENUM_DASH_RE.sub(r"\1一\2", s)
    return s


def _strip_md_hashes(line: str) -> Tuple[int, str]:
    m = _MD_HASH_RE.match(line)
    if not m:
        return 0, line.strip()
    return len(m.group(1)), m.group(2).strip()


def _looks_like_sentence(text: str) -> bool:
    if not text or len(text) > 54:
        return True
    if any(p in text for p in ("。", "；", ";", "，", "！", "？")):
        return True
    if _IGNORE_TITLES_RE.match(text):
        return True
    return False


def _detect_heading(line: str, *, allow_unnumbered_md: bool = False) -> Optional[Tuple[int, str]]:
    """Detect (level, clean_title) for an annual report line."""
    norm = _normalize_heading_line(line)
    if not norm or norm.startswith("|"):
        return None
    md_level, clean = _strip_md_hashes(norm)
    if not clean or _IGNORE_TITLES_RE.match(clean):
        return None

    m1 = _L1_RE.match(clean)
    if m1:
        title = f"{m1.group(1)} {m1.group(2).strip()}".strip()
        if len(title) <= 52 and not any(p in title for p in ("。", "；")):
            return 1, title

    if _L1_SPECIAL_RE.match(clean):
        return 1, clean

    if _L3_NUM_RE.match(clean) and not _looks_like_sentence(clean):
        return 3, clean
    if _L3_PAREN_RE.match(clean) and len(clean) <= 44 and not _looks_like_sentence(clean):
        return 3, clean

    if _L2_NUM_RE.match(clean) and not _looks_like_sentence(clean):
        return 2, clean
    if _L2_CN_RE.match(clean) and not _looks_like_sentence(clean):
        return 2, clean

    if allow_unnumbered_md and md_level in (1, 2, 3) and not _looks_like_sentence(clean) and len(clean) >= 3:
        return (2 if md_level <= 2 else 3), clean

    return None


@dataclass
class TocNode:
    node_id: str
    level: int
    title: str
    start_page: int
    end_page: int
    summary: str = ""
    children: List["TocNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "level": self.level,
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "summary": self.summary,
            "children": [c.to_dict() for c in self.children],
        }

    def to_compact_dict(self, max_summary_chars: int = 80, max_depth: Optional[int] = None) -> Dict[str, Any]:
        summary = self.summary[:max_summary_chars]
        payload: Dict[str, Any] = {
            "node_id": self.node_id,
            "level": self.level,
            "title": self.title,
            "pages": f"{self.start_page}-{self.end_page}",
        }
        if summary:
            payload["summary"] = summary
        if self.children and (max_depth is None or self.level < max_depth):
            children = [
                c.to_compact_dict(max_summary_chars, max_depth=max_depth)
                for c in self.children
                if max_depth is None or c.level <= max_depth
            ]
            if children:
                payload["children"] = children
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TocNode":
        return cls(
            node_id=str(data.get("node_id") or ""),
            level=int(data.get("level") or 1),
            title=str(data.get("title") or ""),
            start_page=int(data.get("start_page") or 1),
            end_page=int(data.get("end_page") or 1),
            summary=str(data.get("summary") or ""),
            children=[cls.from_dict(c) for c in (data.get("children") or [])],
        )


@dataclass
class ReportTocTree:
    doc_id: str
    company_name: str
    report_year: Optional[int]
    total_pages: int
    roots: List[TocNode] = field(default_factory=list)
    token_estimate: int = 0
    source_mode: str = "body_headings"

    def iter_nodes(self) -> Iterable[Tuple[TocNode, str]]:
        """Yield (node, full_section_path) depth-first."""
        def _walk(node: TocNode, prefix: str):
            path = f"{prefix} > {node.title}" if prefix else node.title
            yield node, path
            for child in node.children:
                yield from _walk(child, path)

        for root in self.roots:
            yield from _walk(root, "")

    def node_count(self) -> int:
        return sum(1 for _ in self.iter_nodes())

    def get_node_map(self) -> Dict[str, Tuple[TocNode, str]]:
        return {node.node_id: (node, path) for node, path in self.iter_nodes()}

    def find_section_path(self, page: int) -> str:
        """Return deepest section path covering `page`."""
        p = int(page or 1)
        best_path = ""
        best_depth = -1
        for node, path in self.iter_nodes():
            if node.start_page <= p <= node.end_page and node.level >= best_depth:
                best_depth = node.level
                best_path = path
        return best_path

    def to_compact_prompt_json(self, max_summary_chars: int = 64, max_depth: Optional[int] = None) -> str:
        effective_depth = max_depth if max_depth is not None else (3 if self.node_count() <= 120 else 2)
        payload = {
            "doc_id": self.doc_id,
            "company_name": self.company_name,
            "report_year": self.report_year,
            "total_pages": self.total_pages,
            "sections": [
                r.to_compact_dict(max_summary_chars=max_summary_chars, max_depth=effective_depth)
                for r in self.roots
            ],
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "company_name": self.company_name,
            "report_year": self.report_year,
            "total_pages": self.total_pages,
            "token_estimate": self.token_estimate,
            "source_mode": self.source_mode,
            "roots": [r.to_dict() for r in self.roots],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportTocTree":
        return cls(
            doc_id=str(data.get("doc_id") or ""),
            company_name=str(data.get("company_name") or ""),
            report_year=data.get("report_year"),
            total_pages=int(data.get("total_pages") or 0),
            roots=[TocNode.from_dict(r) for r in (data.get("roots") or [])],
            token_estimate=int(data.get("token_estimate") or 0),
            source_mode=str(data.get("source_mode") or "body_headings"),
        )


def _is_toc_page(text: str) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    head = "\n".join(lines[:12])
    if not _TOC_PAGE_HINT_RE.search(head):
        return False
    pipe_num_rows = sum(1 for ln in lines if _TOC_TABLE_ROW_RE.match(ln))
    return pipe_num_rows >= 3


def _summarize_opening_text(text: str, heading_title: str, max_chars: int = 96) -> str:
    clean_heading = heading_title.replace(" ", "")
    snippets: List[str] = []
    for raw in (text or "").splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("|") or ln.startswith("#"):
            continue
        if _IGNORE_TITLES_RE.match(ln):
            continue
        comp = ln.replace(" ", "")
        if clean_heading and (comp == clean_heading or comp in clean_heading):
            continue
        snippets.append(ln)
        if sum(len(s) for s in snippets) >= max_chars:
            break
    joined = " ".join(snippets).strip()
    if len(joined) > max_chars:
        joined = joined[:max_chars].rstrip() + "..."
    return joined


class TocTreeBuilder:
    """Deterministic zero-LLM structural TOC tree builder for A-share annual reports."""

    def build_from_pages(
        self,
        *,
        doc_id: str,
        company_name: str,
        report_year: Optional[int],
        pages: Sequence[Dict[str, Any]],
        max_summary_chars: int = 96,
    ) -> ReportTocTree:
        sorted_pages = sorted(
            (p for p in pages if isinstance(p, dict) and p.get("page") is not None),
            key=lambda p: int(p["page"]),
        )
        if not sorted_pages:
            return ReportTocTree(
                doc_id=doc_id,
                company_name=company_name,
                report_year=report_year,
                total_pages=0,
                roots=[],
                token_estimate=0,
                source_mode="empty",
            )

        page_text_map: Dict[int, str] = {int(p["page"]): str(p.get("text") or "") for p in sorted_pages}
        total_pages = max(page_text_map.keys())
        toc_entries: List[Tuple[int, str, int]] = []
        line_page_freq: Counter[str] = Counter()
        toc_page_nos: set[int] = set()

        for p in sorted_pages:
            pno = int(p["page"])
            text = page_text_map[pno]
            if _is_toc_page(text) and pno <= 12:
                toc_page_nos.add(pno)
                toc_entries.extend(self._parse_toc_table_lines(text))
                continue
            seen_in_page: set[str] = set()
            for raw_line in text.splitlines():
                norm = _normalize_heading_line(raw_line)
                if not norm or norm.startswith("|"):
                    continue
                _, clean = _strip_md_hashes(norm)
                if clean:
                    seen_in_page.add(clean)
            line_page_freq.update(seen_in_page)

        def _collect_body_hits(allow_unnumbered_md: bool) -> List[Tuple[int, str, int, str]]:
            hits: List[Tuple[int, str, int, str]] = []
            seen_keys: set[Tuple[int, str, int]] = set()
            seen_l1_titles: set[str] = set()
            last_seen_by_level: Dict[int, Tuple[str, int]] = {}
            for p in sorted_pages:
                pno = int(p["page"])
                if pno in toc_page_nos:
                    continue
                text = page_text_map[pno]
                lines = text.splitlines()
                for idx, raw_line in enumerate(lines):
                    detected = _detect_heading(raw_line, allow_unnumbered_md=allow_unnumbered_md)
                    if detected is None:
                        continue
                    level, title = detected
                    clean_norm = _strip_md_hashes(_normalize_heading_line(raw_line))[1]
                    if level == 1:
                        if title in seen_l1_titles:
                            continue
                        seen_l1_titles.add(title)
                    else:
                        if line_page_freq[clean_norm] >= 4 or line_page_freq[title] >= 4:
                            continue
                        prev = last_seen_by_level.get(level)
                        if prev is not None and prev[0] == title and (pno - prev[1]) <= 3:
                            continue
                    last_seen_by_level[level] = (title, pno)
                    key = (level, title, pno)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    after_text = "\n".join(lines[idx + 1 :])
                    summary = _summarize_opening_text(after_text or text, title, max_chars=max_summary_chars)
                    hits.append((level, title, pno, summary))
            return hits

        body_hits = _collect_body_hits(allow_unnumbered_md=False)
        if len(body_hits) < 5 and not toc_entries:
            body_hits = _collect_body_hits(allow_unnumbered_md=True)

        l1_body_count = sum(1 for lvl, _, _, _ in body_hits if lvl == 1)
        source_mode = "body_headings"
        flat_items: List[Tuple[int, str, int, str]] = body_hits

        if l1_body_count < 3 and toc_entries:
            source_mode = "toc_table_aligned"
            aligned_toc = self._align_toc_entries(toc_entries, page_text_map, total_pages, max_summary_chars)
            flat_items = self._merge_toc_and_body(aligned_toc, body_hits)
        elif toc_entries and len(body_hits) < 12:
            source_mode = "hybrid_toc_body"
            aligned_toc = self._align_toc_entries(toc_entries, page_text_map, total_pages, max_summary_chars)
            flat_items = self._merge_toc_and_body(aligned_toc, body_hits)

        if not flat_items:
            # Coarse uniform fallback (20-page windows) for degraded scan PDFs
            source_mode = "page_windows"
            flat_items = [
                (
                    1,
                    f"第{start}-{min(total_pages, start + 19)}页",
                    start,
                    _summarize_opening_text(page_text_map.get(start, ""), "", max_chars=max_summary_chars),
                )
                for start in range(1, total_pages + 1, 20)
            ]

        roots = self._assemble_hierarchy(flat_items, total_pages)
        tree = ReportTocTree(
            doc_id=doc_id,
            company_name=company_name,
            report_year=report_year,
            total_pages=total_pages,
            roots=roots,
            token_estimate=0,
            source_mode=source_mode,
        )
        compact_str = tree.to_compact_prompt_json(max_summary_chars=48)
        tree.token_estimate = max(1, int(len(compact_str) / 2.2))
        return tree

    @staticmethod
    def _parse_toc_table_lines(text: str) -> List[Tuple[int, str, int]]:
        entries: List[Tuple[int, str, int]] = []
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            m = _TOC_TABLE_ROW_RE.match(line)
            if not m:
                continue
            col1 = _normalize_heading_line(m.group(1) or "")
            col2 = _normalize_heading_line(m.group(2) or "")
            page_str = m.group(3)
            if not page_str:
                continue
            printed_page = int(page_str)
            combined = f"{col1} {col2}".strip()
            if not combined or "目录" in combined and len(combined) <= 6:
                continue
            detected = _detect_heading(combined)
            if detected is not None:
                lvl, title = detected
            else:
                # 2-column chapter TOC row without section prefix
                lvl = 1 if re.search(r"第[一二三四五六七八九十0-9]+[节章]|致辞|公司", combined) else 2
                title = combined[:48]
            entries.append((lvl, title, printed_page))
        return entries

    @staticmethod
    def _align_toc_entries(
        toc_entries: List[Tuple[int, str, int]],
        page_text_map: Dict[int, str],
        total_pages: int,
        max_summary_chars: int,
    ) -> List[Tuple[int, str, int, str]]:
        offsets: List[int] = []
        compact_pages = {
            pno: unicodedata.normalize("NFKC", txt).replace(" ", "")
            for pno, txt in page_text_map.items()
        }
        for _, title, printed_page in toc_entries[:15]:
            needle = title.split()[-1].replace(" ", "")
            if len(needle) < 3:
                continue
            for pno, ctxt in compact_pages.items():
                if pno >= printed_page and needle in ctxt[:600]:
                    offsets.append(pno - printed_page)
                    break
        median_offset = sorted(offsets)[len(offsets) // 2] if offsets else 0
        out: List[Tuple[int, str, int, str]] = []
        for lvl, title, printed_page in toc_entries:
            est_page = max(1, min(total_pages, printed_page + median_offset))
            summary = _summarize_opening_text(page_text_map.get(est_page, ""), title, max_chars=max_summary_chars)
            out.append((lvl, title, est_page, summary))
        return out

    @staticmethod
    def _merge_toc_and_body(
        toc_items: List[Tuple[int, str, int, str]],
        body_items: List[Tuple[int, str, int, str]],
    ) -> List[Tuple[int, str, int, str]]:
        merged = list(toc_items)
        seen_titles = {(lvl, title.replace(" ", "")) for lvl, title, _, _ in merged}
        for lvl, title, pno, summary in body_items:
            key = (lvl, title.replace(" ", ""))
            if key not in seen_titles:
                seen_titles.add(key)
                merged.append((lvl, title, pno, summary))
        merged.sort(key=lambda item: (item[2], item[0]))
        return merged

    @staticmethod
    def _assemble_hierarchy(
        flat_items: List[Tuple[int, str, int, str]],
        total_pages: int,
    ) -> List[TocNode]:
        roots: List[TocNode] = []
        stack: List[TocNode] = []
        counter = 0

        for level, title, start_page, summary in sorted(flat_items, key=lambda x: (x[2], x[0])):
            counter += 1
            node = TocNode(
                node_id=f"sec-{counter:04d}",
                level=int(level),
                title=title,
                start_page=max(1, min(total_pages, int(start_page))),
                end_page=total_pages,
                summary=summary,
            )
            while stack and stack[-1].level >= node.level:
                stack.pop()
            if not stack:
                roots.append(node)
            else:
                stack[-1].children.append(node)
            stack.append(node)

        def _assign_end_pages(nodes: List[TocNode], upper_bound: int) -> None:
            for idx, n in enumerate(nodes):
                if idx + 1 < len(nodes):
                    next_start = nodes[idx + 1].start_page
                    child_end = max(n.start_page, next_start - 1) if next_start > n.start_page else next_start
                    child_end = min(upper_bound, child_end)
                else:
                    child_end = upper_bound
                n.end_page = max(n.start_page, child_end)
                if n.children:
                    _assign_end_pages(n.children, n.end_page)
                    n.end_page = max(n.end_page, max(c.end_page for c in n.children))

        _assign_end_pages(roots, max(1, total_pages))
        return roots


class TocTreeStore:
    """Read-only SQLite store + LRU cache for annual report PageIndex TOC trees."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        allowed_base_dir: Optional[Path | str] = None,
        cache_size: int = 64,
    ):
        self.db_path = validate_safe_path(db_path, allowed_base_dir)
        self._cache = _BoundedLRU(cache_size)
        self._local = threading.local()
        self._conns: List[sqlite3.Connection] = []
        self._lock = threading.Lock()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        if not self.db_path.exists():
            raise StorageError("TOC tree SQLite database does not exist")
        try:
            conn = sqlite3.connect(f"{self.db_path.as_uri()}?mode=ro", uri=True, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
        except sqlite3.Error as exc:
            raise StorageError("Failed to open TOC tree SQLite store") from exc
        self._local.conn = conn
        with self._lock:
            self._conns.append(conn)
        return conn

    def get_tree(self, doc_id: str) -> Optional[ReportTocTree]:
        key = str(doc_id or "").strip()
        if not key:
            return None
        cached = self._cache.get(key, default=None)
        if cached is not None:
            return cached
        try:
            row = self._get_conn().execute(
                "SELECT tree_json FROM toc_trees WHERE doc_id = ?",
                (key,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError("TOC tree lookup failed") from exc
        if row is None:
            return None
        tree = ReportTocTree.from_dict(json.loads(row["tree_json"]))
        self._cache.put(key, tree)
        return tree

    def get_tree_by_company(self, company_name: str, report_year: Optional[int] = None) -> Optional[ReportTocTree]:
        comp = str(company_name or "").strip()
        if not comp:
            return None
        try:
            if report_year is not None:
                row = self._get_conn().execute(
                    "SELECT doc_id FROM toc_trees WHERE company_name = ? AND report_year = ? LIMIT 1",
                    (comp, int(report_year)),
                ).fetchone()
            else:
                row = self._get_conn().execute(
                    "SELECT doc_id FROM toc_trees WHERE company_name = ? ORDER BY report_year DESC LIMIT 1",
                    (comp,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError("TOC company lookup failed") from exc
        if row is None:
            return None
        return self.get_tree(str(row["doc_id"]))

    def summary_stats(self) -> Dict[str, Any]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT node_count, token_estimate, source_mode FROM toc_trees"
        ).fetchall()
        node_counts = sorted(int(r["node_count"]) for r in rows)
        toks = sorted(int(r["token_estimate"]) for r in rows)
        modes: Dict[str, int] = {}
        for r in rows:
            m = str(r["source_mode"])
            modes[m] = modes.get(m, 0) + 1
        n = len(rows)
        return {
            "reports_count": n,
            "multi_node_ge_5": sum(1 for c in node_counts if c >= 5),
            "median_node_count": node_counts[n // 2] if n else 0,
            "median_token_estimate": toks[n // 2] if n else 0,
            "p95_token_estimate": toks[int((n - 1) * 0.95)] if n else 0,
            "source_modes": modes,
        }

    def close(self) -> None:
        with self._lock:
            for c in self._conns:
                try:
                    c.close()
                except Exception:
                    pass
            self._conns.clear()
        if hasattr(self._local, "conn"):
            delattr(self._local, "conn")
        self._cache.clear()

    @staticmethod
    def _write_trees_sqlite(
        trees: Sequence[ReportTocTree],
        out_db_path: Path | str,
        *,
        allowed_base_dir: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        out_path = validate_safe_path(out_db_path, allowed_base_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f".{out_path.name}.tmp")
        for suffix in ("", "-wal", "-shm"):
            p = Path(f"{tmp_path}{suffix}")
            if p.exists():
                p.unlink()

        conn = sqlite3.connect(str(tmp_path))
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS toc_trees (
                    doc_id         TEXT PRIMARY KEY,
                    company_name   TEXT,
                    report_year    INTEGER,
                    total_pages    INTEGER,
                    node_count     INTEGER,
                    token_estimate INTEGER,
                    source_mode    TEXT,
                    tree_json      TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_toc_company_year ON toc_trees(company_name, report_year);
                """
            )
            batch = [
                (
                    t.doc_id,
                    t.company_name,
                    t.report_year,
                    t.total_pages,
                    t.node_count(),
                    t.token_estimate,
                    t.source_mode,
                    json.dumps(t.to_dict(), ensure_ascii=False),
                )
                for t in trees
            ]
            conn.executemany(
                """
                INSERT OR REPLACE INTO toc_trees (
                    doc_id, company_name, report_year, total_pages,
                    node_count, token_estimate, source_mode, tree_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()

        os.replace(tmp_path, out_path)
        for suffix in ("-wal", "-shm"):
            leftover = Path(f"{tmp_path}{suffix}")
            if leftover.exists():
                leftover.unlink()

        store = TocTreeStore(out_path, allowed_base_dir=allowed_base_dir)
        try:
            stats = store.summary_stats()
            stats["db_path"] = str(out_path)
            return stats
        finally:
            store.close()

    @staticmethod
    def build_from_sqlite_doc_repo(
        doc_sqlite_path: Path | str,
        out_db_path: Path | str,
        *,
        limit: Optional[int] = None,
        allowed_base_dir: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        from src.storage.sqlite_doc_store import SqliteDocumentRepository

        repo = SqliteDocumentRepository(doc_sqlite_path, allowed_base_dir=allowed_base_dir)
        builder = TocTreeBuilder()
        trees: List[ReportTocTree] = []
        try:
            metas = repo.list_metainfo()
            if limit is not None:
                metas = metas[: max(0, int(limit))]
            for meta in metas:
                did = str(meta.get("sha1_name") or meta.get("doc_id") or "")
                if not did:
                    continue
                pages = repo.get_pages(did)
                raw_year = meta.get("report_year") or meta.get("fiscal_year")
                try:
                    ry = int(raw_year) if raw_year is not None else None
                except (TypeError, ValueError):
                    ry = None
                trees.append(
                    builder.build_from_pages(
                        doc_id=did,
                        company_name=str(meta.get("company_name") or ""),
                        report_year=ry,
                        pages=pages,
                    )
                )
        finally:
            repo.close()
        return TocTreeStore._write_trees_sqlite(trees, out_db_path, allowed_base_dir=allowed_base_dir)

    @staticmethod
    def build_from_chunked_dir(
        chunked_dir: Path | str,
        out_db_path: Path | str,
        *,
        limit: Optional[int] = None,
        allowed_base_dir: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        reports_dir = Path(chunked_dir).expanduser().resolve(strict=False)
        files = sorted(reports_dir.glob("*.json"))
        if limit is not None:
            files = files[: max(0, int(limit))]
        builder = TocTreeBuilder()
        trees: List[ReportTocTree] = []
        for fp in files:
            data = json.loads(fp.read_text(encoding="utf-8"))
            meta = data.get("metainfo") or {}
            pages = (data.get("content") or {}).get("pages") or []
            did = str(meta.get("sha1_name") or meta.get("doc_id") or fp.stem)
            raw_year = meta.get("report_year") or meta.get("fiscal_year")
            try:
                ry = int(raw_year) if raw_year is not None else None
            except (TypeError, ValueError):
                ry = None
            trees.append(
                builder.build_from_pages(
                    doc_id=did,
                    company_name=str(meta.get("company_name") or ""),
                    report_year=ry,
                    pages=pages,
                )
            )
        return TocTreeStore._write_trees_sqlite(trees, out_db_path, allowed_base_dir=allowed_base_dir)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build PageIndex TOC trees SQLite DB across annual reports.")
    parser.add_argument("--doc-sqlite", type=Path, default=_DEFAULT_DOC_SQLITE)
    parser.add_argument("--chunked-dir", type=Path, default=None)
    parser.add_argument("--out-db", type=Path, default=_DEFAULT_TOC_SQLITE)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    if args.chunked_dir is not None:
        stats = TocTreeStore.build_from_chunked_dir(args.chunked_dir, args.out_db, limit=args.limit)
    else:
        stats = TocTreeStore.build_from_sqlite_doc_repo(args.doc_sqlite, args.out_db, limit=args.limit)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
