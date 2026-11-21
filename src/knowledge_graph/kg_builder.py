"""Targeted 4-section Financial Knowledge Graph builder and read-only SQLite WAL store."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from json_repair import repair_json

from src.storage.base import StorageError, validate_safe_path

_DEFAULT_DOC_SQLITE = Path("data/top10_industries_2024_20each/databases_ser_tab/finarag_docs.sqlite")
_DEFAULT_TOC_SQLITE = Path("data/top10_industries_2024_20each/databases_ser_tab/finarag_toc_trees.sqlite")
_DEFAULT_KG_SQLITE = Path("data/top10_industries_2024_20each/databases_ser_tab/finarag_kg.sqlite")

# Strict 4-category scope matching high-relation-density annual report sections
_SECTION_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "equity",
        re.compile(
            r"(?:控股股东|实际控制人|前十名股东|前10名股东|主要控股参股|主要子公司|控股参股|主要参股|参股公司|投资企业|股权投资)"
        ),
    ),
    (
        "related_tx",
        re.compile(r"(?:重大关联交易|日常关联交易|关联交易|关联方|重大担保|违规担保|担保情况|担保事项|非经营性资金占用)"),
    ),
    (
        "customers_suppliers",
        re.compile(r"(?:前五名客户|前五大客户|主要销售客户|前五名供应商|前五大供应商|主要供应商)"),
    ),
    (
        "supply_chain_competition",
        re.compile(r"(?:供应链|核心竞争力|行业竞争格局|上下游|可能面对的风险)"),
    ),
)

_ANONYMIZED_ENTITY_RE = re.compile(
    r"^(?:(?:客户|供应商|单位|公司|交易方|对象|股东)(?:第)?[一二三四五六七八九十甲乙丙丁A-Ea-e0-9\-－—]{1,3}(?:名|号)?|"
    r"第[一二三四五六七八九十0-9\-－—]{1,3}名|(?:Customer|Supplier)\s*[A-E0-9]+)$",
    re.IGNORECASE,
)

_ENTITY_COL_HINTS = (
    "交易对方",
    "关联方",
    "法人股东名称",
    "股东名称",
    "被担保方",
    "担保对象",
    "担保方",
    "客户名称",
    "主要客户",
    "供应商名称",
    "主要供应商",
    "公司名称",
    "单位名称",
    "企业名称",
    "子公司名称",
    "参股公司",
    "股权投资项目",
    "投资企业",
    "名称",
)

_BAD_ENTITY_EXACT = frozenset(
    {
        "合计",
        "小计",
        "总计",
        "其中",
        "其他",
        "不适用",
        "适用",
        "无",
        "名称",
        "公司名称",
        "单位名称",
        "企业名称",
        "交易对方",
        "关联方",
        "股东名称",
        "客户名称",
        "供应商名称",
        "被担保方",
        "担保方",
        "担保对象",
        "项目",
        "序号",
        "排名",
        "占比",
        "金额",
        "备注",
        "说明",
        "是",
        "否",
        "人民币",
        "本集团",
        "本公司",
        "本行",
        "合并",
        "母公司",
        "长期股权投资",
        "其他权益工具投资",
        "交易性金融资产",
        "营业收入",
        "净利润",
        "货币资金",
        "法定代表人",
        "总经理",
        "董事长",
        "其他关联方",
        "其他应收款",
        "关联债权债务形成原因",
        "私募基金",
        "联营企业",
        "合营企业",
        "供应商",
        "客户",
        "股东",
        "其他股东",
        "关联自然人",
    }
)

_CORP_SUFFIX_RE = re.compile(
    r"(?:股份有限公司|有限责任公司|有限公司|集团公司|集团|控股|银行|信托|证券|基金|保险|租赁|"
    r"财务公司|国资委|委员会|人民政府|政府|交易所|Co\.,?\s*Ltd\.?|Inc\.?|Corp\.?)$",
    re.IGNORECASE,
)
_BARE_CORP_SUFFIXES = frozenset({"有限公司", "股份有限公司", "有限责任公司", "集团有限公司", "股份公司", "集团公司", "公司", "集团"})
_BAD_ENTITY_CONTAINS = (
    "说明",
    "情况",
    "合计",
    "总额",
    "余额",
    "发生额",
    "报告期",
    "前十名",
    "前五名",
    "有限售",
    "无限售",
    "转融通",
    "回购",
    "法定代表",
    "负责人",
    "全称",
    "简称",
    "代码",
    "营业收入",
    "净利润",
    "同比增长",
    "占公司",
    "被投资",
    "主要经营",
    "成立日期",
    "追加投资",
    "对子公司",
    "对联营",
    "应收账款",
    "应收款项",
    "预付款项",
    "应付账款",
    "资金占用",
    "单位:",
    "单位：",
)
_BAD_ENTITY_ENDSWITH = ("日期", "时间", "金额", "比例", "用途", "期间", "方式", "类型", "数量", "股权", "账款", "融资")
_FULL_REG_NAME_RE = re.compile(r"([\u4e00-\u9fff]{3,28}(?:股份有限公司|有限责任公司|集团有限公司))")
_OCR_CJK_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
_TRAIL_NOTE_RE = re.compile(r"(?:\s*[（(]?(?:注\s*\d+|\d+)[)）]?)$")
_PURE_NUM_RE = re.compile(r"^[0-9.,%+\-—/年月日至:\s()（）]+$")
_PCT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
_AMOUNT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(万亿元|亿元|百万元|万元|千元|元)?")

_HOLD_PROSE_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9()（）·]{3,36}(?:有限公司|股份有限公司|集团|银行|信托|证券|基金|租赁|国资委|委员会))[^。\n]{0,40}?"
    r"(?:公司|本公司|本行|集团)?持有(?:其|公司|本公司|本行)?\s*(\d{1,3}(?:\.\d+)?)\s*%"
)
_CTRL_SHAREHOLDER_PROSE_RE = re.compile(
    r"(?:控股股东为|实际控制人为)[：:\s]*([\u4e00-\u9fffA-Za-z0-9()（）·]{2,40}?(?:国资委|委员会|人民政府|有限公司|股份有限公司|集团公司|集团|局|公司))"
)
_KV_CONTROLLER_RE = re.compile(
    r"(?:实际控制人|最终控制方|控股股东)[^\n]{0,16}(?:名称)?\s*[:：\n]+\s*([\u4e00-\u9fffA-Za-z0-9()（）·]{2,40})"
)


@dataclass
class EntityNode:
    entity_id: str
    canonical_name: str
    entity_type: str = "organization"
    aliases: List[str] = field(default_factory=list)
    doc_ids: List[str] = field(default_factory=list)
    degree: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "aliases": list(self.aliases),
            "doc_ids": list(self.doc_ids),
            "degree": self.degree,
        }


@dataclass
class RelationEdge:
    edge_id: str
    head_entity: str
    relation_type: str
    tail_entity: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    evidence_doc_id: str = ""
    evidence_page: int = 1
    report_year: Optional[int] = None
    section_title: str = ""
    evidence_snippet: str = ""
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "head_entity": self.head_entity,
            "relation_type": self.relation_type,
            "tail_entity": self.tail_entity,
            "attributes": dict(self.attributes),
            "evidence_doc_id": self.evidence_doc_id,
            "evidence_page": self.evidence_page,
            "report_year": self.report_year,
            "section_title": self.section_title,
            "evidence_snippet": self.evidence_snippet,
            "confidence": self.confidence,
        }


def canonicalize_entity_name(
    raw_name: str,
    *,
    company_scope: Optional[str] = None,
    alias_to_canonical: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """Normalize OCR-split CJK corporate names & isolate anonymized counterparties."""
    s = unicodedata.normalize("NFKC", str(raw_name or "")).strip()
    s = re.sub(r"^[|#*\-\s]+|[|*\s]+$", "", s)
    s = _OCR_CJK_SPACE_RE.sub("", s)
    s = " ".join(s.split())
    s = _TRAIL_NOTE_RE.sub("", s).strip(" ：:;；，,.")
    if not s:
        return "", ""

    if _ANONYMIZED_ENTITY_RE.match(s):
        scoped = f"{company_scope}::{s}" if company_scope else f"ANON::{s}"
        return scoped, "anonymized_counterparty"

    if alias_to_canonical:
        mapped = alias_to_canonical.get(s)
        if mapped:
            return mapped, "organization"
        compact = s.replace(" ", "")
        if compact in alias_to_canonical:
            return alias_to_canonical[compact], "organization"

    etype = "person" if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", s) else "organization"
    return s, etype


def _is_valid_entity_candidate(name: str, *, allow_person: bool = False) -> bool:
    if not name or len(name) < 2 or len(name) > 56:
        return False
    if name in _BAD_ENTITY_EXACT or name in _BARE_CORP_SUFFIXES or _PURE_NUM_RE.match(name):
        return False
    if (name.startswith(("(", "（")) and name.endswith((")", "）"))) or name.endswith(("-", "－")):
        return False
    if any(ch in name for ch in ("。", "；", "？", "！", "|", "=")):
        return False
    if name.count("，") >= 1 or name.count(",") >= 2:
        return False
    if any(bad in name for bad in _BAD_ENTITY_CONTAINS):
        return False
    if "::" in name:
        return True
    if _CORP_SUFFIX_RE.search(name):
        return len(name) >= 4
    if name.endswith(_BAD_ENTITY_ENDSWITH):
        return False
    if allow_person and re.fullmatch(r"[\u4e00-\u9fff·]{2,6}", name):
        return True
    # Non-generic named entities >= 3 chars (short listed company names / partners / research institutes)
    return bool(re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9()（）·&.\-]{3,28}", name)) and not name[0].isdigit()


def classify_target_section(title: str) -> Optional[str]:
    norm = unicodedata.normalize("NFKC", str(title or ""))
    for category, pattern in _SECTION_PATTERNS:
        if pattern.search(norm):
            return category
    return None


def _entity_id_for(name: str) -> str:
    return "ent-" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]


def _edge_id_for(head: str, rel: str, tail: str, doc_id: str, page: int) -> str:
    raw = f"{head}|{rel}|{tail}|{doc_id}|{page}"
    return "edge-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _parse_pipe_tables(page_text: str) -> List[Tuple[List[str], List[List[str]]]]:
    """Extract (headers, data_rows) from Markdown pipe tables in normalized page text."""
    tables: List[Tuple[List[str], List[List[str]]]] = []
    lines = unicodedata.normalize("NFKC", page_text or "").splitlines()
    block: List[str] = []

    def _flush() -> None:
        nonlocal block
        if len(block) >= 2:
            parsed = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in block]
            header = parsed[0]
            data_rows: List[List[str]] = []
            for row in parsed[1:]:
                if all(re.fullmatch(r"[:\-\s]*", c) for c in row):
                    continue
                data_rows.append(row)
            if header and data_rows:
                tables.append((header, data_rows))
        block = []

    for ln in lines:
        s = ln.strip()
        if s.startswith("|") and "|" in s[1:]:
            block.append(s)
        else:
            _flush()
    _flush()
    return tables


def _find_entity_col(headers: Sequence[str], rows: Sequence[Sequence[str]], section_cat: str) -> Optional[int]:
    norm_headers = [_OCR_CJK_SPACE_RE.sub("", unicodedata.normalize("NFKC", h)) for h in headers]
    for hint in _ENTITY_COL_HINTS:
        for idx, h in enumerate(norm_headers):
            if hint in h:
                return idx
    # Fallback: pick leftmost non-index column whose values contain corporate suffixes
    for idx in range(min(3, len(headers))):
        corp_hits = sum(1 for r in rows[:8] if idx < len(r) and _CORP_SUFFIX_RE.search(_OCR_CJK_SPACE_RE.sub("", r[idx])))
        if corp_hits >= 1:
            return idx
    return None


def _relation_for_section(section_cat: str, section_title: str, headers_blob: str) -> str:
    combined = f"{section_title} {headers_blob}"
    if any(k in combined for k in ("控股股东", "股东名称", "法人股东")):
        return "MAJOR_SHAREHOLDER"
    if "实际控制人" in combined:
        return "ACTUAL_CONTROLLER"
    if any(k in combined for k in ("子公司", "控股参股", "主要控股")):
        return "HOLDS_SUBSIDIARY"
    if any(k in combined for k in ("参股", "股权投资", "联营", "合营")):
        return "EQUITY_INVESTMENT"
    if any(k in combined for k in ("担保",)):
        return "GUARANTEES"
    if any(k in combined for k in ("关联交易", "关联方", "交易对方")):
        return "RELATED_PARTY_TX"
    if any(k in combined for k in ("前五名客户", "前五大客户", "主要销售客户", "客户名称", "客户")):
        return "TOP_CUSTOMER"
    if any(k in combined for k in ("前五名供应商", "前五大供应商", "主要供应商", "供应商名称", "供应商")):
        return "TOP_SUPPLIER"
    if section_cat == "equity":
        return "EQUITY_INVESTMENT"
    if section_cat == "related_tx":
        return "RELATED_PARTY_TX"
    if section_cat == "customers_suppliers":
        return "TOP_SUPPLIER"
    return "SUPPLY_CHAIN_DEPENDENCY"


def _row_attributes(headers: Sequence[str], row: Sequence[str], entity_col: int) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for idx, cell in enumerate(row):
        if idx == entity_col:
            continue
        c = unicodedata.normalize("NFKC", str(cell or "")).strip()
        if not c or c in {"-", "—", "/"}:
            continue
        h = unicodedata.normalize("NFKC", str(headers[idx] if idx < len(headers) else f"col_{idx}")).strip()
        if ("比例" in h or "持股" in h or "%" in c) and len(c) <= 16:
            attrs.setdefault("ratio", c[:16])
        elif ("金额" in h or "资产" in h or "收入" in h or "净利润" in h) and len(c) <= 24:
            attrs.setdefault("amount", c[:24])
        elif ("类型" in h or "方式" in h or "业务" in h or "性质" in h) and len(c) <= 32:
            attrs.setdefault("detail", c[:32])
    return attrs


def _hash_entity_vector(text: str, dim: int = 32) -> List[float]:
    vec = np.zeros(dim, dtype=np.float32)
    s = unicodedata.normalize("NFKC", text or "").strip()
    grams = [s[i : i + 2] for i in range(max(1, len(s) - 1))]
    for g in grams:
        digest = hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest()
        idx = int.from_bytes(digest[:2], "little") % dim
        sign = 1.0 if (digest[2] & 1) == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec /= norm
    return vec.round(6).tolist()


class FinancialKGBuilder:
    """Extracts targeted 4-section financial entities & relations into a deduplicated graph."""

    @staticmethod
    def build_alias_map_from_reports(reports: Sequence[Tuple[Dict[str, Any], Sequence[Dict[str, Any]]]]) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        for meta, pages in reports:
            short = str(meta.get("company_name") or "").strip()
            if not short:
                continue
            alias_map[short] = short
            for p in pages[:8]:
                txt = _OCR_CJK_SPACE_RE.sub("", unicodedata.normalize("NFKC", str(p.get("text") or "")))
                for m in _FULL_REG_NAME_RE.finditer(txt[:1500]):
                    full_name = m.group(1)
                    if short[:2] in full_name:
                        alias_map[full_name] = short
        return alias_map

    def _target_pages_for_report(
        self,
        pages: Sequence[Dict[str, Any]],
        toc_tree: Optional[Any] = None,
    ) -> Dict[int, Tuple[str, str]]:
        """Identify (page_no -> (section_category, section_title)) inside the 4 target section families."""
        target_pages: Dict[int, Tuple[str, str]] = {}
        if toc_tree is not None:
            # Deepest nodes override parent chapter titles cleanly while capping oversized spans
            nodes_sorted = sorted(toc_tree.iter_nodes(), key=lambda npair: (npair[0].level, npair[0].start_page))
            for node, path in nodes_sorted:
                cat = classify_target_section(node.title) or classify_target_section(path)
                if not cat:
                    continue
                end_cap = min(int(node.end_page), int(node.start_page) + 5)
                for pno in range(int(node.start_page), end_cap + 1):
                    target_pages[pno] = (cat, node.title)

        # Supplement / fallback with direct heading lines on pages
        for p in pages:
            pno = int(p.get("page") or 0)
            if pno <= 0 or pno in target_pages:
                continue
            txt = unicodedata.normalize("NFKC", str(p.get("text") or ""))
            for ln in txt.splitlines()[:18]:
                clean = ln.strip().lstrip("# ").strip()
                if len(clean) > 48:
                    continue
                cat = classify_target_section(clean)
                if cat:
                    target_pages[pno] = (cat, clean)
                    break
        return target_pages

    def extract_from_report(
        self,
        *,
        doc_id: str,
        company_name: str,
        report_year: Optional[int],
        pages: Sequence[Dict[str, Any]],
        toc_tree: Optional[Any] = None,
        alias_to_canonical: Optional[Dict[str, str]] = None,
        llm_extract_fn: Optional[Callable[[str, str], str]] = None,
    ) -> Tuple[List[EntityNode], List[RelationEdge]]:
        head_canon, _ = canonicalize_entity_name(company_name, alias_to_canonical=alias_to_canonical)
        if not head_canon:
            return [], []

        page_map = {int(p["page"]): str(p.get("text") or "") for p in pages if isinstance(p, dict) and p.get("page") is not None}
        target_pages = self._target_pages_for_report(pages, toc_tree=toc_tree)

        edges_by_id: Dict[str, RelationEdge] = {}
        entities: Dict[str, EntityNode] = {}

        def _ensure_entity(cname: str, etype: str, raw_alias: str = "") -> None:
            eid = _entity_id_for(cname)
            node = entities.get(eid)
            if node is None:
                node = EntityNode(
                    entity_id=eid,
                    canonical_name=cname,
                    entity_type=etype or "organization",
                    aliases=[],
                    doc_ids=[doc_id],
                )
                entities[eid] = node
            if doc_id not in node.doc_ids:
                node.doc_ids.append(doc_id)
            if raw_alias and raw_alias != cname and raw_alias not in node.aliases:
                node.aliases.append(raw_alias)

        def _add_edge(
            h_name: str,
            h_type: str,
            rel: str,
            t_name: str,
            t_type: str,
            page: int,
            sec_title: str,
            snippet: str,
            attrs: Dict[str, Any],
            conf: float,
        ) -> None:
            if not h_name or not t_name or h_name == t_name:
                return
            _ensure_entity(h_name, h_type)
            _ensure_entity(t_name, t_type)
            eid = _edge_id_for(h_name, rel, t_name, doc_id, page)
            if eid in edges_by_id:
                return
            edges_by_id[eid] = RelationEdge(
                edge_id=eid,
                head_entity=h_name,
                relation_type=rel,
                tail_entity=t_name,
                attributes=attrs,
                evidence_doc_id=doc_id,
                evidence_page=page,
                report_year=report_year,
                section_title=sec_title[:80],
                evidence_snippet=snippet[:180],
                confidence=round(conf, 3),
            )

        _ensure_entity(head_canon, "organization")

        for pno, (cat, sec_title) in sorted(target_pages.items()):
            raw_text = page_map.get(pno, "")
            if not raw_text.strip():
                continue
            norm_text = unicodedata.normalize("NFKC", raw_text)

            # 1) Markdown pipe table extraction inside target sections
            for headers, rows in _parse_pipe_tables(norm_text):
                ecol = _find_entity_col(headers, rows, cat)
                if ecol is None:
                    continue
                headers_blob = _OCR_CJK_SPACE_RE.sub("", " ".join(headers))
                rel = _relation_for_section(cat, sec_title, headers_blob)
                for row in rows:
                    if ecol >= len(row):
                        continue
                    raw_cell = row[ecol]
                    cname, ctype = canonicalize_entity_name(
                        raw_cell,
                        company_scope=head_canon,
                        alias_to_canonical=alias_to_canonical,
                    )
                    if not _is_valid_entity_candidate(cname, allow_person=(rel in {"ACTUAL_CONTROLLER", "MAJOR_SHAREHOLDER"})):
                        continue
                    attrs = _row_attributes(headers, row, ecol)
                    # Orient parent->child ownership/shareholder cleanly for graph traversal
                    if rel in {"ACTUAL_CONTROLLER", "MAJOR_SHAREHOLDER"}:
                        _add_edge(cname, ctype, rel, head_canon, "organization", pno, sec_title, " | ".join(row[:4]), attrs, 0.9)
                    else:
                        _add_edge(head_canon, "organization", rel, cname, ctype, pno, sec_title, " | ".join(row[:4]), attrs, 0.88)

            # 2) Prose patterns for controlling shareholder / actual controller / subsidiary equity %
            for m in _CTRL_SHAREHOLDER_PROSE_RE.finditer(norm_text):
                raw_ent = m.group(1)
                cname, ctype = canonicalize_entity_name(raw_ent, alias_to_canonical=alias_to_canonical)
                if _is_valid_entity_candidate(cname, allow_person=True):
                    rel = "ACTUAL_CONTROLLER" if "实际控制人" in m.group(0) else "CONTROLLING_SHAREHOLDER"
                    _add_edge(cname, ctype, rel, head_canon, "organization", pno, sec_title, m.group(0), {}, 0.92)

            if "控股股东" in norm_text or "实际控制人" in norm_text:
                lines = [ln.strip() for ln in norm_text.splitlines() if ln.strip()]
                for idx, ln in enumerate(lines[:-1]):
                    if ln in {"名称", "姓名", "单位名称"} and idx > 0 and any(k in lines[max(0, idx - 3) : idx] for k in ("控股股东", "实际控制人")):
                        nxt = lines[idx + 1]
                        cname, ctype = canonicalize_entity_name(nxt, alias_to_canonical=alias_to_canonical)
                        if _is_valid_entity_candidate(cname, allow_person=True):
                            rel = "ACTUAL_CONTROLLER" if any("实际控制人" in x for x in lines[max(0, idx - 3) : idx]) else "CONTROLLING_SHAREHOLDER"
                            _add_edge(cname, ctype, rel, head_canon, "organization", pno, sec_title, f"{ln}: {nxt}", {}, 0.86)

            for m in _HOLD_PROSE_RE.finditer(norm_text):
                sub_name, ratio = m.group(1), m.group(2)
                cname, ctype = canonicalize_entity_name(sub_name, alias_to_canonical=alias_to_canonical)
                if _is_valid_entity_candidate(cname):
                    rel = "HOLDS_SUBSIDIARY" if float(ratio) >= 50.0 else "EQUITY_INVESTMENT"
                    _add_edge(head_canon, "organization", rel, cname, ctype, pno, sec_title, m.group(0), {"ratio": f"{ratio}%"}, 0.9)

            # 3) Optional fail-closed LLM triple extractor hook
            if llm_extract_fn is not None:
                try:
                    llm_raw = llm_extract_fn(norm_text[:3000], head_canon)
                    parsed = json.loads(repair_json(str(llm_raw or "")))
                    triples = parsed.get("triples") if isinstance(parsed, dict) else parsed
                    if isinstance(triples, list):
                        for item in triples[:20]:
                            if not isinstance(item, dict):
                                continue
                            h, htype = canonicalize_entity_name(item.get("head") or head_canon, company_scope=head_canon, alias_to_canonical=alias_to_canonical)
                            t, ttype = canonicalize_entity_name(item.get("tail") or "", company_scope=head_canon, alias_to_canonical=alias_to_canonical)
                            rel = str(item.get("relation") or "RELATED_PARTY_TX").upper()
                            if _is_valid_entity_candidate(h, allow_person=True) and _is_valid_entity_candidate(t, allow_person=True):
                                _add_edge(h, htype, rel, t, ttype, pno, sec_title, str(item.get("evidence") or ""), dict(item.get("attributes") or {}), 0.85)
                except Exception:
                    pass

        return list(entities.values()), list(edges_by_id.values())


class FinancialKGStore:
    """Thread-safe read-only SQLite WAL store + cached NetworkX MultiDiGraph."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        allowed_base_dir: Optional[Path | str] = None,
    ):
        self.db_path = validate_safe_path(db_path, allowed_base_dir)
        self._local = threading.local()
        self._conns: List[sqlite3.Connection] = []
        self._lock = threading.RLock()
        self._nx_lock = threading.Lock()
        self._nx_graph: Optional[nx.MultiDiGraph] = None

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        if not self.db_path.exists():
            raise StorageError("Financial KG SQLite database does not exist")
        try:
            conn = sqlite3.connect(f"{self.db_path.as_uri()}?mode=ro", uri=True, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
        except sqlite3.Error as exc:
            raise StorageError("Failed to open Financial KG SQLite database") from exc
        self._local.conn = conn
        with self._lock:
            self._conns.append(conn)
        return conn

    def close(self) -> None:
        with self._lock:
            for c in self._conns:
                try:
                    c.close()
                except sqlite3.Error:
                    pass
            self._conns.clear()
        with self._nx_lock:
            self._nx_graph = None

    def list_entities(self) -> List[EntityNode]:
        rows = self._get_conn().execute(
            "SELECT entity_id, canonical_name, entity_type, aliases_json, doc_ids_json, degree FROM kg_entities ORDER BY degree DESC, canonical_name ASC"
        ).fetchall()
        return [
            EntityNode(
                entity_id=str(r["entity_id"]),
                canonical_name=str(r["canonical_name"]),
                entity_type=str(r["entity_type"] or "organization"),
                aliases=json.loads(r["aliases_json"] or "[]"),
                doc_ids=json.loads(r["doc_ids_json"] or "[]"),
                degree=int(r["degree"] or 0),
            )
            for r in rows
        ]

    def get_entity_edges(self, entity_name: str, *, limit: int = 50) -> List[RelationEdge]:
        rows = self._get_conn().execute(
            """
            SELECT edge_id, head_entity, relation_type, tail_entity, attributes_json,
                   evidence_doc_id, evidence_page, report_year, section_title, evidence_snippet, confidence
            FROM kg_edges
            WHERE head_entity = ? OR tail_entity = ?
            ORDER BY confidence DESC, evidence_page ASC
            LIMIT ?
            """,
            (entity_name, entity_name, int(limit)),
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    @staticmethod
    def _row_to_edge(r: sqlite3.Row) -> RelationEdge:
        return RelationEdge(
            edge_id=str(r["edge_id"]),
            head_entity=str(r["head_entity"]),
            relation_type=str(r["relation_type"]),
            tail_entity=str(r["tail_entity"]),
            attributes=json.loads(r["attributes_json"] or "{}"),
            evidence_doc_id=str(r["evidence_doc_id"]),
            evidence_page=int(r["evidence_page"] or 1),
            report_year=r["report_year"],
            section_title=str(r["section_title"] or ""),
            evidence_snippet=str(r["evidence_snippet"] or ""),
            confidence=float(r["confidence"] or 0.8),
        )

    def to_networkx(self) -> nx.MultiDiGraph:
        conn = self._get_conn()
        with self._nx_lock:
            if self._nx_graph is not None:
                return self._nx_graph
            G = nx.MultiDiGraph()
            for r in conn.execute("SELECT entity_id, canonical_name, entity_type, aliases_json, doc_ids_json, degree FROM kg_entities"):
                G.add_node(
                    str(r["canonical_name"]),
                    entity_id=str(r["entity_id"]),
                    entity_type=str(r["entity_type"] or "organization"),
                    aliases=json.loads(r["aliases_json"] or "[]"),
                    doc_ids=json.loads(r["doc_ids_json"] or "[]"),
                    degree=int(r["degree"] or 0),
                )
            for r in conn.execute(
                "SELECT edge_id, head_entity, relation_type, tail_entity, attributes_json, "
                "evidence_doc_id, evidence_page, report_year, section_title, evidence_snippet, confidence FROM kg_edges"
            ):
                G.add_edge(
                    str(r["head_entity"]),
                    str(r["tail_entity"]),
                    key=str(r["edge_id"]),
                    edge_id=str(r["edge_id"]),
                    relation_type=str(r["relation_type"]),
                    attributes=json.loads(r["attributes_json"] or "{}"),
                    evidence_doc_id=str(r["evidence_doc_id"]),
                    evidence_page=int(r["evidence_page"] or 1),
                    report_year=r["report_year"],
                    section_title=str(r["section_title"] or ""),
                    evidence_snippet=str(r["evidence_snippet"] or ""),
                    confidence=float(r["confidence"] or 0.8),
                )
            self._nx_graph = G
            return G

    def export_qdrant_entity_points(
        self,
        embed_fn: Optional[Callable[[str], Sequence[float]]] = None,
    ) -> List[Dict[str, Any]]:
        """Export KG entity points with vectors for optional Qdrant entity collection sync."""
        encoder = embed_fn or _hash_entity_vector
        points: List[Dict[str, Any]] = []
        for ent in self.list_entities():
            text = f"{ent.canonical_name} {' '.join(ent.aliases)}".strip()
            points.append(
                {
                    "id": ent.entity_id,
                    "vector": list(encoder(text)),
                    "payload": {
                        "canonical_name": ent.canonical_name,
                        "entity_type": ent.entity_type,
                        "aliases": ent.aliases,
                        "doc_ids": ent.doc_ids,
                        "degree": ent.degree,
                    },
                }
            )
        return points

    def summary_stats(self) -> Dict[str, Any]:
        conn = self._get_conn()
        ent_cnt = conn.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0]
        edge_cnt = conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
        doc_edges_cnt = conn.execute("SELECT COUNT(DISTINCT evidence_doc_id) FROM kg_edges").fetchone()[0]
        rel_counts = {
            str(r[0]): int(r[1])
            for r in conn.execute("SELECT relation_type, COUNT(*) FROM kg_edges GROUP BY relation_type ORDER BY COUNT(*) DESC").fetchall()
        }
        # Bridge entities appearing across >= 2 distinct reports (excluding anonymized scope nodes)
        bridge_cnt = 0
        for r in conn.execute("SELECT entity_type, doc_ids_json FROM kg_entities WHERE entity_type != 'anonymized_counterparty'"):
            docs = json.loads(r[1] or "[]")
            if len(docs) >= 2:
                bridge_cnt += 1
        return {
            "entities_count": int(ent_cnt),
            "edges_count": int(edge_cnt),
            "reports_with_edges": int(doc_edges_cnt),
            "relation_type_counts": rel_counts,
            "cross_report_bridge_entities": bridge_cnt,
        }

    @staticmethod
    def write_kg_sqlite(
        entities: Sequence[EntityNode],
        edges: Sequence[RelationEdge],
        out_db_path: Path | str,
        *,
        reports_count: Optional[int] = None,
        allowed_base_dir: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        out_path = validate_safe_path(out_db_path, allowed_base_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f".{out_path.name}.tmp")
        for suffix in ("", "-wal", "-shm"):
            leftover = Path(f"{tmp_path}{suffix}")
            if leftover.exists():
                leftover.unlink()

        # Merge entities by canonical_name & compute degree from deduplicated edges
        merged_entities: Dict[str, EntityNode] = {}
        for ent in entities:
            prev = merged_entities.get(ent.canonical_name)
            if prev is None:
                merged_entities[ent.canonical_name] = EntityNode(
                    entity_id=_entity_id_for(ent.canonical_name),
                    canonical_name=ent.canonical_name,
                    entity_type=ent.entity_type,
                    aliases=list(ent.aliases),
                    doc_ids=list(dict.fromkeys(ent.doc_ids)),
                )
            else:
                for al in ent.aliases:
                    if al not in prev.aliases:
                        prev.aliases.append(al)
                for did in ent.doc_ids:
                    if did not in prev.doc_ids:
                        prev.doc_ids.append(did)

        dedup_edges: Dict[str, RelationEdge] = {e.edge_id: e for e in edges}
        deg_map: Dict[str, int] = {}
        for e in dedup_edges.values():
            deg_map[e.head_entity] = deg_map.get(e.head_entity, 0) + 1
            deg_map[e.tail_entity] = deg_map.get(e.tail_entity, 0) + 1
            for cname in (e.head_entity, e.tail_entity):
                if cname not in merged_entities:
                    merged_entities[cname] = EntityNode(
                        entity_id=_entity_id_for(cname),
                        canonical_name=cname,
                        entity_type="anonymized_counterparty" if "::" in cname else "organization",
                        doc_ids=[e.evidence_doc_id] if e.evidence_doc_id else [],
                    )
                elif e.evidence_doc_id and e.evidence_doc_id not in merged_entities[cname].doc_ids:
                    merged_entities[cname].doc_ids.append(e.evidence_doc_id)

        conn = sqlite3.connect(str(tmp_path))
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS kg_entities (
                    entity_id      TEXT PRIMARY KEY,
                    canonical_name TEXT NOT NULL UNIQUE,
                    entity_type    TEXT NOT NULL,
                    aliases_json   TEXT NOT NULL,
                    doc_ids_json   TEXT NOT NULL,
                    degree         INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kg_edges (
                    edge_id          TEXT PRIMARY KEY,
                    head_entity      TEXT NOT NULL,
                    relation_type    TEXT NOT NULL,
                    tail_entity      TEXT NOT NULL,
                    attributes_json  TEXT NOT NULL,
                    evidence_doc_id  TEXT NOT NULL,
                    evidence_page    INTEGER NOT NULL,
                    report_year      INTEGER,
                    section_title    TEXT,
                    evidence_snippet TEXT,
                    confidence       REAL
                );
                CREATE INDEX IF NOT EXISTS idx_kg_head ON kg_edges(head_entity);
                CREATE INDEX IF NOT EXISTS idx_kg_tail ON kg_edges(tail_entity);
                CREATE INDEX IF NOT EXISTS idx_kg_rel  ON kg_edges(relation_type);
                CREATE INDEX IF NOT EXISTS idx_kg_doc  ON kg_edges(evidence_doc_id);
                """
            )
            conn.executemany(
                "INSERT OR REPLACE INTO kg_entities VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        ent.entity_id,
                        ent.canonical_name,
                        ent.entity_type,
                        json.dumps(ent.aliases, ensure_ascii=False),
                        json.dumps(ent.doc_ids, ensure_ascii=False),
                        deg_map.get(ent.canonical_name, 0),
                    )
                    for ent in merged_entities.values()
                ],
            )
            conn.executemany(
                "INSERT OR REPLACE INTO kg_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        e.edge_id,
                        e.head_entity,
                        e.relation_type,
                        e.tail_entity,
                        json.dumps(e.attributes, ensure_ascii=False),
                        e.evidence_doc_id,
                        int(e.evidence_page),
                        e.report_year,
                        e.section_title,
                        e.evidence_snippet,
                        float(e.confidence),
                    )
                    for e in dedup_edges.values()
                ],
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

        store = FinancialKGStore(out_path, allowed_base_dir=allowed_base_dir)
        try:
            stats = store.summary_stats()
            if reports_count is not None:
                stats["reports_count"] = int(reports_count)
            stats["db_path"] = str(out_path)
            return stats
        finally:
            store.close()

    @staticmethod
    def build_from_sqlite_doc_repo(
        doc_sqlite_path: Path | str,
        out_kg_path: Path | str,
        *,
        toc_sqlite_path: Optional[Path | str] = None,
        limit: Optional[int] = None,
        allowed_base_dir: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        from src.multimodal_nav.toc_tree_builder import TocTreeStore
        from src.storage.sqlite_doc_store import SqliteDocumentRepository

        repo = SqliteDocumentRepository(doc_sqlite_path, allowed_base_dir=allowed_base_dir)
        toc_store = None
        if toc_sqlite_path and Path(toc_sqlite_path).exists():
            toc_store = TocTreeStore(toc_sqlite_path, allowed_base_dir=allowed_base_dir)
        builder = FinancialKGBuilder()
        all_entities: List[EntityNode] = []
        all_edges: List[RelationEdge] = []
        try:
            metas = repo.list_metainfo()
            if limit is not None:
                metas = metas[: max(0, int(limit))]
            loaded_reports: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
            for meta in metas:
                did = str(meta.get("sha1_name") or meta.get("doc_id") or "")
                if not did:
                    continue
                loaded_reports.append((meta, repo.get_pages(did)))

            alias_map = builder.build_alias_map_from_reports(loaded_reports)
            for meta, pages in loaded_reports:
                did = str(meta.get("sha1_name") or meta.get("doc_id") or "")
                tree = toc_store.get_tree(did) if toc_store is not None else None
                raw_year = meta.get("report_year") or meta.get("fiscal_year")
                try:
                    ry = int(raw_year) if raw_year is not None else None
                except (TypeError, ValueError):
                    ry = None
                ents, edges = builder.extract_from_report(
                    doc_id=did,
                    company_name=str(meta.get("company_name") or ""),
                    report_year=ry,
                    pages=pages,
                    toc_tree=tree,
                    alias_to_canonical=alias_map,
                )
                all_entities.extend(ents)
                all_edges.extend(edges)
            reports_n = len(loaded_reports)
        finally:
            if toc_store is not None:
                toc_store.close()
            repo.close()

        return FinancialKGStore.write_kg_sqlite(
            all_entities,
            all_edges,
            out_kg_path,
            reports_count=reports_n,
            allowed_base_dir=allowed_base_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build targeted financial KG SQLite database from annual reports")
    parser.add_argument("--doc-db", type=Path, default=_DEFAULT_DOC_SQLITE)
    parser.add_argument("--toc-db", type=Path, default=_DEFAULT_TOC_SQLITE)
    parser.add_argument("--out-db", type=Path, default=_DEFAULT_KG_SQLITE)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    stats = FinancialKGStore.build_from_sqlite_doc_repo(
        args.doc_db,
        args.out_db,
        toc_sqlite_path=args.toc_db,
        limit=args.limit,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
