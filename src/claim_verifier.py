"""Claim-level grounding verifier for long-form financial reports (ALCE / CiteBench style).

Two-stage low-latency verification (< 5ms Stage-1 deterministic numeric/year/SQL cross-check,
optional Stage-2 semantic/NLI scorer + polarity inversion detection).
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.text_normalization import tokenize_for_bm25

_CNY_UNIT_FACTORS: Dict[str, float] = {
    "元": 1.0,
    "千元": 1e3,
    "万元": 1e4,
    "百万元": 1e6,
    "亿元": 1e8,
    "万亿元": 1e12,
}

_UNIT_HEADER_RE = re.compile(r"单位[：:\s]*(?:人民币)?\s*(万亿元|亿元|百万元|万元|千元|元)")
_REF_INDEX_RE = re.compile(
    r"(?:第\s*\d+\s*[页章节条款项份位笔名号]|(?<![A-Za-z])p\.?\s*\d+|(?<!\d)(?:00|30|60|68|83|87|43)\d{4}(?!\d))",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(
    r"(?:人民币|RMB|CNY)?\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*(万亿元|亿元|百万元|万元|千元|元)"
)
_PERCENT_RE = re.compile(r"([+-]?\d[\d,]*(?:\.\d+)?)\s*(%|％|百分点)")
_YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})\s*(?:年(?:度)?(?![月日])|\b)")
_SCALAR_RE = re.compile(r"(?<![A-Za-z0-9.])([+-]?\d[\d,]*(?:\.\d+)?)(?![A-Za-z0-9.%％万亿千百])")
_SENTENCE_SPLIT_RE = re.compile(r"(?:[。！？；\n]+|(?<!\d)\.(?!\d)(?=\s|$))")
_BULLET_PREFIX_RE = re.compile(r"^(?:[-*•]+|\d+[.)、])\s*")
_PURE_REF_RE = re.compile(r"^(?:第\s*\d+\s*页|p\.?\s*\d+|详见|注\s*\d*|\[.*?\])$", re.IGNORECASE)

_INCREASE_WORDS = ("同比增长", "上升", "增长", "提升", "提高", "扩大", "增加", "改善", "回升")
_DECREASE_WORDS = ("同比下降", "同比减少", "下降", "下滑", "减少", "降低", "萎缩", "恶化", "缩窄")
_NEG_ABSENT_WORDS = ("未发生", "不存在", "未受到", "无重大", "零违规")
_POS_PRESENT_WORDS = ("已发生", "受到行政处罚", "存在违规", "被立案调查", "重大诉讼")

_CLAIM_STOP_TOKENS = frozenset(
    {
        "公司",
        "本行",
        "本集团",
        "年度",
        "报告",
        "报告期",
        "期间",
        "实现",
        "达到",
        "约为",
        "分别为",
        "以及",
        "其中",
        "方面",
    }
)


@dataclass
class ClaimAuditItem:
    claim: str
    status: str  # "supported" | "unverified" | "contradicted"
    evidence_ref: Optional[Dict[str, Any]] = None
    numeric_checks: List[Dict[str, Any]] = field(default_factory=list)
    support_score: float = 0.0
    contradiction_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "status": self.status,
            "evidence_ref": dict(self.evidence_ref) if self.evidence_ref else None,
            "numeric_checks": list(self.numeric_checks),
            "support_score": round(float(self.support_score), 4),
            "contradiction_reason": self.contradiction_reason,
        }


@dataclass
class ClaimAuditReport:
    claims: List[ClaimAuditItem]
    total_claims: int
    supported_count: int
    unverified_count: int
    contradicted_count: int
    unsupported_claim_rate: float
    contradiction_rate: float
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "supported_count": self.supported_count,
            "unverified_count": self.unverified_count,
            "contradicted_count": self.contradicted_count,
            "unsupported_claim_rate": round(float(self.unsupported_claim_rate), 4),
            "contradiction_rate": round(float(self.contradiction_rate), 4),
            "latency_ms": round(float(self.latency_ms), 3),
            "claims": [c.to_dict() for c in self.claims],
        }


def extract_atomic_claims(answer_text: str) -> List[str]:
    """Split long financial answer prose into verifiable factual atomic claims."""
    norm = unicodedata.normalize("NFKC", str(answer_text or "")).strip()
    if not norm or norm == "N/A":
        return []

    claims: List[str] = []
    seen: set[str] = set()
    for raw_part in _SENTENCE_SPLIT_RE.split(norm):
        s = raw_part.strip()
        if not s or s.startswith("#") or s.startswith("|---"):
            continue
        s = _BULLET_PREFIX_RE.sub("", s).strip()
        # Strip inline markdown pipe separators for table-row bullets
        if s.startswith("|"):
            s = " ".join(c.strip() for c in s.strip("|").split("|") if c.strip())
        if not s or _PURE_REF_RE.match(s):
            continue
        has_num = bool(re.search(r"\d", s))
        if not has_num and len(s) < 10:
            continue
        if len(s) < 6:
            continue
        if s not in seen:
            seen.add(s)
            claims.append(s)
    return claims


def extract_numeric_mentions(
    text: str,
    *,
    default_cny_unit: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract normalized numeric, monetary (CNY), percent, and year mentions from text."""
    norm = unicodedata.normalize("NFKC", str(text or ""))
    mentions: List[Tuple[int, Dict[str, Any]]] = []
    occupied: List[Tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in occupied)

    def _to_float(num_str: str) -> Optional[float]:
        try:
            return float(num_str.replace(",", ""))
        except ValueError:
            return None

    for m in _REF_INDEX_RE.finditer(norm):
        occupied.append(m.span())

    for m in _MONEY_RE.finditer(norm):
        if _overlaps(*m.span()):
            continue
        val = _to_float(m.group(1))
        if val is None:
            continue
        unit = m.group(2)
        occupied.append(m.span())
        mentions.append(
            (
                m.start(),
                {
                    "raw": m.group(0).strip(),
                    "kind": "money_cny",
                    "value": val,
                    "unit": unit,
                    "base_value": val * _CNY_UNIT_FACTORS.get(unit, 1.0),
                },
            )
        )

    for m in _PERCENT_RE.finditer(norm):
        if _overlaps(*m.span()):
            continue
        val = _to_float(m.group(1))
        if val is None:
            continue
        occupied.append(m.span())
        mentions.append(
            (
                m.start(),
                {
                    "raw": m.group(0).strip(),
                    "kind": "percent",
                    "value": val,
                    "unit": "%",
                    "base_value": val,
                },
            )
        )

    for m in _YEAR_RE.finditer(norm):
        if _overlaps(*m.span()):
            continue
        val = _to_float(m.group(1))
        if val is None:
            continue
        occupied.append(m.span())
        mentions.append(
            (
                m.start(),
                {
                    "raw": m.group(0).strip(),
                    "kind": "year",
                    "value": val,
                    "unit": "年",
                    "base_value": val,
                },
            )
        )

    unit_factor = _CNY_UNIT_FACTORS.get(default_cny_unit or "", 0.0)
    for m in _SCALAR_RE.finditer(norm):
        if _overlaps(*m.span()):
            continue
        val = _to_float(m.group(1))
        if val is None:
            continue
        occupied.append(m.span())
        item: Dict[str, Any] = {
            "raw": m.group(0).strip(),
            "kind": "scalar",
            "value": val,
            "unit": default_cny_unit or "",
            "base_value": val,
        }
        if unit_factor > 0.0:
            item["scaled_cny"] = val * unit_factor
        mentions.append((m.start(), item))

    mentions.sort(key=lambda pair: pair[0])
    return [item for _, item in mentions]


def _numbers_close(a: float, b: float, rel_tol: float) -> bool:
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= rel_tol


def _passage_tokens(text: str) -> set[str]:
    return {t for t in tokenize_for_bm25(unicodedata.normalize("NFKC", text or "")) if len(t) >= 2 and t not in _CLAIM_STOP_TOKENS}


def _polarity_conflict(claim: str, evidence_text: str) -> str:
    c = unicodedata.normalize("NFKC", claim)
    e = unicodedata.normalize("NFKC", evidence_text)
    if any(w in c for w in _INCREASE_WORDS) and any(w in e for w in _DECREASE_WORDS):
        if not any(w in e for w in _INCREASE_WORDS):
            return "趋势极性冲突（断言称增长/上升，证据显示下降/减少）"
    if any(w in c for w in _DECREASE_WORDS) and any(w in e for w in _INCREASE_WORDS):
        if not any(w in e for w in _DECREASE_WORDS):
            return "趋势极性冲突（断言称下降/减少，证据显示增长/上升）"
    if any(w in c for w in _NEG_ABSENT_WORDS) and any(w in e for w in _POS_PRESENT_WORDS):
        return "事实存在性冲突（断言称未发生/不存在，证据显示已发生/存在）"
    if any(w in c for w in _POS_PRESENT_WORDS) and any(w in e for w in _NEG_ABSENT_WORDS):
        return "事实存在性冲突（断言称存在/发生，证据显示未发生/不存在）"
    return ""


class ClaimVerifier:
    """Two-stage sentence/claim grounding verifier backed by citations, chunks, and optional SQL facts."""

    def __init__(
        self,
        *,
        sql_store: Optional[Any] = None,
        nli_scorer_fn: Optional[Callable[[str, str], float]] = None,
        rel_tol: float = 0.015,
    ):
        self.sql_store = sql_store
        self.nli_scorer_fn = nli_scorer_fn
        self.rel_tol = float(rel_tol)

    @staticmethod
    def _build_evidence_items(
        citations: Sequence[Dict[str, Any]],
        retrieval_results: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        seen: set[Tuple[Any, str]] = set()

        def _add(text: str, page: Any, doc_id: str, src: str) -> None:
            cleaned = unicodedata.normalize("NFKC", str(text or "")).strip()
            if not cleaned:
                return
            key = (page, cleaned[:120])
            if key in seen:
                return
            seen.add(key)
            um = _UNIT_HEADER_RE.search(cleaned[:600])
            cny_unit = um.group(1) if um else None
            mentions = extract_numeric_mentions(cleaned, default_cny_unit=cny_unit)
            items.append(
                {
                    "text": cleaned,
                    "page": page,
                    "doc_id": doc_id,
                    "source": src,
                    "tokens": _passage_tokens(cleaned[:3000]),
                    "default_cny_unit": cny_unit,
                    "mentions": mentions,
                    "sentences": [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if len(s.strip()) >= 6],
                }
            )

        for r in retrieval_results or []:
            meta = r.get("metadata") or {}
            did = str(r.get("pdf_sha1") or meta.get("sha1_name") or meta.get("doc_id") or "")
            _add(str(r.get("text") or ""), r.get("page"), did, "retrieval_result")

        for c in citations or []:
            pages = c.get("pages") or ([c["page"]] if c.get("page") is not None else [])
            p0 = pages[0] if pages else None
            did = str(c.get("pdf_sha1") or c.get("doc_id") or "")
            txt = c.get("text") or c.get("snippet") or c.get("excerpt") or c.get("quote") or ""
            _add(str(txt), p0, did, "citation")

        return items

    def _load_sql_facts(
        self,
        company_name: Optional[str],
        report_year: Optional[int],
    ) -> List[Dict[str, Any]]:
        if self.sql_store is None or not company_name:
            return []
        try:
            period = str(report_year) if report_year is not None else None
            rows = self.sql_store.query_company_metric(
                company_name=company_name,
                metric_atom_id=None,
                period=period,
                prefer_consolidated=True,
                exclude_pct=False,
            )
            return list(rows or [])
        except Exception:
            return []

    def _check_numeric_against_evidence(
        self,
        mention: Dict[str, Any],
        claim_tokens: set[str],
        best_passage: Optional[Dict[str, Any]],
        evidence_items: Sequence[Dict[str, Any]],
        sql_facts: Sequence[Dict[str, Any]],
        claim_text: str,
    ) -> Tuple[str, Optional[Dict[str, Any]], str]:
        kind = mention["kind"]
        val = float(mention["base_value"])

        if kind == "year":
            year_int = int(round(val))
            for ev in evidence_items:
                if any(em["kind"] == "year" and int(round(em["base_value"])) == year_int for em in ev["mentions"]):
                    return "supported", {"page": ev["page"], "doc_id": ev["doc_id"], "source": ev["source"]}, ""
            for fact in sql_facts:
                if str(year_int) in str(fact.get("period") or ""):
                    return "supported", {"page": fact.get("source_page"), "doc_id": fact.get("doc_id"), "source": "sql_facts"}, ""
            return "unverified", None, ""

        # Check matching values across SQL facts first
        claim_compact = re.sub(r"\s+", "", claim_text)
        claim_mentions_delta = any(w in claim_compact for w in ("增减", "变动", "同比"))
        claim_years = {m.group(1) for m in _YEAR_RE.finditer(claim_text)}
        matched_sql_rows: List[Dict[str, Any]] = []
        for fact in sql_facts:
            col_h = str(fact.get("col_header") or "")
            if not claim_mentions_delta and any(k in col_h for k in ("增减", "变动", "比例", "占比")):
                continue
            mraw = re.sub(r"\s+", "", str(fact.get("metric_raw") or ""))
            if len(mraw) >= 2 and (mraw in claim_compact or (len(mraw) >= 4 and mraw[:4] in claim_compact)):
                matched_sql_rows.append(fact)
        if claim_years and matched_sql_rows:
            year_scoped = [
                f
                for f in matched_sql_rows
                if any(y in str(f.get("period") or "") or y in str(f.get("col_header") or "") for y in claim_years)
            ]
            if year_scoped:
                matched_sql_rows = year_scoped

        for fact in matched_sql_rows:
            norm_cny = fact.get("normalized_cny")
            raw_v = fact.get("raw_value")
            if kind == "money_cny" and norm_cny is not None and _numbers_close(val, float(norm_cny), self.rel_tol):
                return "supported", {"page": fact.get("source_page"), "doc_id": fact.get("doc_id"), "source": "sql_facts"}, ""
            if kind == "scalar" and raw_v is not None and _numbers_close(val, float(raw_v), self.rel_tol):
                return "supported", {"page": fact.get("source_page"), "doc_id": fact.get("doc_id"), "source": "sql_facts"}, ""
            if kind == "percent" and fact.get("raw_unit") == "%" and raw_v is not None and _numbers_close(val, float(raw_v), self.rel_tol):
                return "supported", {"page": fact.get("source_page"), "doc_id": fact.get("doc_id"), "source": "sql_facts"}, ""

        # Check matching values across cited/retrieved passages
        for ev in evidence_items:
            for em in ev["mentions"]:
                if em["kind"] == "year":
                    continue
                candidates = [float(em["base_value"])]
                if "scaled_cny" in em:
                    candidates.append(float(em["scaled_cny"]))
                if kind == "scalar" and mention.get("unit") == "" and em["kind"] == "money_cny":
                    candidates.append(float(em["value"]))
                if kind == "money_cny" and em["kind"] == "scalar" and ev.get("default_cny_unit"):
                    # Already covered by scaled_cny; also guard percent vs money confusion
                    candidates = [float(em["scaled_cny"])] if "scaled_cny" in em else []
                if kind == "percent" and em["kind"] != "percent":
                    continue
                if kind == "money_cny" and em["kind"] == "percent":
                    continue
                if any(_numbers_close(val, cval, self.rel_tol) for cval in candidates):
                    return "supported", {"page": ev["page"], "doc_id": ev["doc_id"], "source": ev["source"]}, ""

        # Stage-1B: Contradiction check against matched SQL metric or high-overlap evidence passage
        for fact in matched_sql_rows:
            if kind == "money_cny" and fact.get("normalized_cny") is not None:
                return (
                    "contradicted",
                    {"page": fact.get("source_page"), "doc_id": fact.get("doc_id"), "source": "sql_facts"},
                    f"SQL事实库[{fact.get('metric_raw')}={fact.get('raw_value')}{fact.get('raw_unit')}]与断言[{mention['raw']}]冲突",
                )
            if kind == "percent" and fact.get("raw_unit") == "%" and fact.get("raw_value") is not None:
                return (
                    "contradicted",
                    {"page": fact.get("source_page"), "doc_id": fact.get("doc_id"), "source": "sql_facts"},
                    f"SQL事实库[{fact.get('metric_raw')}={fact.get('raw_value')}%]与断言[{mention['raw']}]冲突",
                )

        if best_passage is not None and claim_tokens:
            ev_unit = best_passage.get("default_cny_unit")
            best_cand: Optional[Tuple[float, Dict[str, Any]]] = None
            for sent in best_passage.get("sentences") or []:
                s_toks = _passage_tokens(sent)
                shared = claim_tokens & s_toks
                ov = len(shared) / max(1, len(claim_tokens))
                if len(shared) < 2 or ov < 0.45:
                    continue
                s_mentions = extract_numeric_mentions(sent, default_cny_unit=ev_unit)
                same_kind = [
                    em
                    for em in s_mentions
                    if (
                        em["kind"] == kind
                        or (kind == "money_cny" and em["kind"] == "scalar" and "scaled_cny" in em)
                    )
                    and em["kind"] != "year"
                ]
                if same_kind and (best_cand is None or ov > best_cand[0]):
                    best_cand = (ov, same_kind[0])
            if best_cand is not None:
                first_ev = best_cand[1]
                return (
                    "contradicted",
                    {"page": best_passage["page"], "doc_id": best_passage["doc_id"], "source": best_passage["source"]},
                    f"高重叠证据数字[{first_ev['raw']}]与断言[{mention['raw']}]冲突",
                )

        return "unverified", None, ""

    def verify(
        self,
        answer_text: str,
        *,
        citations: Sequence[Dict[str, Any]] = (),
        retrieval_results: Sequence[Dict[str, Any]] = (),
        company_name: Optional[str] = None,
        report_year: Optional[int] = None,
    ) -> ClaimAuditReport:
        t0 = time.perf_counter()
        claims = extract_atomic_claims(answer_text)
        if not claims:
            return ClaimAuditReport(
                claims=[],
                total_claims=0,
                supported_count=0,
                unverified_count=0,
                contradicted_count=0,
                unsupported_claim_rate=0.0,
                contradiction_rate=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        evidence_items = self._build_evidence_items(citations, retrieval_results)
        sql_facts = self._load_sql_facts(company_name, report_year)

        audited: List[ClaimAuditItem] = []
        for claim in claims:
            c_tokens = _passage_tokens(claim)
            best_ev: Optional[Dict[str, Any]] = None
            best_overlap = 0.0
            for ev in evidence_items:
                ov = len(c_tokens & ev["tokens"]) / max(1, len(c_tokens)) if c_tokens else 0.0
                if ov > best_overlap:
                    best_overlap = ov
                    best_ev = ev

            # Check polarity contradiction against highest-overlap sentence in best passage
            polarity_reason = ""
            if best_ev is not None and best_overlap >= 0.35:
                best_sent = best_ev["text"]
                if best_ev["sentences"]:
                    scored_sents = sorted(
                        best_ev["sentences"],
                        key=lambda s: len(c_tokens & _passage_tokens(s)),
                        reverse=True,
                    )
                    if scored_sents and len(c_tokens & _passage_tokens(scored_sents[0])) >= 2:
                        best_sent = scored_sents[0]
                        polarity_reason = _polarity_conflict(claim, best_sent)

            mentions = extract_numeric_mentions(claim)
            numeric_checks: List[Dict[str, Any]] = []
            num_contradiction = ""
            num_ref: Optional[Dict[str, Any]] = None
            has_non_year_num = False
            all_non_year_supported = True

            for m in mentions:
                status_m, ref_m, reason_m = self._check_numeric_against_evidence(
                    m,
                    c_tokens,
                    best_ev,
                    evidence_items,
                    sql_facts,
                    claim,
                )
                numeric_checks.append({"raw": m["raw"], "kind": m["kind"], "status": status_m, "reason": reason_m})
                if m["kind"] != "year":
                    has_non_year_num = True
                    if status_m == "supported" and num_ref is None:
                        num_ref = ref_m
                    if status_m == "contradicted" and not num_contradiction:
                        num_contradiction = reason_m
                        num_ref = ref_m
                    if status_m != "supported":
                        all_non_year_supported = False
                elif status_m == "supported" and num_ref is None:
                    num_ref = ref_m

            nli_score = 0.0
            if self.nli_scorer_fn is not None and best_ev is not None:
                try:
                    nli_score = float(self.nli_scorer_fn(claim, best_ev["text"][:1500]))
                except Exception:
                    nli_score = 0.0
            support_score = max(best_overlap, nli_score)

            if num_contradiction or polarity_reason:
                final_status = "contradicted"
                final_reason = "; ".join(x for x in (num_contradiction, polarity_reason) if x)
                final_ref = num_ref or (
                    {"page": best_ev["page"], "doc_id": best_ev["doc_id"], "source": best_ev["source"]}
                    if best_ev is not None
                    else None
                )
            elif has_non_year_num:
                if all_non_year_supported and (best_overlap >= 0.15 or (num_ref or {}).get("source") == "sql_facts" or nli_score >= 0.5):
                    final_status = "supported"
                    final_reason = ""
                    final_ref = num_ref or (
                        {"page": best_ev["page"], "doc_id": best_ev["doc_id"], "source": best_ev["source"]}
                        if best_ev is not None
                        else None
                    )
                else:
                    final_status = "unverified"
                    final_reason = ""
                    final_ref = num_ref
            else:
                if best_ev is not None and (best_overlap >= 0.35 or nli_score >= 0.5):
                    final_status = "supported"
                    final_ref = {"page": best_ev["page"], "doc_id": best_ev["doc_id"], "source": best_ev["source"]}
                else:
                    final_status = "unverified"
                    final_ref = None
                final_reason = ""

            audited.append(
                ClaimAuditItem(
                    claim=claim,
                    status=final_status,
                    evidence_ref=final_ref,
                    numeric_checks=numeric_checks,
                    support_score=support_score,
                    contradiction_reason=final_reason,
                )
            )

        total = len(audited)
        sup_cnt = sum(1 for c in audited if c.status == "supported")
        unv_cnt = sum(1 for c in audited if c.status == "unverified")
        con_cnt = sum(1 for c in audited if c.status == "contradicted")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ClaimAuditReport(
            claims=audited,
            total_claims=total,
            supported_count=sup_cnt,
            unverified_count=unv_cnt,
            contradicted_count=con_cnt,
            unsupported_claim_rate=(unv_cnt + con_cnt) / total if total else 0.0,
            contradiction_rate=con_cnt / total if total else 0.0,
            latency_ms=latency_ms,
        )
