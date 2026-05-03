from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.retrieval_filters import RetrievalFilters
from src.table_grounding import _convert_to_target_unit, _score_overlap, _target_unit_from_question
from src.text_normalization import normalize_text, parse_numeric_value, tokenize_for_bm25


def _page_number(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _question_mentions_chart(question: str) -> bool:
    return any(term in (question or "") for term in ("图表", "柱状图", "折线图", "趋势图", "走势图", "曲线图", "趋势"))


def _unit_mismatch_penalty(question: str, unit: Optional[str]) -> float:
    question_text = question or ""
    unit_text = str(unit or "")
    if ("%" in question_text or "占比" in question_text or "比例" in question_text) and "%" not in unit_text:
        return 2.0
    target_unit = _target_unit_from_question(question_text)
    if target_unit and unit_text and target_unit != unit_text:
        if target_unit == "元" and unit_text.endswith("元"):
            return 0.0
        return 0.4
    return 0.0


class ChartGrounder:
    def __init__(self, documents_dir: Path, confidence_threshold: float = 0.0):
        self.documents_dir = Path(documents_dir)
        self.confidence_threshold = float(confidence_threshold)
        self._documents_cache: Dict[str, Dict] = {}

    def _load_document(self, doc_id: str) -> Optional[Dict]:
        if doc_id in self._documents_cache:
            return self._documents_cache[doc_id]
        candidate_paths = [self.documents_dir / f"{doc_id}.json", self.documents_dir / doc_id]
        for path in candidate_paths:
            if not path.exists() or not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            self._documents_cache[doc_id] = payload
            return payload
        for path in self.documents_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            metainfo = payload.get("metainfo") or {}
            observed_doc_id = str(metainfo.get("sha1_name") or metainfo.get("doc_id") or path.stem)
            if observed_doc_id == str(doc_id):
                self._documents_cache[doc_id] = payload
                return payload
        return None

    def _candidate_doc_ids(
        self,
        retrieval_results: List[Dict],
        candidate_doc_ids: Optional[List[str]],
        allowed_doc_ids: Optional[List[str]],
    ) -> List[str]:
        doc_ids = [str(doc_id) for doc_id in (allowed_doc_ids or candidate_doc_ids or []) if doc_id]
        if doc_ids:
            return list(dict.fromkeys(doc_ids))
        return list(
            dict.fromkeys(
                str((result.get("metadata") or {}).get("sha1_name"))
                for result in retrieval_results
                if (result.get("metadata") or {}).get("sha1_name")
            )
        )

    @staticmethod
    def _record_context(record: Dict[str, Any]) -> str:
        return "\n".join(
            str(value)
            for value in (
                record.get("series_name"),
                record.get("x_label"),
                record.get("unit"),
                record.get("context_text"),
                record.get("table_markdown"),
            )
            if value
        )

    def _record_score(
        self,
        *,
        question: str,
        query_tokens: List[str],
        record: Dict[str, Any],
        candidate_pages: set,
        filters: RetrievalFilters,
    ) -> float:
        series_name = str(record.get("series_name") or "")
        x_label = str(record.get("x_label") or "")
        context = self._record_context(record)
        score = 0.0
        score += _score_overlap(query_tokens, series_name) * 2.2
        score += _score_overlap(query_tokens, x_label) * 1.8
        score += _score_overlap(query_tokens, context) * 0.7
        normalized_question = normalize_text(question)
        normalized_series = normalize_text(series_name)
        if ("营业收入" in normalized_question or "营收" in normalized_question) and any(
            term in normalized_series for term in ("营业收入", "营收")
        ):
            score += 3.0
        if filters.year is not None and str(filters.year) in normalize_text(f"{x_label} {context}"):
            score += 1.5
        if filters.period and normalize_text(filters.period) in normalize_text(f"{x_label} {context}"):
            score += 1.0
        if record.get("page") in candidate_pages:
            score += 1.0
        if _question_mentions_chart(question):
            score += 0.8
        score += min(1.0, max(0.0, float(record.get("confidence") or 0.0))) * 2.0
        score -= _unit_mismatch_penalty(question, record.get("unit"))
        return score

    def _build_match(
        self,
        *,
        doc_id: str,
        metainfo: Dict[str, Any],
        record: Dict[str, Any],
        question: str,
        score: float,
    ) -> Dict[str, Any]:
        unit = record.get("unit")
        raw_value = record.get("raw_value")
        normalized_value = record.get("normalized_value")
        if normalized_value is None:
            normalized_value = parse_numeric_value(str(raw_value or ""), unit_hint=unit)
        target_unit = _target_unit_from_question(question)
        answer_value = _convert_to_target_unit(normalized_value, target_unit)
        match = {
            "source_doc_id": doc_id,
            "company_name": metainfo.get("company_name"),
            "security_code": metainfo.get("security_code") or metainfo.get("stock_code"),
            "currency": metainfo.get("currency"),
            "report_year": metainfo.get("report_year") or metainfo.get("fiscal_year"),
            "report_type": metainfo.get("report_type"),
            "doc_source_type": metainfo.get("doc_source_type"),
            "page": record.get("page"),
            "chart_id": record.get("chart_id"),
            "picture_id": record.get("picture_id"),
            "bbox": record.get("bbox"),
            "series_name": record.get("series_name"),
            "x_label": record.get("x_label"),
            "period": record.get("x_label"),
            "raw_value": raw_value,
            "normalized_value": normalized_value,
            "unit": unit,
            "target_unit": target_unit,
            "answer_value": answer_value,
            "chart_context": record.get("context_text") or "",
            "chart_confidence": record.get("confidence"),
            "confidence": record.get("confidence"),
            "match_score": round(score, 4),
        }
        if target_unit:
            match["unit_conversion"] = {
                "from": unit,
                "to": target_unit,
                "base_value": normalized_value,
                "converted_value": answer_value,
            }
        return match

    def ground_number_query(
        self,
        question: str,
        retrieval_results: List[Dict],
        filters: RetrievalFilters,
        candidate_doc_ids: Optional[List[str]] = None,
        allowed_doc_ids: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        query_tokens = tokenize_for_bm25(question)
        if not query_tokens:
            return None
        candidate_pages = {result.get("page") for result in retrieval_results if result.get("page") is not None}
        doc_ids = self._candidate_doc_ids(retrieval_results, candidate_doc_ids, allowed_doc_ids)
        best_match = None
        best_score = 0.0

        for doc_id in doc_ids:
            document = self._load_document(doc_id)
            if not document:
                continue
            metainfo = document.get("metainfo") or {}
            if filters.company_name and metainfo.get("company_name"):
                if normalize_text(str(filters.company_name)) != normalize_text(str(metainfo.get("company_name"))):
                    continue
            if filters.security_code:
                observed_code = metainfo.get("security_code") or metainfo.get("stock_code")
                if observed_code and str(observed_code) != str(filters.security_code):
                    continue

            for record in (document.get("content") or {}).get("chart_records") or []:
                confidence = float(record.get("confidence") or 0.0)
                if confidence < self.confidence_threshold:
                    continue
                raw_value = record.get("raw_value")
                if parse_numeric_value(str(raw_value or ""), unit_hint=record.get("unit")) is None and record.get("normalized_value") is None:
                    continue
                score = self._record_score(
                    question=question,
                    query_tokens=query_tokens,
                    record=record,
                    candidate_pages=candidate_pages,
                    filters=filters,
                )
                if score > best_score:
                    best_score = score
                    best_match = self._build_match(
                        doc_id=doc_id,
                        metainfo=metainfo,
                        record=record,
                        question=question,
                        score=score,
                    )

        if not best_match or best_score < 1.5:
            return None
        return best_match
