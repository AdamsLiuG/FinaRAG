from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.retrieval_filters import RetrievalFilters
from src.text_normalization import normalize_text, parse_numeric_value, tokenize_for_bm25


def _score_overlap(query_tokens: List[str], text: str) -> float:
    if not query_tokens or not text:
        return 0.0
    normalized = normalize_text(text)
    return float(sum(1 for token in query_tokens if token and token in normalized))


class TableGrounder:
    def __init__(self, documents_dir: Path):
        self.documents_dir = Path(documents_dir)
        self._documents_cache: Dict[str, Dict] = {}

    def _load_document(self, doc_id: str) -> Optional[Dict]:
        if doc_id in self._documents_cache:
            return self._documents_cache[doc_id]

        for path in self.documents_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            metainfo = payload.get("metainfo") or {}
            if str(metainfo.get("sha1_name") or metainfo.get("doc_id") or path.stem) == str(doc_id):
                self._documents_cache[doc_id] = payload
                return payload
        return None

    def ground_number_query(
        self,
        question: str,
        retrieval_results: List[Dict],
        filters: RetrievalFilters,
        candidate_doc_ids: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        query_tokens = tokenize_for_bm25(question)
        if not query_tokens:
            return None

        candidate_pages = {result.get("page") for result in retrieval_results if result.get("page") is not None}
        doc_ids = list(candidate_doc_ids or [])
        if not doc_ids:
            doc_ids = list(
                dict.fromkeys(
                    str((result.get("metadata") or {}).get("sha1_name"))
                    for result in retrieval_results
                    if (result.get("metadata") or {}).get("sha1_name")
                )
            )

        best_match = None
        best_score = 0.0
        for doc_id in doc_ids:
            document = self._load_document(doc_id)
            if not document:
                continue
            structured_tables = document.get("content", {}).get("structured_tables") or []
            for table in structured_tables:
                table_context = table.get("markdown", "")
                for cell in table.get("cell_records") or []:
                    score = 0.0
                    row_text = " ".join(cell.get("matched_row_headers") or [])
                    col_text = " ".join(cell.get("matched_col_headers") or [])
                    score += _score_overlap(query_tokens, row_text) * 2.2
                    score += _score_overlap(query_tokens, col_text) * 1.8
                    score += _score_overlap(query_tokens, table_context) * 0.8

                    if cell.get("page") in candidate_pages:
                        score += 1.5
                    if filters.year is not None:
                        cell_period = normalize_text(str(cell.get("period") or table_context))
                        if str(filters.year) in cell_period:
                            score += 1.5
                    if filters.period:
                        cell_period = normalize_text(str(cell.get("period") or table_context))
                        if normalize_text(filters.period) in cell_period:
                            score += 1.0
                    if filters.currency:
                        unit_hint = normalize_text(str(cell.get("unit_hint") or table_context))
                        if normalize_text(filters.currency) in unit_hint or (
                            filters.currency == "CNY" and "人民币" in unit_hint
                        ):
                            score += 0.6
                    if "%" in question and "%" in str(cell.get("raw_value")):
                        score += 0.5

                    if score > best_score:
                        best_score = score
                        best_match = {
                            "source_doc_id": doc_id,
                            "table_id": cell.get("table_id"),
                            "page": cell.get("page"),
                            "row_idx": cell.get("row_idx"),
                            "col_idx": cell.get("col_idx"),
                            "matched_row_headers": cell.get("matched_row_headers") or [],
                            "matched_col_headers": cell.get("matched_col_headers") or [],
                            "unit": cell.get("unit_hint"),
                            "period": cell.get("period"),
                            "raw_value": cell.get("raw_value"),
                            "normalized_value": cell.get("normalized_value"),
                            "footnote_refs": cell.get("footnote_refs") or [],
                            "table_snippet": table_context[:320],
                            "match_score": round(score, 4),
                        }

        if not best_match or best_score < 2.2:
            return None

        best_match["normalized_value"] = parse_numeric_value(
            str(best_match.get("raw_value") or ""),
            unit_hint=best_match.get("unit"),
        )
        return best_match
