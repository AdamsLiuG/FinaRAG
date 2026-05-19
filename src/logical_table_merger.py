from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.text_normalization import normalize_text


_CONTINUATION_RE = re.compile(r"(续表|续页|continued)", re.IGNORECASE)
_TERMINAL_ROW_TERMS = (
    "合计",
    "小计",
    "总计",
    "本期末",
    "期末余额",
    "结束",
    "合并",
)


def _compact_text(value: object) -> str:
    return normalize_text(str(value or "")).replace(" ", "")


def _coerce_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_list(values: Iterable[object]) -> List[str]:
    normalized: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _mapping_values(mapping: object, key: int) -> List[str]:
    if not isinstance(mapping, dict):
        return []
    values = mapping.get(key)
    if values is None:
        values = mapping.get(str(key))
    if isinstance(values, list):
        return _string_list(values)
    if values in (None, ""):
        return []
    return [str(values)]


def _max_mapping_index(mapping: object) -> int:
    if not isinstance(mapping, dict):
        return -1
    indices: List[int] = []
    for key in mapping:
        parsed = _coerce_int(key)
        if parsed is not None:
            indices.append(parsed)
    return max(indices) if indices else -1


class LogicalTableMerger:
    def __init__(
        self,
        *,
        confirmed_threshold: float = 0.62,
        candidate_threshold: float = 0.42,
        materialize_threshold: float = 0.72,
    ):
        self.confirmed_threshold = float(confirmed_threshold)
        self.candidate_threshold = float(candidate_threshold)
        self.materialize_threshold = float(materialize_threshold)

    def link_tables(self, tables: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        enriched_tables = [self._with_defaults(deepcopy(table)) for table in tables]
        if len(enriched_tables) < 2:
            return enriched_tables, []

        candidates = self._candidate_edges(enriched_tables)
        selected_edges = self._select_confirmed_edges(candidates)
        logical_tables = self._assign_logical_ids(enriched_tables, selected_edges)
        self._mark_candidates(enriched_tables, candidates, selected_edges)
        return enriched_tables, logical_tables

    def materialize_logical_table(
        self,
        structured_tables: List[Dict[str, Any]],
        logical_tables: List[Dict[str, Any]],
        logical_table_id: object,
    ) -> Optional[Dict[str, Any]]:
        logical_table_id = str(logical_table_id or "").strip()
        if not logical_table_id:
            return None

        registry = next(
            (
                record
                for record in logical_tables or []
                if str(record.get("logical_table_id") or "") == logical_table_id
                and str(record.get("merge_state") or "") == "confirmed"
                and bool(record.get("materializable", False))
            ),
            None,
        )
        if registry is None:
            return None

        table_lookup = {str(table.get("table_id")): table for table in structured_tables or []}
        member_tables = [
            table_lookup[table_id]
            for table_id in registry.get("member_table_ids") or []
            if str(table_id) in table_lookup
        ]
        if len(member_tables) < 2:
            return None

        head_table = member_tables[0]
        inherited_unit = self._preferred_unit(member_tables)
        inherited_period = self._preferred_period(member_tables)
        inherited_headers = self._column_headers(head_table)

        member_pages = [int(table.get("page")) for table in member_tables if _coerce_int(table.get("page")) is not None]
        materialized_cells: List[Dict[str, Any]] = []
        row_offset = 0
        for table in member_tables:
            table_headers = self._column_headers(table) or inherited_headers
            table_unit = table.get("unit_hint") or inherited_unit
            table_period = table.get("period") or inherited_period
            max_row_idx = 0
            for cell in sorted(
                table.get("cell_records") or [],
                key=lambda item: (
                    _coerce_int(item.get("row_idx")) or 0,
                    _coerce_int(item.get("col_idx")) or 0,
                ),
            ):
                row_idx = _coerce_int(cell.get("row_idx")) or 0
                col_idx = _coerce_int(cell.get("col_idx")) or 0
                max_row_idx = max(max_row_idx, row_idx)
                materialized_cell = dict(cell)
                materialized_cell["logical_table_id"] = logical_table_id
                materialized_cell["logical_table_materialized"] = True
                materialized_cell["source_table_id"] = cell.get("table_id") or table.get("table_id")
                materialized_cell["source_page"] = cell.get("page") or table.get("page")
                materialized_cell["source_row_idx"] = row_idx
                materialized_cell["source_col_idx"] = col_idx
                materialized_cell["materialized_from_tables"] = list(registry.get("member_table_ids") or [])
                materialized_cell["member_pages"] = list(registry.get("member_pages") or [])
                materialized_cell["logical_row_idx"] = row_offset + row_idx
                materialized_cell["logical_col_idx"] = col_idx
                materialized_cell["page_span"] = list(registry.get("page_span") or [])
                materialized_cell["table_id"] = cell.get("table_id") or table.get("table_id")
                materialized_cell["page"] = cell.get("page") or table.get("page")
                materialized_cell["matched_row_headers"] = _string_list(materialized_cell.get("matched_row_headers") or [])
                materialized_cell["matched_col_headers"] = _string_list(
                    materialized_cell.get("matched_col_headers") or table_headers.get(col_idx) or inherited_headers.get(col_idx)
                )
                materialized_cell["unit_hint"] = materialized_cell.get("unit_hint") or table_unit
                materialized_cell["period"] = materialized_cell.get("period") or table_period
                materialized_cells.append(materialized_cell)
            row_offset += max_row_idx + 1

        return {
            "logical_table_id": logical_table_id,
            "page": head_table.get("page"),
            "page_span": list(registry.get("page_span") or []),
            "member_pages": member_pages,
            "member_table_ids": list(registry.get("member_table_ids") or []),
            "materialized_from_tables": list(registry.get("member_table_ids") or []),
            "merge_confidence": registry.get("merge_confidence"),
            "merge_state": registry.get("merge_state"),
            "logical_table_materialized": True,
            "materializable": True,
            "caption": head_table.get("caption"),
            "section_title": head_table.get("section_title"),
            "section_name": head_table.get("section_name"),
            "report_section": head_table.get("report_section"),
            "unit_hint": inherited_unit,
            "period": inherited_period,
            "markdown": "\n\n".join(str(table.get("markdown") or "") for table in member_tables if table.get("markdown")),
            "col_headers_by_col": {str(key): list(value) for key, value in inherited_headers.items()},
            "cell_records": materialized_cells,
        }

    def _candidate_edges(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for left_index, left in enumerate(tables):
            left_page = _coerce_int(left.get("page"))
            if left_page is None:
                continue
            for right_index, right in enumerate(tables):
                right_page = _coerce_int(right.get("page"))
                if right_page is None or right_page != left_page + 1:
                    continue
                score, signals = self._edge_score(left, right)
                if score < self.candidate_threshold:
                    continue
                candidates.append(
                    {
                        "left_index": left_index,
                        "right_index": right_index,
                        "left_table_id": left.get("table_id"),
                        "right_table_id": right.get("table_id"),
                        "score": round(score, 4),
                        "signals": signals,
                        "state": "confirmed" if score >= self.confirmed_threshold else "candidate",
                    }
                )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates

    def _select_confirmed_edges(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        used_left = set()
        used_right = set()
        for edge in candidates:
            if edge["state"] != "confirmed":
                continue
            if edge["left_index"] in used_left or edge["right_index"] in used_right:
                continue
            selected.append(edge)
            used_left.add(edge["left_index"])
            used_right.add(edge["right_index"])
        return selected

    def _assign_logical_ids(self, tables: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not edges:
            return []

        next_by_left = {edge["left_index"]: edge for edge in edges}
        prev_by_right = {edge["right_index"]: edge for edge in edges}
        logical_tables: List[Dict[str, Any]] = []
        visited = set()

        for head_index in sorted(next_by_left):
            if head_index in visited or head_index in prev_by_right:
                continue
            chain_indices = [head_index]
            chain_edges: List[Dict[str, Any]] = []
            current_index = head_index
            while current_index in next_by_left:
                edge = next_by_left[current_index]
                chain_edges.append(edge)
                current_index = edge["right_index"]
                chain_indices.append(current_index)
            for index in chain_indices:
                visited.add(index)

            chain_tables = [tables[index] for index in chain_indices]
            member_table_ids = [str(table.get("table_id")) for table in chain_tables]
            member_pages = [int(table.get("page")) for table in chain_tables if _coerce_int(table.get("page")) is not None]
            page_span = [min(member_pages), max(member_pages)] if member_pages else []
            logical_table_id = f"logical::{member_table_ids[0]}::p{page_span[0]}-{page_span[1]}" if page_span else f"logical::{member_table_ids[0]}"
            chain_confidence = round(min(edge["score"] for edge in chain_edges), 4) if chain_edges else 0.0
            materializable = chain_confidence >= self.materialize_threshold

            for offset, index in enumerate(chain_indices):
                table = tables[index]
                table["logical_table_id"] = logical_table_id
                table["continuation_of"] = member_table_ids[offset - 1] if offset > 0 else None
                table["logical_role"] = (
                    "singleton"
                    if len(chain_indices) == 1
                    else "head"
                    if offset == 0
                    else "tail"
                    if offset == len(chain_indices) - 1
                    else "middle"
                )
                table["page_span"] = list(page_span)
                table["merge_confidence"] = chain_confidence
                table["merge_state"] = "confirmed"
                inbound_edge = prev_by_right.get(index) or next_by_left.get(index)
                table["merge_signals"] = dict((inbound_edge or {}).get("signals") or {})
                for cell in table.get("cell_records") or []:
                    cell["logical_table_id"] = logical_table_id
                    cell["page_span"] = list(page_span)

            logical_tables.append(
                {
                    "logical_table_id": logical_table_id,
                    "head_table_id": member_table_ids[0],
                    "member_table_ids": member_table_ids,
                    "member_pages": member_pages,
                    "page_span": list(page_span),
                    "merge_confidence": chain_confidence,
                    "merge_state": "confirmed",
                    "materializable": materializable,
                    "merge_signals": [dict(edge.get("signals") or {}) for edge in chain_edges],
                }
            )

        return logical_tables

    def _mark_candidates(
        self,
        tables: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        selected_edges: List[Dict[str, Any]],
    ) -> None:
        selected_pairs = {(edge["left_index"], edge["right_index"]) for edge in selected_edges}
        for edge in candidates:
            pair = (edge["left_index"], edge["right_index"])
            if pair in selected_pairs or edge["state"] != "candidate":
                continue
            left = tables[edge["left_index"]]
            right = tables[edge["right_index"]]
            if left.get("merge_state") == "none":
                left["merge_state"] = "candidate"
                left["merge_confidence"] = edge["score"]
                left["merge_signals"] = dict(edge.get("signals") or {})
                left["page_span"] = [left.get("page"), right.get("page")]
            if right.get("merge_state") == "none":
                right["merge_state"] = "candidate"
                right["merge_confidence"] = edge["score"]
                right["merge_signals"] = dict(edge.get("signals") or {})
                right["page_span"] = [left.get("page"), right.get("page")]

    def _with_defaults(self, table: Dict[str, Any]) -> Dict[str, Any]:
        table.setdefault("caption", table.get("caption"))
        table.setdefault("unit_hint", table.get("unit_hint"))
        table.setdefault("period", table.get("period"))
        table.setdefault("col_headers_by_col", table.get("col_headers_by_col") or {})
        table.setdefault("row_headers_by_row", table.get("row_headers_by_row") or {})
        table.setdefault("logical_table_id", None)
        table.setdefault("continuation_of", None)
        table.setdefault("logical_role", "singleton")
        table.setdefault("page_span", [table.get("page"), table.get("page")] if table.get("page") is not None else [])
        table.setdefault("merge_confidence", 0.0)
        table.setdefault("merge_state", "none")
        table.setdefault("merge_signals", {})
        return table

    def _edge_score(self, left: Dict[str, Any], right: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        signals = {
            "adjacent_pages": True,
            "geometry_edge_match": False,
            "section_match": False,
            "column_match": False,
            "unit_match": False,
            "continuation_marker": False,
            "repeated_headers": False,
            "terminal_row_penalty": False,
            "row_continuity": False,
        }
        score = 0.18

        geometry_score = self._geometry_score(left, right)
        if geometry_score > 0:
            score += geometry_score
            signals["geometry_edge_match"] = True

        section_score = self._section_score(left, right)
        if section_score > 0:
            score += section_score
            signals["section_match"] = True

        column_score, repeated_headers = self._column_score(left, right)
        if column_score > 0:
            score += column_score
            signals["column_match"] = True
            signals["repeated_headers"] = repeated_headers

        unit_score = self._unit_score(left, right)
        if unit_score > 0:
            score += unit_score
            signals["unit_match"] = True

        if self._has_continuation_marker(right):
            score += 0.18
            signals["continuation_marker"] = True

        if not self._has_terminal_row(left):
            score += 0.06
            signals["row_continuity"] = self._row_continuity(left, right)
            if signals["row_continuity"]:
                score += 0.04
        else:
            score -= 0.22
            signals["terminal_row_penalty"] = True

        return max(0.0, min(score, 1.0)), signals

    def _geometry_score(self, left: Dict[str, Any], right: Dict[str, Any]) -> float:
        left_bbox = left.get("bbox") or []
        right_bbox = right.get("bbox") or []
        if len(left_bbox) != 4 or len(right_bbox) != 4:
            return 0.0
        left_bottom = float(left_bbox[3])
        right_top = float(right_bbox[1])
        scale = max(left_bottom, float(right_bbox[3]), 1.0)
        if scale <= 2.0:
            if left_bottom >= 0.78 and right_top <= 0.22:
                return 0.12
            if left_bottom >= 0.68 and right_top <= 0.30:
                return 0.07
            return 0.0
        if left_bottom >= 700 and right_top <= 120:
            return 0.12
        if left_bottom >= 620 and right_top <= 180:
            return 0.07
        return 0.0

    def _section_score(self, left: Dict[str, Any], right: Dict[str, Any]) -> float:
        for field in ("section_name", "report_section", "section_title"):
            left_text = _compact_text(left.get(field))
            right_text = _compact_text(right.get(field))
            if left_text and right_text and left_text == right_text:
                return 0.14
        return 0.0

    def _column_score(self, left: Dict[str, Any], right: Dict[str, Any]) -> Tuple[float, bool]:
        left_ncols = _coerce_int(left.get("ncols")) or _coerce_int(left.get("#-cols"))
        right_ncols = _coerce_int(right.get("ncols")) or _coerce_int(right.get("#-cols"))
        if left_ncols is None or right_ncols is None or left_ncols != right_ncols:
            return 0.0, False

        left_headers = self._column_headers(left)
        right_headers = self._column_headers(right)
        if left_headers and right_headers:
            overlap = 0
            for col_idx, headers in left_headers.items():
                if headers and headers == right_headers.get(col_idx):
                    overlap += 1
            if overlap == len(left_headers):
                return 0.18, True
            if overlap > 0:
                return 0.1, True
        if left_ncols == right_ncols and self._has_continuation_marker(right):
            return 0.1, False
        return 0.06, False

    def _unit_score(self, left: Dict[str, Any], right: Dict[str, Any]) -> float:
        left_unit = _compact_text(left.get("unit_hint"))
        right_unit = _compact_text(right.get("unit_hint"))
        if left_unit and right_unit and left_unit == right_unit:
            return 0.08
        if left_unit and not right_unit:
            return 0.06
        return 0.0

    def _row_continuity(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        left_last = self._last_row_headers(left)
        right_first = self._first_row_headers(right)
        if not right_first:
            return False
        if not left_last:
            return True
        return _compact_text(" ".join(left_last)) != _compact_text(" ".join(right_first))

    def _has_continuation_marker(self, table: Dict[str, Any]) -> bool:
        haystack = " ".join(
            str(table.get(field) or "")
            for field in ("caption", "markdown", "section_title", "section_name", "report_section")
        )
        return bool(_CONTINUATION_RE.search(haystack))

    def _has_terminal_row(self, table: Dict[str, Any]) -> bool:
        last_headers = " ".join(self._last_row_headers(table))
        compact = _compact_text(last_headers)
        return any(term in compact for term in map(_compact_text, _TERMINAL_ROW_TERMS))

    def _last_row_headers(self, table: Dict[str, Any]) -> List[str]:
        mapping = table.get("row_headers_by_row") or {}
        max_index = _max_mapping_index(mapping)
        if max_index >= 0:
            return _mapping_values(mapping, max_index)
        rows = [cell.get("matched_row_headers") or [] for cell in table.get("cell_records") or []]
        return _string_list(rows[-1]) if rows else []

    def _first_row_headers(self, table: Dict[str, Any]) -> List[str]:
        mapping = table.get("row_headers_by_row") or {}
        indices = sorted(
            parsed
            for parsed in (_coerce_int(key) for key in mapping.keys())
            if parsed is not None
        )
        if indices:
            return _mapping_values(mapping, indices[0])
        rows = [cell.get("matched_row_headers") or [] for cell in table.get("cell_records") or []]
        return _string_list(rows[0]) if rows else []

    def _column_headers(self, table: Dict[str, Any]) -> Dict[int, List[str]]:
        mapping = table.get("col_headers_by_col") or {}
        headers: Dict[int, List[str]] = {}
        for key in list(mapping.keys()):
            parsed = _coerce_int(key)
            if parsed is None:
                continue
            values = _mapping_values(mapping, parsed)
            if values:
                headers[parsed] = values
        if headers:
            return headers

        derived: Dict[int, List[str]] = {}
        for cell in table.get("cell_records") or []:
            col_idx = _coerce_int(cell.get("col_idx"))
            if col_idx is None:
                continue
            values = _string_list(cell.get("matched_col_headers") or [])
            if values:
                derived[col_idx] = values
        return derived

    def _preferred_unit(self, tables: List[Dict[str, Any]]) -> Optional[str]:
        for table in tables:
            unit = str(table.get("unit_hint") or "").strip()
            if unit:
                return unit
        for table in tables:
            for cell in table.get("cell_records") or []:
                unit = str(cell.get("unit_hint") or "").strip()
                if unit:
                    return unit
        return None

    def _preferred_period(self, tables: List[Dict[str, Any]]) -> Optional[str]:
        for table in tables:
            period = str(table.get("period") or "").strip()
            if period:
                return period
        for table in tables:
            for cell in table.get("cell_records") or []:
                period = str(cell.get("period") or "").strip()
                if period:
                    return period
        return None
