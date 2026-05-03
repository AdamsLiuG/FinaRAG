from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_TAG_FIELDS = (
    "business_tags",
    "strategy_tags",
    "factor_tags",
    "chain_position_minor",
    "listing_tags",
    "ownership_tags",
    "status_tags",
    "style_tags",
)

_PAGE_METADATA_FIELDS = (
    "chunk_id",
    "stock_code",
    "company_name",
    "report_id",
    "report_year",
    "report_type",
    "exchange",
    "board",
    "market_type",
    "industry_code_raw",
    "industry_name_raw",
    "industry_l1",
    "industry_l2",
    "business_tags",
    "strategy_tags",
    "factor_tags",
    "chain_position_major",
    "chain_position_minor",
    "listing_tags",
    "ownership_tags",
    "status_tags",
    "style_tags",
    "page_start",
    "page_end",
    "section_name",
    "section_l1",
    "section_l2",
    "section_l3",
    "section_path",
    "section_leaf",
    "local_heading",
    "block_index",
    "chunk_kind",
    "page_role",
    "file_path",
)

_PDFCRAWL_CHUNK_FIELDS = (
    "chunk_id",
    "parent_chunk_id",
    "stock_code",
    "company_name",
    "report_id",
    "report_year",
    "page_start",
    "page_end",
    "section_name",
    "section_l1",
    "section_l2",
    "section_l3",
    "section_path",
    "section_leaf",
    "local_heading",
    "block_index",
    "chunk_kind",
    "page_role",
    "file_path",
    "embedding_text",
    "search_text",
)

_COMPANY_PROFILE_FIELDS = (
    "company_name",
    "website",
    "mailbox",
    "tel",
    "fax",
    "established_date",
    "listed_date",
    "register_address",
    "office_address",
    "isin",
    "industry_raw",
    "detail_industry_raw",
    "market_raw",
    "exchange_raw",
    "board_raw",
    "main_business_raw",
    "business_scope_raw",
    "highlight_raw",
    "risk_raw",
    "controller_section_raw",
    "top_shareholder_raw",
    "related_securities_raw",
    "source_url",
    "source_status",
    "error",
    "dividend_history_raw",
)


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def _to_int(value: Any) -> Optional[int]:
    value = _clean_scalar(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    normalized: List[Any] = []
    seen = set()
    for item in items:
        if isinstance(item, (dict, list)):
            candidate = item
            marker = json.dumps(item, ensure_ascii=False, sort_keys=True)
        else:
            candidate = _clean_scalar(item)
            if candidate is None:
                continue
            marker = str(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        normalized.append(candidate)
    return normalized


def _merge_unique(values: Iterable[Any]) -> List[Any]:
    merged: List[Any] = []
    seen = set()
    for value in values:
        if isinstance(value, list):
            candidates = value
        else:
            candidates = [value]
        for candidate in candidates:
            if isinstance(candidate, (dict, list)):
                marker = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
            else:
                cleaned = _clean_scalar(candidate)
                if cleaned is None:
                    continue
                candidate = cleaned
                marker = str(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(candidate)
    return merged


def _section_segments(record: Dict[str, Any]) -> List[str]:
    section_path = _clean_scalar(record.get("section_path"))
    if isinstance(section_path, str) and section_path:
        segments = [segment.strip() for segment in section_path.split(">") if segment.strip()]
        if segments:
            return segments

    segments = []
    for field in ("section_l1", "section_l2", "section_l3"):
        value = _clean_scalar(record.get(field))
        if value:
            segments.append(str(value))
    section_leaf = _clean_scalar(record.get("section_leaf"))
    if section_leaf and (not segments or str(section_leaf) != segments[-1]):
        segments.append(str(section_leaf))
    section_name = _clean_scalar(record.get("section_name"))
    if not segments and section_name:
        segments.append(str(section_name))
    return segments


def _common_prefix(sequences: List[List[str]]) -> List[str]:
    if not sequences:
        return []
    prefix = list(sequences[0])
    for sequence in sequences[1:]:
        max_index = min(len(prefix), len(sequence))
        match_count = 0
        while match_count < max_index and prefix[match_count] == sequence[match_count]:
            match_count += 1
        prefix = prefix[:match_count]
        if not prefix:
            break
    return prefix


def _first_nonempty(records: Iterable[Dict[str, Any]], field: str) -> Any:
    for record in records:
        value = _clean_scalar(record.get(field))
        if value is not None:
            return value
    return None


def _normalize_chunk_type(chunk_kind: Any, node_type: str) -> str:
    normalized = str(_clean_scalar(chunk_kind) or "").strip().lower()
    mapping = {
        "text": "content",
        "content": "content",
        "table": "serialized_table",
        "serialized_table": "serialized_table",
        "chart": "chart_to_table",
        "chart_to_table": "chart_to_table",
    }
    if normalized in mapping:
        return mapping[normalized]
    return "content" if node_type in {"parent", "child"} else normalized or "content"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_report_page_lookup(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    lookup: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for record in records:
        report_id = str(record.get("report_id") or "").strip()
        page = _to_int(record.get("page"))
        if not report_id or page is None:
            continue
        if page in lookup[report_id]:
            raise ValueError(f"Duplicate PDFCrawl page metadata found for ({report_id}, {page}).")
        lookup[report_id][page] = record
    return {report_id: dict(page_map) for report_id, page_map in lookup.items()}


def load_report_page_lookup(metadata_store_dir: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    report_page_path = metadata_store_dir / "report_page.jsonl"
    if not report_page_path.exists():
        return {}
    return build_report_page_lookup(_read_jsonl(report_page_path))


def load_company_label_snapshot_lookup(metadata_store_dir: Path) -> Dict[str, Dict[str, Any]]:
    snapshot_path = metadata_store_dir / "company_label_snapshot.jsonl"
    if not snapshot_path.exists():
        return {}
    lookup: Dict[str, Dict[str, Any]] = {}
    for record in _read_jsonl(snapshot_path):
        report_id = str(record.get("report_id") or "").strip()
        if report_id:
            lookup[report_id] = record
    return lookup


def _normalize_report_page_record(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    report_id = str(raw_record.get("report_id") or "").strip()
    page_start = _to_int(raw_record.get("page_start"))
    page_end = _to_int(raw_record.get("page_end"))
    if not report_id:
        raise ValueError("PDFCrawl metadata row is missing report_id.")
    if page_start is None or page_end is None:
        raise ValueError(f"PDFCrawl metadata row for '{report_id}' is missing page_start/page_end.")
    if page_start != page_end:
        raise ValueError(
            f"PDFCrawl metadata row for '{report_id}' spans multiple pages ({page_start}, {page_end}); "
            "current FinaRAG join logic requires page_start == page_end."
        )

    normalized = {field: raw_record.get(field) for field in _PAGE_METADATA_FIELDS}
    normalized["report_id"] = report_id
    normalized["stock_code"] = _clean_scalar(raw_record.get("stock_code"))
    normalized["company_name"] = _clean_scalar(raw_record.get("company_name"))
    normalized["report_year"] = _to_int(raw_record.get("report_year"))
    normalized["page_start"] = page_start
    normalized["page_end"] = page_end
    normalized["page"] = page_start
    normalized["section_name"] = _clean_scalar(raw_record.get("section_name")) or f"Page {page_start}"
    normalized["section_l1"] = _clean_scalar(raw_record.get("section_l1"))
    normalized["section_l2"] = _clean_scalar(raw_record.get("section_l2"))
    normalized["section_l3"] = _clean_scalar(raw_record.get("section_l3"))
    normalized["section_path"] = _clean_scalar(raw_record.get("section_path"))
    normalized["section_leaf"] = _clean_scalar(raw_record.get("section_leaf")) or normalized["section_name"]
    normalized["local_heading"] = _clean_scalar(raw_record.get("local_heading"))
    normalized["block_index"] = _to_int(raw_record.get("block_index"))
    normalized["chunk_kind"] = _clean_scalar(raw_record.get("chunk_kind"))
    normalized["page_role"] = _clean_scalar(raw_record.get("page_role")) or "content"
    normalized["file_path"] = _clean_scalar(raw_record.get("file_path"))
    normalized["chain_position_major"] = _clean_scalar(raw_record.get("chain_position_major"))
    for field in _TAG_FIELDS:
        normalized[field] = _coerce_list(raw_record.get(field))
    return normalized


def _aggregate_report_page_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        report_id = str(record.get("report_id") or "").strip()
        page = _to_int(record.get("page"))
        if not report_id or page is None:
            continue
        grouped[(report_id, page)].append(record)

    aggregated_rows: List[Dict[str, Any]] = []
    for (report_id, page), page_records in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        first = page_records[0]
        common_section = _common_prefix([_section_segments(record) for record in page_records if _section_segments(record)])
        if common_section:
            section_path = " > ".join(common_section)
            section_l1 = common_section[0] if len(common_section) >= 1 else None
            section_l2 = common_section[1] if len(common_section) >= 2 else None
            section_l3 = common_section[2] if len(common_section) >= 3 else None
            section_leaf = common_section[-1]
            section_name = section_leaf
        else:
            section_path = _clean_scalar(first.get("section_path")) or _clean_scalar(first.get("section_name")) or f"Page {page}"
            section_l1 = _clean_scalar(first.get("section_l1"))
            section_l2 = _clean_scalar(first.get("section_l2"))
            section_l3 = _clean_scalar(first.get("section_l3"))
            section_leaf = _clean_scalar(first.get("section_leaf")) or _clean_scalar(first.get("section_name")) or f"Page {page}"
            section_name = _clean_scalar(first.get("section_name")) or section_leaf

        aggregated = {
            "chunk_id": f"{report_id}_page_{page:04d}",
            "stock_code": _first_nonempty(page_records, "stock_code"),
            "company_name": _first_nonempty(page_records, "company_name"),
            "report_id": report_id,
            "report_year": _first_nonempty(page_records, "report_year"),
            "report_type": _first_nonempty(page_records, "report_type"),
            "exchange": _first_nonempty(page_records, "exchange"),
            "board": _first_nonempty(page_records, "board"),
            "market_type": _first_nonempty(page_records, "market_type"),
            "industry_code_raw": _first_nonempty(page_records, "industry_code_raw"),
            "industry_name_raw": _first_nonempty(page_records, "industry_name_raw"),
            "industry_l1": _first_nonempty(page_records, "industry_l1"),
            "industry_l2": _first_nonempty(page_records, "industry_l2"),
            "chain_position_major": _first_nonempty(page_records, "chain_position_major"),
            "page_start": page,
            "page_end": page,
            "page": page,
            "section_name": section_name,
            "section_l1": section_l1,
            "section_l2": section_l2,
            "section_l3": section_l3,
            "section_path": section_path,
            "section_leaf": section_leaf,
            "local_heading": _first_nonempty(page_records, "local_heading"),
            "block_index": None,
            "chunk_kind": "page",
            "page_role": _first_nonempty(page_records, "page_role") or "content",
            "file_path": _first_nonempty(page_records, "file_path"),
        }
        for field in _TAG_FIELDS:
            aggregated[field] = _merge_unique(record.get(field) or [] for record in page_records)
        aggregated_rows.append(aggregated)
    return aggregated_rows


def _normalize_pdfcrawl_chunk_record(raw_record: Dict[str, Any], *, node_type: str) -> Dict[str, Any]:
    report_id = str(raw_record.get("report_id") or "").strip()
    page_start = _to_int(raw_record.get("page_start"))
    page_end = _to_int(raw_record.get("page_end"))
    if not report_id:
        raise ValueError("PDFCrawl chunk row is missing report_id.")
    if page_start is None or page_end is None:
        raise ValueError(f"PDFCrawl chunk row for '{report_id}' is missing page_start/page_end.")

    section_name = _clean_scalar(raw_record.get("section_name")) or _clean_scalar(raw_record.get("section_leaf")) or f"Page {page_start}"
    section_path = _clean_scalar(raw_record.get("section_path")) or section_name
    section_leaf = _clean_scalar(raw_record.get("section_leaf")) or section_name
    local_heading = _clean_scalar(raw_record.get("local_heading")) or section_leaf
    chunk_id = _clean_scalar(raw_record.get("chunk_id"))
    normalized = {
        "doc_id": report_id,
        "report_id": report_id,
        "stock_code": _clean_scalar(raw_record.get("stock_code")),
        "company_name": _clean_scalar(raw_record.get("company_name")),
        "report_year": _to_int(raw_record.get("report_year")),
        "chunk_id": chunk_id,
        "node_type": node_type,
        "chunk_type": _normalize_chunk_type(raw_record.get("chunk_kind"), node_type),
        "parent_chunk_id": _clean_scalar(raw_record.get("parent_chunk_id")),
        "page": page_start,
        "page_start": page_start,
        "page_end": page_end,
        "section_name": section_name,
        "section_title": local_heading,
        "report_section": section_path,
        "section_l1": _clean_scalar(raw_record.get("section_l1")),
        "section_l2": _clean_scalar(raw_record.get("section_l2")),
        "section_l3": _clean_scalar(raw_record.get("section_l3")),
        "section_path": section_path,
        "section_leaf": section_leaf,
        "local_heading": local_heading,
        "table_id": None,
        "chart_id": None,
        "picture_id": None,
        "chart_type": None,
        "series_name": None,
        "x_label": None,
        "chart_confidence": None,
        "has_chart_context": False,
        "evidence_type": "narrative",
        "has_table_context": False,
        "exchange": _clean_scalar(raw_record.get("exchange")),
        "board": _clean_scalar(raw_record.get("board")),
        "market_type": _clean_scalar(raw_record.get("market_type")),
        "industry_l1": _clean_scalar(raw_record.get("industry_l1")),
        "industry_l2": _clean_scalar(raw_record.get("industry_l2")),
        "chain_position_major": _clean_scalar(raw_record.get("chain_position_major")),
        "embedding_text": _clean_scalar(raw_record.get("embedding_text")),
        "search_text": _clean_scalar(raw_record.get("search_text"))
        or _clean_scalar(raw_record.get("embedding_text"))
        or _clean_scalar(raw_record.get("chunk_text")),
        "chunk_metadata_id": chunk_id,
        "page_role": _clean_scalar(raw_record.get("page_role")) or "content",
        "child_chunk_ids": [],
    }
    for field in _TAG_FIELDS:
        normalized[field] = _coerce_list(raw_record.get(field))
    return normalized


def _attach_child_chunk_ids(parent_chunks: List[Dict[str, Any]], child_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parent_lookup = {
        str(chunk.get("chunk_id")): chunk
        for chunk in parent_chunks
        if _clean_scalar(chunk.get("chunk_id")) is not None
    }
    for child in child_chunks:
        parent_chunk_id = str(_clean_scalar(child.get("parent_chunk_id")) or "")
        child_chunk_id = _clean_scalar(child.get("chunk_id"))
        if not parent_chunk_id or child_chunk_id is None:
            continue
        parent = parent_lookup.get(parent_chunk_id)
        if parent is None:
            continue
        parent["child_chunk_ids"] = _merge_unique([parent.get("child_chunk_ids") or [], [child_chunk_id]])
    return parent_chunks


def _flatten_company_label_record(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    labels = raw_record.get("labels") or {}
    official_industry = labels.get("official_industry") or {}
    theme_tags = labels.get("theme_tags") or {}
    chain_position = labels.get("chain_position") or {}
    company_attributes = labels.get("company_attributes") or {}

    official_fields = official_industry.get("normalized_fields") or {}
    theme_fields = theme_tags.get("normalized_fields") or {}
    chain_fields = chain_position.get("normalized_fields") or {}
    attribute_fields = company_attributes.get("normalized_fields") or {}

    return {
        "snapshot_id": _clean_scalar(raw_record.get("snapshot_id")),
        "report_id": _clean_scalar(raw_record.get("report_id")),
        "report_year": _to_int(raw_record.get("report_year")),
        "stock_code": _clean_scalar(raw_record.get("stock_code")),
        "company_name": _clean_scalar(raw_record.get("company_name")),
        "industry_l1": _clean_scalar(official_fields.get("industry_l1")),
        "industry_l2": _clean_scalar(official_fields.get("industry_l2")),
        "market_type": _clean_scalar(official_fields.get("market_type")),
        "exchange": _clean_scalar(official_fields.get("exchange")),
        "board": _clean_scalar(official_fields.get("board")),
        "business_tags": _coerce_list(theme_fields.get("business_tags")),
        "strategy_tags": _coerce_list(theme_fields.get("strategy_tags")),
        "factor_tags": _coerce_list(theme_fields.get("factor_tags")),
        "chain_position_major": _clean_scalar(chain_fields.get("chain_position_major")),
        "chain_position_minor": _coerce_list(chain_fields.get("chain_position_minor")),
        "listing_tags": _coerce_list(attribute_fields.get("listing_tags")),
        "ownership_tags": _coerce_list(attribute_fields.get("ownership_tags")),
        "status_tags": _coerce_list(attribute_fields.get("status_tags")),
        "style_tags": _coerce_list(attribute_fields.get("style_tags")),
        "label_source": "company_labels.jsonl",
    }


def _derive_snapshot_from_page_records(report_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = records[0]
    return {
        "snapshot_id": report_id,
        "report_id": report_id,
        "report_year": first.get("report_year"),
        "stock_code": first.get("stock_code"),
        "company_name": first.get("company_name"),
        "industry_l1": first.get("industry_l1"),
        "industry_l2": first.get("industry_l2"),
        "market_type": first.get("market_type"),
        "exchange": first.get("exchange"),
        "board": first.get("board"),
        "business_tags": _merge_unique(record.get("business_tags") or [] for record in records),
        "strategy_tags": _merge_unique(record.get("strategy_tags") or [] for record in records),
        "factor_tags": _merge_unique(record.get("factor_tags") or [] for record in records),
        "chain_position_major": first.get("chain_position_major"),
        "chain_position_minor": _merge_unique(record.get("chain_position_minor") or [] for record in records),
        "listing_tags": _merge_unique(record.get("listing_tags") or [] for record in records),
        "ownership_tags": _merge_unique(record.get("ownership_tags") or [] for record in records),
        "status_tags": _merge_unique(record.get("status_tags") or [] for record in records),
        "style_tags": _merge_unique(record.get("style_tags") or [] for record in records),
        "label_source": "metadata.jsonl",
    }


@dataclass
class PdfcrawlMetadataBundle:
    report_pages: List[Dict[str, Any]]
    report_page_lookup: Dict[str, Dict[int, Dict[str, Any]]]
    chunk_metadata_rows: List[Dict[str, Any]]
    company_label_snapshots: Dict[str, Dict[str, Any]]
    company_profiles: Dict[str, Dict[str, Any]]
    metadata_paths: List[Path]
    chunk_paths: List[Path]
    child_chunk_paths: List[Path]
    company_label_paths: List[Path]
    company_profile_paths: List[Path]

    @property
    def report_summaries(self) -> Dict[str, Dict[str, Any]]:
        summaries: Dict[str, Dict[str, Any]] = {}
        for report_id, pages in self.report_page_lookup.items():
            first = next(iter(pages.values()), None)
            if first is None:
                continue
            summaries[report_id] = {
                "exchange": first.get("exchange"),
                "board": first.get("board"),
                "market_type": first.get("market_type"),
                "industry_l1": first.get("industry_l1"),
                "industry_l2": first.get("industry_l2"),
                "stock_code": first.get("stock_code"),
                "company_name": first.get("company_name"),
                "report_year": first.get("report_year"),
                "has_pdfcrawl_metadata": True,
            }
        return summaries


def load_pdfcrawl_metadata(
    manifest_paths: Iterable[Path],
    *,
    metadata_mode: str = "auto",
) -> PdfcrawlMetadataBundle:
    metadata_paths: List[Path] = []
    chunk_paths: List[Path] = []
    child_chunk_paths: List[Path] = []
    company_label_paths: List[Path] = []
    company_profile_paths: List[Path] = []

    for manifest_path in manifest_paths:
        manifest_dir = manifest_path.parent
        candidate_metadata = manifest_dir / "metadata.jsonl"
        candidate_chunks = manifest_dir / "chunks.jsonl"
        candidate_child_chunks = manifest_dir / "child_chunks.jsonl"
        candidate_labels = manifest_dir / "company_labels.jsonl"
        candidate_profiles = manifest_dir / "company_profiles.jsonl"
        if candidate_metadata.exists():
            metadata_paths.append(candidate_metadata)
        if candidate_chunks.exists():
            chunk_paths.append(candidate_chunks)
        if candidate_child_chunks.exists():
            child_chunk_paths.append(candidate_child_chunks)
        if candidate_labels.exists():
            company_label_paths.append(candidate_labels)
        if candidate_profiles.exists():
            company_profile_paths.append(candidate_profiles)

    normalized_mode = (metadata_mode or "auto").strip().lower()
    if normalized_mode not in {"auto", "required", "ignore"}:
        raise ValueError("metadata_mode must be one of: auto, required, ignore")

    if normalized_mode == "ignore":
        return PdfcrawlMetadataBundle(
            report_pages=[],
            report_page_lookup={},
            chunk_metadata_rows=[],
            company_label_snapshots={},
            company_profiles={},
            metadata_paths=[],
            chunk_paths=[],
            child_chunk_paths=[],
            company_label_paths=[],
            company_profile_paths=[],
        )

    if normalized_mode == "required" and not metadata_paths:
        raise FileNotFoundError("No metadata.jsonl files were found alongside the discovered manifest.csv files.")

    source_page_records: List[Dict[str, Any]] = []
    for metadata_path in metadata_paths:
        source_page_records.extend(_normalize_report_page_record(record) for record in _read_jsonl(metadata_path))
    report_pages = _aggregate_report_page_records(source_page_records) if source_page_records else []
    report_page_lookup = build_report_page_lookup(report_pages) if report_pages else {}

    parent_chunk_rows: List[Dict[str, Any]] = []
    for chunk_path in chunk_paths:
        parent_chunk_rows.extend(
            _normalize_pdfcrawl_chunk_record(record, node_type="parent")
            for record in _read_jsonl(chunk_path)
        )
    child_chunk_rows: List[Dict[str, Any]] = []
    for child_chunk_path in child_chunk_paths:
        child_chunk_rows.extend(
            _normalize_pdfcrawl_chunk_record(record, node_type="child")
            for record in _read_jsonl(child_chunk_path)
        )
    parent_chunk_rows = _attach_child_chunk_ids(parent_chunk_rows, child_chunk_rows)
    chunk_metadata_rows = parent_chunk_rows + child_chunk_rows

    company_profiles: Dict[str, Dict[str, Any]] = {}
    for profile_path in company_profile_paths:
        for record in _read_jsonl(profile_path):
            stock_code = _clean_scalar(record.get("stock_code"))
            if not stock_code:
                continue
            flattened = {"stock_code": stock_code}
            for field in _COMPANY_PROFILE_FIELDS:
                if field in {"related_securities_raw", "dividend_history_raw"}:
                    flattened[field] = _coerce_list(record.get(field))
                else:
                    flattened[field] = record.get(field)
            company_profiles[stock_code] = flattened

    company_label_snapshots: Dict[str, Dict[str, Any]] = {}
    for label_path in company_label_paths:
        for record in _read_jsonl(label_path):
            flattened = _flatten_company_label_record(record)
            report_id = str(flattened.get("report_id") or "").strip()
            if report_id:
                company_label_snapshots[report_id] = flattened

    if source_page_records:
        grouped_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in source_page_records:
            grouped_records[record["report_id"]].append(record)
        for report_id, records in grouped_records.items():
            company_label_snapshots.setdefault(report_id, _derive_snapshot_from_page_records(report_id, records))

    return PdfcrawlMetadataBundle(
        report_pages=report_pages,
        report_page_lookup=report_page_lookup,
        chunk_metadata_rows=chunk_metadata_rows,
        company_label_snapshots=company_label_snapshots,
        company_profiles=company_profiles,
        metadata_paths=metadata_paths,
        chunk_paths=chunk_paths,
        child_chunk_paths=child_chunk_paths,
        company_label_paths=company_label_paths,
        company_profile_paths=company_profile_paths,
    )
