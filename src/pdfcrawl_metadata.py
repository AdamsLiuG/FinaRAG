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
    "file_path",
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
    normalized["file_path"] = _clean_scalar(raw_record.get("file_path"))
    normalized["chain_position_major"] = _clean_scalar(raw_record.get("chain_position_major"))
    for field in _TAG_FIELDS:
        normalized[field] = _coerce_list(raw_record.get(field))
    return normalized


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
    company_label_snapshots: Dict[str, Dict[str, Any]]
    company_profiles: Dict[str, Dict[str, Any]]
    metadata_paths: List[Path]
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
    company_label_paths: List[Path] = []
    company_profile_paths: List[Path] = []

    for manifest_path in manifest_paths:
        manifest_dir = manifest_path.parent
        candidate_metadata = manifest_dir / "metadata.jsonl"
        candidate_labels = manifest_dir / "company_labels.jsonl"
        candidate_profiles = manifest_dir / "company_profiles.jsonl"
        if candidate_metadata.exists():
            metadata_paths.append(candidate_metadata)
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
            company_label_snapshots={},
            company_profiles={},
            metadata_paths=[],
            company_label_paths=[],
            company_profile_paths=[],
        )

    if normalized_mode == "required" and not metadata_paths:
        raise FileNotFoundError("No metadata.jsonl files were found alongside the discovered manifest.csv files.")

    report_pages: List[Dict[str, Any]] = []
    for metadata_path in metadata_paths:
        report_pages.extend(_normalize_report_page_record(record) for record in _read_jsonl(metadata_path))
    report_page_lookup = build_report_page_lookup(report_pages) if report_pages else {}

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

    if report_pages:
        grouped_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in report_pages:
            grouped_records[record["report_id"]].append(record)
        for report_id, records in grouped_records.items():
            company_label_snapshots.setdefault(report_id, _derive_snapshot_from_page_records(report_id, records))

    return PdfcrawlMetadataBundle(
        report_pages=report_pages,
        report_page_lookup=report_page_lookup,
        company_label_snapshots=company_label_snapshots,
        company_profiles=company_profiles,
        metadata_paths=metadata_paths,
        company_label_paths=company_label_paths,
        company_profile_paths=company_profile_paths,
    )
