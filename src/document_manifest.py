from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_ALIAS_SEPARATORS = ("|", ";", ",", "、", "，", "/", "\\")


def _coerce_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        lowered = stripped.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return stripped
    return value


def _split_aliases(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        candidates = raw_value
    else:
        buffer = str(raw_value)
        for separator in _ALIAS_SEPARATORS[1:]:
            buffer = buffer.replace(separator, _ALIAS_SEPARATORS[0])
        candidates = buffer.split(_ALIAS_SEPARATORS[0])

    aliases: List[str] = []
    seen = set()
    for candidate in candidates:
        alias = str(candidate).strip().strip('"')
        if not alias or alias in seen:
            continue
        seen.add(alias)
        aliases.append(alias)
    return aliases


def _normalize_doc_source_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    research_keywords = ("research", "broker", "研报", "深度", "点评", "首次覆盖", "行业")
    annual_keywords = ("annual", "10-k", "年报", "年度报告", "annual report")
    interim_keywords = ("q1", "q2", "q3", "quarter", "季报", "中报", "半年报")

    if any(keyword in text for keyword in research_keywords):
        return "research_report"
    if any(keyword in text for keyword in annual_keywords):
        return "annual_report"
    if any(keyword in text for keyword in interim_keywords):
        return "interim_report"
    return text


def _normalize_language(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"zh", "zh-cn", "zh-hans", "cn", "chinese"}:
        return "zh"
    if text in {"bilingual", "multi", "multilingual", "zh+en", "en+zh"}:
        return "bilingual"
    return "en" if text in {"", "english", "en"} else text


def _normalize_manifest_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized_row = {
        key: _coerce_value(value)
        for key, value in row.items()
        if value not in (None, "")
    }
    doc_id = (
        normalized_row.get("doc_id")
        or normalized_row.get("sha1")
        or normalized_row.get("sha1_name")
        or normalized_row.get("document_id")
    )
    company_name = str(normalized_row.get("company_name") or normalized_row.get("name") or "").strip('" ')
    report_type = normalized_row.get("report_type") or normalized_row.get("doc_type") or normalized_row.get("filing_type")
    doc_source_type = _normalize_doc_source_type(
        normalized_row.get("doc_source_type")
        or normalized_row.get("source_type")
        or report_type
    )
    aliases = _split_aliases(
        normalized_row.get("company_aliases")
        or normalized_row.get("aliases")
        or normalized_row.get("company_short_name")
        or normalized_row.get("ticker")
        or normalized_row.get("security_code")
    )
    if company_name and company_name not in aliases:
        aliases.insert(0, company_name)

    normalized_row["doc_id"] = doc_id
    normalized_row["company_name"] = company_name
    normalized_row["company_aliases"] = aliases
    normalized_row["security_code"] = normalized_row.get("security_code") or normalized_row.get("ticker") or normalized_row.get("stock_code")
    normalized_row["doc_source_type"] = doc_source_type
    normalized_row["report_type"] = report_type
    normalized_row["report_title"] = normalized_row.get("report_title") or normalized_row.get("title")
    normalized_row["broker_name"] = normalized_row.get("broker_name") or normalized_row.get("broker")
    normalized_row["major_industry"] = normalized_row.get("major_industry") or normalized_row.get("industry")
    normalized_row["language"] = _normalize_language(normalized_row.get("language"))
    normalized_row["currency"] = normalized_row.get("currency") or normalized_row.get("cur")
    normalized_row["fiscal_year"] = normalized_row.get("fiscal_year") or normalized_row.get("report_year")
    return normalized_row


def _read_manifest_rows(manifest_path: Path) -> Iterable[Dict[str, Any]]:
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("documents") or payload.get("items") or []
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError(f"Unsupported manifest JSON format in {manifest_path}")
        return rows

    with open(manifest_path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_document_manifest(manifest_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if manifest_path is None or not manifest_path.exists():
        return {}

    manifest: Dict[str, Dict[str, Any]] = {}
    for row in _read_manifest_rows(manifest_path):
        if not isinstance(row, dict):
            continue
        normalized_row = _normalize_manifest_row(row)
        doc_id = normalized_row.get("doc_id")
        if not doc_id:
            continue
        manifest[str(doc_id)] = normalized_row
    return manifest
