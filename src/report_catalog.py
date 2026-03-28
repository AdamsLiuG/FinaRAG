from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from src.query_plan import QueryPlan
from src.text_normalization import normalize_currency_token, normalize_text


_TOPIC_PREFIXES = ("has_", "mentions_")
_TRUTHY = {"1", "true", "yes", "y", "on"}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY


def _extract_topic_flags(row: Dict[str, Any]) -> List[str]:
    return sorted(
        key
        for key, value in row.items()
        if key.startswith(_TOPIC_PREFIXES) and _coerce_bool(value)
    )


@dataclass
class CompanyReport:
    sha1: str
    company_name: str
    currency: Optional[str]
    major_industry: Optional[str]
    report_year: Optional[int]
    report_type: Optional[str]
    topic_flags: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sha1": self.sha1,
            "company_name": self.company_name,
            "currency": self.currency,
            "major_industry": self.major_industry,
            "report_year": self.report_year,
            "report_type": self.report_type,
            "topic_flags": self.topic_flags,
        }


class ReportCatalog:
    def __init__(self, subset_path: Path | None, documents_dir: Path | None = None):
        self.subset_path = Path(subset_path) if subset_path else None
        self.documents_dir = Path(documents_dir) if documents_dir else None
        self._reports: List[CompanyReport] | None = None

    def _load_document_meta(self) -> Dict[str, Dict[str, Any]]:
        if self.documents_dir is None or not self.documents_dir.exists():
            return {}

        report_meta: Dict[str, Dict[str, Any]] = {}
        for document_path in self.documents_dir.glob("*.json"):
            try:
                with open(document_path, "r", encoding="utf-8") as file:
                    document = json.load(file)
            except Exception:
                continue

            metainfo = document.get("metainfo") or {}
            sha1 = metainfo.get("sha1_name")
            if not sha1:
                continue
            report_meta[sha1] = metainfo
        return report_meta

    def _load_reports(self) -> List[CompanyReport]:
        if self._reports is not None:
            return self._reports
        if self.subset_path is None:
            self._reports = []
            return self._reports

        companies_df = pd.read_csv(self.subset_path)
        document_meta = self._load_document_meta()
        reports: List[CompanyReport] = []

        for row in companies_df.to_dict(orient="records"):
            sha1 = str(row.get("sha1") or "").strip()
            document_entry = document_meta.get(sha1, {})
            company_name = str(row.get("company_name") or row.get("name") or "").strip('" ')
            currency = normalize_currency_token(row.get("cur") or row.get("currency"))
            report_year = document_entry.get("report_year")
            report_type = (
                row.get("report_type")
                or row.get("doc_type")
                or row.get("filing_type")
                or document_entry.get("report_type")
            )
            major_industry = row.get("major_industry") or document_entry.get("major_industry")

            metadata = {key: value for key, value in row.items() if pd.notna(value)}
            metadata.update({key: value for key, value in document_entry.items() if value is not None})

            reports.append(
                CompanyReport(
                    sha1=sha1,
                    company_name=company_name,
                    currency=currency,
                    major_industry=major_industry,
                    report_year=report_year if isinstance(report_year, int) else None,
                    report_type=str(report_type).strip() if report_type else None,
                    topic_flags=_extract_topic_flags(metadata),
                    metadata=metadata,
                )
            )

        self._reports = reports
        return reports

    def get_reports(self) -> List[CompanyReport]:
        return list(self._load_reports())

    def get_company_names(self) -> List[str]:
        return [report.company_name for report in self._load_reports()]

    def get_report_by_company_name(self, company_name: str) -> CompanyReport | None:
        for report in self._load_reports():
            if report.company_name == company_name:
                return report
        return None

    def extract_companies_from_question(self, question_text: str) -> List[str]:
        found_companies: List[str] = []
        company_names = sorted(self.get_company_names(), key=len, reverse=True)
        question_buffer = question_text

        for company in company_names:
            escaped_company = re.escape(company)
            pattern = rf"{escaped_company}(?:\W|$)"
            if re.search(pattern, question_buffer, re.IGNORECASE):
                found_companies.append(company)
                question_buffer = re.sub(pattern, "", question_buffer, flags=re.IGNORECASE)

        return found_companies

    def _score_report(self, report: CompanyReport, query_plan: QueryPlan) -> tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []
        question_text = query_plan.original_query
        normalized_question = normalize_text(question_text)

        if query_plan.filters.currency and report.currency == normalize_currency_token(query_plan.filters.currency):
            score += 2.0
            reasons.append("currency_match")

        if query_plan.filters.year is not None and report.report_year == query_plan.filters.year:
            score += 2.5
            reasons.append("year_match")

        if query_plan.filters.major_industry and report.major_industry:
            if normalize_text(report.major_industry) == normalize_text(query_plan.filters.major_industry):
                score += 1.5
                reasons.append("industry_filter_match")

        if report.major_industry and normalize_text(report.major_industry) in normalized_question:
            score += 1.0
            reasons.append("industry_mention_match")

        matched_topic_flags = sorted(set(query_plan.topic_flags) & set(report.topic_flags))
        if matched_topic_flags:
            score += 3.0 + len(matched_topic_flags)
            reasons.append(f"topic_flags:{','.join(matched_topic_flags)}")

        if query_plan.route_hints.get("report_type") and report.report_type:
            if normalize_text(report.report_type) == normalize_text(query_plan.route_hints["report_type"]):
                score += 1.5
                reasons.append("report_type_match")

        return score, reasons

    def rank_candidate_reports(self, query_plan: QueryPlan) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for report in self._load_reports():
            score, reasons = self._score_report(report, query_plan)
            candidates.append(
                {
                    "report": report,
                    "score": round(score, 4),
                    "reasons": reasons,
                }
            )

        candidates.sort(
            key=lambda item: (
                item["score"],
                len(item["report"].topic_flags),
                item["report"].company_name,
            ),
            reverse=True,
        )
        return candidates

    def resolve_single_company(self, query_plan: QueryPlan) -> tuple[str, Dict[str, Any]]:
        if query_plan.mentioned_companies:
            company_name = query_plan.mentioned_companies[0]
            report = self.get_report_by_company_name(company_name)
            return company_name, {
                "route_mode": "explicit_company",
                "selected_company": company_name,
                "candidate_companies": [company_name],
                "selected_report": report.to_dict() if report else None,
                "selection_reasons": ["company_mentioned_in_question"],
            }

        ranked_candidates = self.rank_candidate_reports(query_plan)
        non_zero = [candidate for candidate in ranked_candidates if candidate["score"] > 0]
        if not non_zero:
            available = self._load_reports()
            if len(available) == 1:
                report = available[0]
                return report.company_name, {
                    "route_mode": "only_report_available",
                    "selected_company": report.company_name,
                    "candidate_companies": [report.company_name],
                    "selected_report": report.to_dict(),
                    "selection_reasons": ["only_available_report"],
                }
            raise ValueError("No company name found in the question and metadata routing found no confident candidate.")

        selected = non_zero[0]
        second_score = non_zero[1]["score"] if len(non_zero) > 1 else None
        if second_score is not None and selected["score"] <= second_score:
            raise ValueError(
                "No company name found in the question and metadata routing remained ambiguous."
            )

        report: CompanyReport = selected["report"]
        return report.company_name, {
            "route_mode": "metadata_inference",
            "selected_company": report.company_name,
            "candidate_companies": [item["report"].company_name for item in non_zero[:5]],
            "selected_report": report.to_dict(),
            "selection_reasons": selected["reasons"],
            "candidate_scores": [
                {
                    "company_name": item["report"].company_name,
                    "score": item["score"],
                    "reasons": item["reasons"],
                }
                for item in non_zero[:5]
            ],
        }
