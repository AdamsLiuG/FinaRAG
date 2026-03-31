from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.document_manifest import load_document_manifest
from src.query_plan import QueryPlan
from src.text_normalization import extract_security_codes, normalize_text


@dataclass
class CompanyReport:
    sha1: str
    company_name: str
    company_aliases: List[str]
    currency: Optional[str]
    major_industry: Optional[str]
    report_year: Optional[int]
    report_type: Optional[str]
    doc_source_type: Optional[str]
    broker_name: Optional[str]
    security_code: Optional[str]
    report_title: Optional[str]
    language: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sha1": self.sha1,
            "company_name": self.company_name,
            "company_aliases": self.company_aliases,
            "currency": self.currency,
            "major_industry": self.major_industry,
            "report_year": self.report_year,
            "report_type": self.report_type,
            "doc_source_type": self.doc_source_type,
            "broker_name": self.broker_name,
            "security_code": self.security_code,
            "report_title": self.report_title,
            "language": self.language,
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
            sha1 = metainfo.get("sha1_name") or metainfo.get("doc_id")
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

        manifest_rows = load_document_manifest(self.subset_path)
        document_meta = self._load_document_meta()
        reports: List[CompanyReport] = []

        for doc_id, row in manifest_rows.items():
            document_entry = document_meta.get(doc_id, {})
            metadata = {key: value for key, value in row.items() if value is not None}
            metadata.update({key: value for key, value in document_entry.items() if value is not None})
            company_name = str(metadata.get("company_name") or "").strip('" ')
            aliases = list(metadata.get("company_aliases") or ([] if not company_name else [company_name]))
            if company_name and company_name not in aliases:
                aliases.insert(0, company_name)

            report_year = metadata.get("report_year") or metadata.get("fiscal_year")
            if isinstance(report_year, str) and report_year.isdigit():
                report_year = int(report_year)

            reports.append(
                CompanyReport(
                    sha1=str(doc_id),
                    company_name=company_name,
                    company_aliases=aliases,
                    currency=metadata.get("currency"),
                    major_industry=metadata.get("major_industry"),
                    report_year=report_year if isinstance(report_year, int) else None,
                    report_type=str(metadata.get("report_type")).strip() if metadata.get("report_type") else None,
                    doc_source_type=metadata.get("doc_source_type"),
                    broker_name=metadata.get("broker_name"),
                    security_code=str(metadata.get("security_code")).strip() if metadata.get("security_code") else None,
                    report_title=metadata.get("report_title"),
                    language=str(metadata.get("language") or "en"),
                    metadata=metadata,
                )
            )

        self._reports = reports
        return reports

    def get_reports(self) -> List[CompanyReport]:
        return list(self._load_reports())

    def get_company_names(self) -> List[str]:
        return sorted({report.company_name for report in self._load_reports() if report.company_name})

    def get_report_by_company_name(self, company_name: str) -> CompanyReport | None:
        for report in self._load_reports():
            if report.company_name == company_name:
                return report
        return None

    def extract_companies_from_question(self, question_text: str) -> List[str]:
        normalized_question = normalize_text(question_text)
        found_companies: List[str] = []
        seen = set()
        for report in self._load_reports():
            aliases = sorted(set(report.company_aliases), key=len, reverse=True)
            if report.security_code:
                aliases.append(report.security_code)
            if any(alias and normalize_text(alias) in normalized_question for alias in aliases):
                if report.company_name and report.company_name not in seen:
                    seen.add(report.company_name)
                    found_companies.append(report.company_name)
        return found_companies

    def _score_report(self, report: CompanyReport, query_plan: QueryPlan) -> tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []
        normalized_question = normalize_text(query_plan.original_query)
        security_codes = query_plan.route_hints.get("security_codes") or extract_security_codes(query_plan.original_query)

        if report.company_name and normalize_text(report.company_name) in normalized_question:
            score += 6.0
            reasons.append("company_name_match")

        matched_aliases = [
            alias
            for alias in report.company_aliases
            if alias and normalize_text(alias) in normalized_question and normalize_text(alias) != normalize_text(report.company_name)
        ]
        if matched_aliases:
            score += 5.0
            reasons.append(f"company_alias:{matched_aliases[0]}")

        if report.security_code and any(str(report.security_code) == str(code) for code in security_codes):
            score += 5.5
            reasons.append("security_code_match")

        if query_plan.filters.doc_source_type and report.doc_source_type == query_plan.filters.doc_source_type:
            score += 2.5
            reasons.append("doc_source_type_match")

        if query_plan.filters.report_type and report.report_type:
            if normalize_text(report.report_type) == normalize_text(query_plan.filters.report_type):
                score += 1.5
                reasons.append("report_type_match")

        if query_plan.filters.year is not None and report.report_year == query_plan.filters.year:
            score += 2.0
            reasons.append("year_match")

        if query_plan.filters.currency and report.currency and report.currency == query_plan.filters.currency:
            score += 1.0
            reasons.append("currency_match")

        if report.major_industry and normalize_text(report.major_industry) in normalized_question:
            score += 1.0
            reasons.append("industry_match")

        if report.broker_name and normalize_text(report.broker_name) in normalized_question:
            score += 1.5
            reasons.append("broker_match")

        if report.report_title and normalize_text(report.report_title) in normalized_question:
            score += 2.0
            reasons.append("report_title_match")

        if report.language == "zh":
            score += 0.2

        return score, reasons

    def rank_candidate_reports(self, query_plan: QueryPlan, limit: int = 5) -> List[Dict[str, Any]]:
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
                item["report"].report_year or -1,
                item["report"].company_name,
            ),
            reverse=True,
        )
        return candidates[:limit]

    def resolve_single_company(self, query_plan: QueryPlan, limit: int = 5) -> tuple[str, Dict[str, Any]]:
        ranked_candidates = self.rank_candidate_reports(query_plan, limit=max(limit, 5))
        non_zero = [candidate for candidate in ranked_candidates if candidate["score"] > 0]
        if not non_zero:
            available = self._load_reports()
            if len(available) == 1:
                report = available[0]
                return report.company_name, {
                    "route_mode": "only_report_available",
                    "selected_company": report.company_name,
                    "candidate_companies": [report.company_name],
                    "candidate_doc_ids": [report.sha1],
                    "selected_report": report.to_dict(),
                    "selection_reasons": ["only_available_report"],
                }
            raise ValueError("No company name found in the question and document catalog routing found no confident candidate.")

        selected = non_zero[0]
        report: CompanyReport = selected["report"]
        candidate_doc_ids = [item["report"].sha1 for item in non_zero[:limit]]
        candidate_companies = []
        for item in non_zero[:limit]:
            company_name = item["report"].company_name
            if company_name and company_name not in candidate_companies:
                candidate_companies.append(company_name)

        return report.company_name, {
            "route_mode": "document_catalog",
            "selected_company": report.company_name,
            "candidate_companies": candidate_companies,
            "candidate_doc_ids": candidate_doc_ids,
            "selected_report": report.to_dict(),
            "selection_reasons": selected["reasons"],
            "candidate_scores": [
                {
                    "company_name": item["report"].company_name,
                    "doc_id": item["report"].sha1,
                    "doc_source_type": item["report"].doc_source_type,
                    "score": item["score"],
                    "reasons": item["reasons"],
                }
                for item in non_zero[:limit]
            ],
        }
