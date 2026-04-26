from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.document_store import get_document_store
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
        self._company_label_snapshots: Dict[str, Dict[str, Any]] | None = None
        self._company_label_evidence: Dict[str, List[Dict[str, Any]]] | None = None

    def _metadata_store_dir(self) -> Path | None:
        if self.documents_dir is not None:
            candidate = self.documents_dir.parent.parent / "metadata_store"
            if candidate.exists():
                return candidate
        if self.subset_path is not None:
            candidate = self.subset_path.parent / "metadata_store"
            if candidate.exists():
                return candidate
        return None

    def _load_company_label_snapshots(self) -> Dict[str, Dict[str, Any]]:
        if self._company_label_snapshots is not None:
            return self._company_label_snapshots

        snapshots: Dict[str, Dict[str, Any]] = {}
        metadata_store_dir = self._metadata_store_dir()
        if metadata_store_dir is None:
            self._company_label_snapshots = snapshots
            return snapshots

        snapshot_path = metadata_store_dir / "company_label_snapshot.jsonl"
        if not snapshot_path.exists():
            self._company_label_snapshots = snapshots
            return snapshots

        with open(snapshot_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                report_id = payload.get("report_id")
                if report_id:
                    snapshots[str(report_id)] = payload

        self._company_label_snapshots = snapshots
        return snapshots

    def _load_company_label_evidence(self) -> Dict[str, List[Dict[str, Any]]]:
        if self._company_label_evidence is not None:
            return self._company_label_evidence

        evidence_by_report: Dict[str, List[Dict[str, Any]]] = {}
        metadata_store_dir = self._metadata_store_dir()
        if metadata_store_dir is None:
            self._company_label_evidence = evidence_by_report
            return evidence_by_report

        evidence_path = metadata_store_dir / "company_label_evidence.jsonl"
        if not evidence_path.exists():
            self._company_label_evidence = evidence_by_report
            return evidence_by_report

        with open(evidence_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                report_id = payload.get("report_id") or payload.get("doc_id")
                if report_id:
                    evidence_by_report.setdefault(str(report_id), []).append(payload)

        self._company_label_evidence = evidence_by_report
        return evidence_by_report

    def _load_document_meta(self) -> Dict[str, Dict[str, Any]]:
        if self.documents_dir is None or not self.documents_dir.exists():
            return {}

        store = get_document_store(self.documents_dir)
        return {doc_id: dict(metainfo) for doc_id, metainfo in store.metainfo_by_doc_id.items()}

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

    def get_report_by_doc_id(self, doc_id: str) -> CompanyReport | None:
        doc_id = str(doc_id or "")
        for report in self._load_reports():
            if report.sha1 == doc_id:
                return report
        return None

    def get_company_label_evidence(
        self,
        report_id: str,
        *,
        label_field: Optional[str] = None,
        labels: Optional[List[str]] = None,
        literal_only: bool = False,
    ) -> List[Dict[str, Any]]:
        rows = list(self._load_company_label_evidence().get(str(report_id), []))
        if label_field:
            rows = [row for row in rows if row.get("label_field") == label_field]
        if labels:
            expected = {normalize_text(str(label)) for label in labels if label not in (None, "")}
            rows = [row for row in rows if normalize_text(str(row.get("label"))) in expected]
        if literal_only:
            rows = [row for row in rows if row.get("has_literal_evidence")]
        return rows

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

    @staticmethod
    def _matches_text_filter(expected: Optional[str], observed: Optional[str]) -> bool:
        if not expected or not observed:
            return True
        expected_norm = normalize_text(expected)
        observed_norm = normalize_text(observed)
        return expected_norm in observed_norm or observed_norm in expected_norm

    @classmethod
    def _matches_list_filter(cls, expected: Optional[List[str]], observed: Optional[List[str]]) -> bool:
        if not expected:
            return True
        observed_norm = {normalize_text(str(item)) for item in observed or [] if item not in (None, "")}
        if not observed_norm:
            return False
        for item in expected:
            normalized = normalize_text(str(item))
            if not normalized:
                continue
            if not any(normalized in candidate or candidate in normalized for candidate in observed_norm):
                return False
        return True

    def _report_filter_metadata(self, report: CompanyReport) -> Dict[str, Any]:
        metadata = dict(report.metadata)
        metadata.update(self._load_company_label_snapshots().get(report.sha1, {}))
        return metadata

    def _matches_query_filters(self, report: CompanyReport, query_plan: QueryPlan) -> tuple[bool, List[str]]:
        filters = query_plan.filters
        metadata = self._report_filter_metadata(report)
        reasons: List[str] = []

        if filters.doc_source_type and metadata.get("doc_source_type") and metadata.get("doc_source_type") != filters.doc_source_type:
            return False, reasons
        if filters.year is not None:
            observed_year = metadata.get("report_year") or metadata.get("fiscal_year")
            try:
                observed_year = int(observed_year)
            except (TypeError, ValueError):
                observed_year = None
            if observed_year is not None and observed_year != filters.year:
                return False, reasons
            if observed_year == filters.year:
                reasons.append("year_filter_match")
        if filters.exchange and not self._matches_text_filter(filters.exchange, metadata.get("exchange")):
            return False, reasons
        if filters.board and not self._matches_text_filter(filters.board, metadata.get("board")):
            return False, reasons
        if filters.market_type and not self._matches_text_filter(filters.market_type, metadata.get("market_type")):
            return False, reasons
        if filters.industry_l1 and not self._matches_text_filter(filters.industry_l1, metadata.get("industry_l1")):
            return False, reasons
        if filters.industry_l2 and not self._matches_text_filter(filters.industry_l2, metadata.get("industry_l2")):
            return False, reasons
        if filters.chain_position_major and not self._matches_text_filter(filters.chain_position_major, metadata.get("chain_position_major")):
            return False, reasons
        if not self._matches_list_filter(filters.business_tags, metadata.get("business_tags")):
            return False, reasons
        if not self._matches_list_filter(filters.strategy_tags, metadata.get("strategy_tags")):
            return False, reasons
        if not self._matches_list_filter(filters.factor_tags, metadata.get("factor_tags")):
            return False, reasons
        if not self._matches_list_filter(filters.chain_position_minor, metadata.get("chain_position_minor")):
            return False, reasons
        if not self._matches_list_filter(filters.listing_tags, metadata.get("listing_tags")):
            return False, reasons
        if not self._matches_list_filter(filters.ownership_tags, metadata.get("ownership_tags")):
            return False, reasons
        if not self._matches_list_filter(filters.status_tags, metadata.get("status_tags")):
            return False, reasons
        if not self._matches_list_filter(filters.style_tags, metadata.get("style_tags")):
            return False, reasons

        for label, value in (
            ("exchange", filters.exchange),
            ("board", filters.board),
            ("market_type", filters.market_type),
            ("industry_l1", filters.industry_l1),
            ("industry_l2", filters.industry_l2),
            ("chain_position_major", filters.chain_position_major),
        ):
            if value:
                reasons.append(f"{label}_filter_match")

        for label, values in (
            ("business_tags", filters.business_tags),
            ("strategy_tags", filters.strategy_tags),
            ("factor_tags", filters.factor_tags),
            ("chain_position_minor", filters.chain_position_minor),
            ("listing_tags", filters.listing_tags),
            ("ownership_tags", filters.ownership_tags),
            ("status_tags", filters.status_tags),
            ("style_tags", filters.style_tags),
        ):
            for value in values or []:
                reasons.append(f"{label}:{value}")

        if filters.doc_source_type:
            reasons.append("doc_source_type_filter_match")

        return True, reasons

    def report_matches_query_filters(self, report: CompanyReport, query_plan: QueryPlan) -> tuple[bool, List[str]]:
        return self._matches_query_filters(report, query_plan)

    def get_report_filter_metadata(self, report: CompanyReport) -> Dict[str, Any]:
        return self._report_filter_metadata(report)

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

    def resolve_candidate_reports(self, query_plan: QueryPlan, limit: Optional[int] = None) -> Dict[str, Any]:
        matched_candidates: List[Dict[str, Any]] = []
        for report in self._load_reports():
            matched, reasons = self._matches_query_filters(report, query_plan)
            if not matched:
                continue
            matched_candidates.append(
                {
                    "report": report,
                    "reasons": reasons,
                }
            )

        matched_candidates.sort(
            key=lambda item: (
                item["report"].report_year or -1,
                item["report"].company_name,
                item["report"].sha1,
            ),
            reverse=True,
        )

        if limit is not None:
            matched_candidates = matched_candidates[: max(1, limit)]

        candidate_companies: List[str] = []
        candidate_doc_ids: List[str] = []
        for item in matched_candidates:
            report = item["report"]
            candidate_doc_ids.append(report.sha1)
            if report.company_name and report.company_name not in candidate_companies:
                candidate_companies.append(report.company_name)

        return {
            "route_mode": "document_catalog_multi",
            "selected_company": None,
            "candidate_companies": candidate_companies,
            "candidate_doc_ids": candidate_doc_ids,
            "selected_report": None,
            "selection_reasons": sorted({reason for item in matched_candidates for reason in item["reasons"]}),
            "candidate_scores": [
                {
                    "company_name": item["report"].company_name,
                    "doc_id": item["report"].sha1,
                    "doc_source_type": item["report"].doc_source_type,
                    "reasons": item["reasons"],
                }
                for item in matched_candidates
            ],
        }
