from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


DOCUMENT_MANIFEST_COLUMNS = [
    "doc_id",
    "company_name",
    "company_aliases",
    "security_code",
    "doc_source_type",
    "report_title",
    "report_date",
    "fiscal_year",
    "broker_name",
    "major_industry",
    "language",
    "currency",
    "source_manifest",
    "source_file_path",
    "pdf_url",
]


@dataclass
class PreparedDatasetSummary:
    dataset_dir: Path
    manifest_paths: List[Path]
    documents_written: int
    skipped_rows: int
    link_mode: str


def _discover_manifest_paths(pdfcrawl_root: Path) -> List[Path]:
    if pdfcrawl_root.is_file():
        return [pdfcrawl_root]
    return sorted(path for path in pdfcrawl_root.rglob("manifest.csv") if path.is_file())


def _normalize_industry_label(raw_label: str) -> str:
    label = (raw_label or "").strip()
    if not label:
        return ""
    return re.sub(r"_\d{4}$", "", label)


def _build_company_aliases(company_name: str, security_code: str) -> str:
    aliases = []
    if company_name:
        aliases.append(company_name)
    if security_code and security_code not in aliases:
        aliases.append(security_code)
    return "|".join(aliases)


def _iter_manifest_rows(
    manifest_paths: Iterable[Path],
    *,
    currency: str,
    language: str,
) -> Iterable[Dict[str, str]]:
    for manifest_path in manifest_paths:
        source_label = manifest_path.parent.name
        inferred_industry = _normalize_industry_label(source_label)
        with manifest_path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                status = (row.get("status") or "").strip().lower()
                if status and status != "success":
                    continue

                source_file_path = Path((row.get("file_path") or "").strip())
                if not source_file_path.exists():
                    continue

                doc_id = source_file_path.stem
                security_code = (row.get("code") or "").strip()
                company_name = (row.get("company_name") or "").strip()
                industry_name = (row.get("industry_name") or "").strip() or inferred_industry

                yield {
                    "doc_id": doc_id,
                    "company_name": company_name,
                    "company_aliases": _build_company_aliases(company_name, security_code),
                    "security_code": security_code,
                    "doc_source_type": "annual_report",
                    "report_title": (row.get("title") or "").strip(),
                    "report_date": (row.get("announcement_date") or "").strip(),
                    "fiscal_year": (row.get("report_year") or "").strip(),
                    "broker_name": "",
                    "major_industry": industry_name,
                    "language": language,
                    "currency": currency,
                    "source_manifest": str(manifest_path),
                    "source_file_path": str(source_file_path),
                    "pdf_url": (row.get("pdf_url") or "").strip(),
                }


def _materialize_pdf(source_pdf_path: Path, target_pdf_path: Path, link_mode: str) -> None:
    target_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if target_pdf_path.exists() or target_pdf_path.is_symlink():
        target_pdf_path.unlink()

    if link_mode == "copy":
        shutil.copy2(source_pdf_path, target_pdf_path)
        return

    target_pdf_path.symlink_to(source_pdf_path)


def _prune_stale_pdfs(pdf_reports_dir: Path, active_doc_ids: set[str]) -> int:
    if not pdf_reports_dir.exists():
        return 0

    removed = 0
    for pdf_path in pdf_reports_dir.glob("*.pdf"):
        if pdf_path.stem in active_doc_ids:
            continue
        pdf_path.unlink()
        removed += 1
    return removed


def prepare_pdfcrawl_dataset(
    pdfcrawl_root: Path,
    dataset_dir: Path,
    *,
    link_mode: str = "symlink",
    currency: str = "CNY",
    language: str = "zh",
    write_questions_stub: bool = True,
) -> PreparedDatasetSummary:
    manifest_paths = _discover_manifest_paths(pdfcrawl_root)
    if not manifest_paths:
        raise FileNotFoundError(f"No manifest.csv files were found under {pdfcrawl_root}")

    if link_mode not in {"symlink", "copy"}:
        raise ValueError("link_mode must be either 'symlink' or 'copy'.")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    pdf_reports_dir = dataset_dir / "pdf_reports"
    document_manifest_path = dataset_dir / "document_manifest.csv"
    questions_path = dataset_dir / "questions.json"

    manifest_rows: List[Dict[str, str]] = []
    seen_doc_ids: Dict[str, str] = {}
    skipped_rows = 0

    for row in _iter_manifest_rows(manifest_paths, currency=currency, language=language):
        doc_id = row["doc_id"]
        source_file_path = row["source_file_path"]
        previous_source = seen_doc_ids.get(doc_id)
        if previous_source is not None and previous_source != source_file_path:
            raise ValueError(
                f"Duplicate doc_id '{doc_id}' detected for different PDFs: "
                f"{previous_source} vs {source_file_path}"
            )
        if previous_source is not None:
            skipped_rows += 1
            continue

        seen_doc_ids[doc_id] = source_file_path
        manifest_rows.append(row)

        target_pdf_path = pdf_reports_dir / f"{doc_id}.pdf"
        _materialize_pdf(Path(source_file_path), target_pdf_path, link_mode)

    _prune_stale_pdfs(pdf_reports_dir, set(seen_doc_ids))

    manifest_rows.sort(key=lambda item: item["doc_id"])
    with document_manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DOCUMENT_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    if write_questions_stub and not questions_path.exists():
        questions_path.write_text("[]\n", encoding="utf-8")

    summary = {
        "pdfcrawl_root": str(pdfcrawl_root),
        "manifests": [str(path) for path in manifest_paths],
        "documents": len(manifest_rows),
        "link_mode": link_mode,
        "currency": currency,
        "language": language,
    }
    (dataset_dir / "dataset_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return PreparedDatasetSummary(
        dataset_dir=dataset_dir,
        manifest_paths=manifest_paths,
        documents_written=len(manifest_rows),
        skipped_rows=skipped_rows,
        link_mode=link_mode,
    )
