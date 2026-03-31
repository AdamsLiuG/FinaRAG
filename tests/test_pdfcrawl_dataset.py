import csv
import json

from src.pdfcrawl_dataset import prepare_pdfcrawl_dataset


def test_prepare_pdfcrawl_dataset_creates_finarag_layout(tmp_path):
    pdfcrawl_root = tmp_path / "PDFCrawl" / "output"
    industry_dir = pdfcrawl_root / "semiconductor_2024"
    source_pdf_dir = industry_dir / "pdfs" / "688981" / "2024"
    source_pdf_dir.mkdir(parents=True)

    source_pdf_path = source_pdf_dir / "688981_2024_20250328.pdf"
    source_pdf_path.write_bytes(b"%PDF-1.4 dummy\n")

    manifest_path = industry_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "code",
                "company_name",
                "industry_code",
                "industry_name",
                "report_year",
                "announcement_date",
                "title",
                "bulletin_type",
                "pdf_url",
                "status",
                "file_path",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "code": "688981",
                "company_name": "中芯国际",
                "industry_code": "",
                "industry_name": "",
                "report_year": "2024",
                "announcement_date": "2025-03-28",
                "title": "中芯国际2024年年度报告",
                "bulletin_type": "年报",
                "pdf_url": "https://example.com/report.pdf",
                "status": "success",
                "file_path": str(source_pdf_path),
                "error": "",
            }
        )

    dataset_dir = tmp_path / "dataset"
    summary = prepare_pdfcrawl_dataset(pdfcrawl_root, dataset_dir, link_mode="copy")

    assert summary.documents_written == 1
    assert (dataset_dir / "pdf_reports" / "688981_2024_20250328.pdf").exists()
    assert json.loads((dataset_dir / "questions.json").read_text(encoding="utf-8")) == []

    with (dataset_dir / "document_manifest.csv").open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["doc_id"] == "688981_2024_20250328"
    assert rows[0]["company_name"] == "中芯国际"
    assert rows[0]["company_aliases"] == "中芯国际|688981"
    assert rows[0]["security_code"] == "688981"
    assert rows[0]["doc_source_type"] == "annual_report"
    assert rows[0]["major_industry"] == "semiconductor"
    assert rows[0]["language"] == "zh"
    assert rows[0]["currency"] == "CNY"


def test_prepare_pdfcrawl_dataset_prunes_stale_pdf_entries(tmp_path):
    pdfcrawl_root = tmp_path / "PDFCrawl" / "output"
    industry_dir = pdfcrawl_root / "semiconductor_2024"
    source_pdf_dir = industry_dir / "pdfs" / "688981" / "2024"
    source_pdf_dir.mkdir(parents=True)

    source_pdf_path = source_pdf_dir / "688981_2024_20250328.pdf"
    source_pdf_path.write_bytes(b"%PDF-1.4 dummy\n")

    manifest_path = industry_dir / "manifest.csv"
    fieldnames = [
        "code",
        "company_name",
        "industry_code",
        "industry_name",
        "report_year",
        "announcement_date",
        "title",
        "bulletin_type",
        "pdf_url",
        "status",
        "file_path",
        "error",
    ]

    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "code": "688981",
                "company_name": "中芯国际",
                "industry_code": "",
                "industry_name": "",
                "report_year": "2024",
                "announcement_date": "2025-03-28",
                "title": "中芯国际2024年年度报告",
                "bulletin_type": "年报",
                "pdf_url": "https://example.com/report.pdf",
                "status": "success",
                "file_path": str(source_pdf_path),
                "error": "",
            }
        )

    dataset_dir = tmp_path / "dataset"
    prepare_pdfcrawl_dataset(pdfcrawl_root, dataset_dir, link_mode="symlink")
    target_pdf_path = dataset_dir / "pdf_reports" / "688981_2024_20250328.pdf"
    assert target_pdf_path.is_symlink()

    source_pdf_path.unlink()
    prepare_pdfcrawl_dataset(pdfcrawl_root, dataset_dir, link_mode="symlink")

    assert not target_pdf_path.exists()
    assert not target_pdf_path.is_symlink()

    with (dataset_dir / "document_manifest.csv").open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows == []
