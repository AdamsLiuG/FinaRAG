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
    assert rows[0]["exchange"] == ""
    assert rows[0]["industry_l1"] == "semiconductor"


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


def test_prepare_pdfcrawl_dataset_enriches_manifest_and_metadata_store(tmp_path):
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
                "industry_code": "C39",
                "industry_name": "计算机、通信和其他电子设备制造业",
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

    (industry_dir / "metadata.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk_id": "688981_2024_20250328_p0001_b01",
                        "stock_code": "688981",
                        "company_name": "中芯国际",
                        "report_id": "688981_2024_20250328",
                        "report_year": 2024,
                        "report_type": "annual_report",
                        "exchange": "上海证券交易所",
                        "board": "科创板",
                        "market_type": "A股",
                        "industry_code_raw": "C39",
                        "industry_name_raw": "计算机、通信和其他电子设备制造业",
                        "industry_l1": "半导体",
                        "industry_l2": "晶圆代工",
                        "business_tags": ["晶圆制造"],
                        "strategy_tags": ["国产替代"],
                        "factor_tags": ["高资本开支"],
                        "chain_position_major": "中游制造",
                        "chain_position_minor": ["晶圆代工"],
                        "listing_tags": ["A股", "科创板"],
                        "ownership_tags": ["公众公司"],
                        "status_tags": ["龙头"],
                        "style_tags": ["硬科技"],
                        "page_start": 1,
                        "page_end": 1,
                        "section_name": "（一）主营业务分析",
                        "section_l1": "第三节 管理层讨论与分析",
                        "section_l2": "一、经营情况讨论与分析",
                        "section_l3": "（一）主营业务分析",
                        "section_path": "第三节 管理层讨论与分析 > 一、经营情况讨论与分析 > （一）主营业务分析",
                        "section_leaf": "（一）主营业务分析",
                        "local_heading": "（一）主营业务分析",
                        "block_index": 1,
                        "chunk_kind": "text",
                        "page_role": "content",
                        "file_path": str(source_pdf_path),
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "chunk_id": "688981_2024_20250328_p0001_b02",
                        "stock_code": "688981",
                        "company_name": "中芯国际",
                        "report_id": "688981_2024_20250328",
                        "report_year": 2024,
                        "report_type": "annual_report",
                        "exchange": "上海证券交易所",
                        "board": "科创板",
                        "market_type": "A股",
                        "industry_code_raw": "C39",
                        "industry_name_raw": "计算机、通信和其他电子设备制造业",
                        "industry_l1": "半导体",
                        "industry_l2": "晶圆代工",
                        "business_tags": ["晶圆制造"],
                        "strategy_tags": ["国产替代"],
                        "factor_tags": ["高资本开支"],
                        "chain_position_major": "中游制造",
                        "chain_position_minor": ["晶圆代工"],
                        "listing_tags": ["A股", "科创板"],
                        "ownership_tags": ["公众公司"],
                        "status_tags": ["龙头"],
                        "style_tags": ["硬科技"],
                        "page_start": 1,
                        "page_end": 1,
                        "section_name": "1. 产品结构",
                        "section_l1": "第三节 管理层讨论与分析",
                        "section_l2": "一、经营情况讨论与分析",
                        "section_l3": "（一）主营业务分析",
                        "section_path": "第三节 管理层讨论与分析 > 一、经营情况讨论与分析 > （一）主营业务分析 > 1. 产品结构",
                        "section_leaf": "1. 产品结构",
                        "local_heading": "1. 产品结构",
                        "block_index": 2,
                        "chunk_kind": "text",
                        "page_role": "content",
                        "file_path": str(source_pdf_path),
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (industry_dir / "chunks.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk_id": "688981_2024_20250328_p0001_b01",
                        "stock_code": "688981",
                        "company_name": "中芯国际",
                        "report_id": "688981_2024_20250328",
                        "report_year": 2024,
                        "page_start": 1,
                        "page_end": 1,
                        "section_name": "（一）主营业务分析",
                        "section_l1": "第三节 管理层讨论与分析",
                        "section_l2": "一、经营情况讨论与分析",
                        "section_l3": "（一）主营业务分析",
                        "section_path": "第三节 管理层讨论与分析 > 一、经营情况讨论与分析 > （一）主营业务分析",
                        "section_leaf": "（一）主营业务分析",
                        "local_heading": "（一）主营业务分析",
                        "block_index": 1,
                        "chunk_kind": "text",
                        "chunk_text": "公司坚持先进工艺研发。",
                        "file_path": str(source_pdf_path),
                        "embedding_text": "章节路径：第三节 管理层讨论与分析 > 一、经营情况讨论与分析 > （一）主营业务分析\n正文：公司坚持先进工艺研发。",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "chunk_id": "688981_2024_20250328_p0001_b02",
                        "stock_code": "688981",
                        "company_name": "中芯国际",
                        "report_id": "688981_2024_20250328",
                        "report_year": 2024,
                        "page_start": 1,
                        "page_end": 1,
                        "section_name": "1. 产品结构",
                        "section_l1": "第三节 管理层讨论与分析",
                        "section_l2": "一、经营情况讨论与分析",
                        "section_l3": "（一）主营业务分析",
                        "section_path": "第三节 管理层讨论与分析 > 一、经营情况讨论与分析 > （一）主营业务分析 > 1. 产品结构",
                        "section_leaf": "1. 产品结构",
                        "local_heading": "1. 产品结构",
                        "block_index": 2,
                        "chunk_kind": "text",
                        "chunk_text": "产品结构持续优化。",
                        "file_path": str(source_pdf_path),
                        "embedding_text": "章节路径：第三节 管理层讨论与分析 > 一、经营情况讨论与分析 > （一）主营业务分析 > 1. 产品结构\n正文：产品结构持续优化。",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (industry_dir / "child_chunks.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk_id": "688981_2024_20250328_p0001_b01_c01",
                        "parent_chunk_id": "688981_2024_20250328_p0001_b01",
                        "stock_code": "688981",
                        "company_name": "中芯国际",
                        "report_id": "688981_2024_20250328",
                        "report_year": 2024,
                        "page_start": 1,
                        "page_end": 1,
                        "section_name": "（一）主营业务分析",
                        "section_l1": "第三节 管理层讨论与分析",
                        "section_l2": "一、经营情况讨论与分析",
                        "section_l3": "（一）主营业务分析",
                        "section_path": "第三节 管理层讨论与分析 > 一、经营情况讨论与分析 > （一）主营业务分析",
                        "section_leaf": "（一）主营业务分析",
                        "chunk_text": "公司坚持先进工艺研发。",
                        "file_path": str(source_pdf_path),
                        "embedding_text": "正文：公司坚持先进工艺研发。",
                    },
                    ensure_ascii=False,
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (industry_dir / "company_profiles.jsonl").write_text(
        json.dumps(
            {
                "stock_code": "688981",
                "company_name": "中芯国际",
                "exchange_raw": "上海证券交易所",
                "board_raw": "科创板",
                "main_business_raw": "晶圆代工",
                "related_securities_raw": ["SMIC"],
                "dividend_history_raw": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    dataset_dir = tmp_path / "dataset"
    prepare_pdfcrawl_dataset(pdfcrawl_root, dataset_dir, link_mode="copy", metadata_mode="required")

    with (dataset_dir / "document_manifest.csv").open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["exchange"] == "上海证券交易所"
    assert rows[0]["board"] == "科创板"
    assert rows[0]["market_type"] == "A股"
    assert rows[0]["industry_l1"] == "半导体"
    assert rows[0]["industry_l2"] == "晶圆代工"
    assert rows[0]["has_pdfcrawl_metadata"] == "true"

    metadata_store_dir = dataset_dir / "metadata_store"
    report_page_rows = [
        json.loads(line)
        for line in (metadata_store_dir / "report_page.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    company_master_rows = [
        json.loads(line)
        for line in (metadata_store_dir / "company_master.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunk_metadata_rows = [
        json.loads(line)
        for line in (metadata_store_dir / "chunk_metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(report_page_rows) == 1
    assert report_page_rows[0]["section_name"] == "（一）主营业务分析"
    assert report_page_rows[0]["page"] == 1
    assert (
        report_page_rows[0]["section_path"]
        == "第三节 管理层讨论与分析 > 一、经营情况讨论与分析 > （一）主营业务分析"
    )
    assert company_master_rows[0]["stock_code"] == "688981"
    assert company_master_rows[0]["main_business_raw"] == "晶圆代工"
    assert len(chunk_metadata_rows) == 3
    assert {row["node_type"] for row in chunk_metadata_rows} == {"parent", "child"}
    assert any(row["section_path"].endswith("1. 产品结构") for row in chunk_metadata_rows if row["node_type"] == "parent")
    assert any(row["parent_chunk_id"] == "688981_2024_20250328_p0001_b01" for row in chunk_metadata_rows if row["node_type"] == "child")


def test_prepare_pdfcrawl_dataset_requires_metadata_when_requested(tmp_path):
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
    try:
        prepare_pdfcrawl_dataset(pdfcrawl_root, dataset_dir, link_mode="copy", metadata_mode="required")
    except FileNotFoundError as exc:
        assert "metadata.jsonl" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError when metadata_mode='required' and metadata.jsonl is missing.")
