from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any, Dict, Iterable, List, Optional

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.dataset_schema import (
    FinanceEvalManifest,
    FinanceEvalQuestionSet,
    FinanceGoldAnswerSet,
    load_gold_answer_set,
    load_question_set,
    validate_dataset_alignment,
)
from src.text_normalization import parse_numeric_value


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "data" / "top10_industries_2024_20each"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "finance_eval_benchmark_v2" / "number_holdout_metrics80"
DEFAULT_TRAIN_FILE = ROOT / "training" / "reranker_distill" / "processed" / "qwen3_reranker_sft_train.jsonl"
RAW_QUERY_FILES = (
    ROOT / "training" / "reranker_distill" / "raw" / "candidate_pool.jsonl",
    ROOT / "training" / "reranker_distill" / "raw" / "teacher_answers_with_auto_evidence.jsonl",
)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _compact_text(text: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = normalized.replace("％", "%")
    return re.sub(r"\s+", "", normalized)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _question_blob(records: Iterable[Dict[str, Any]]) -> str:
    pieces: List[str] = []
    for record in records:
        for key in ("query", "question_text", "question"):
            value = record.get(key)
            if value:
                pieces.append(str(value))
    return "\n".join(pieces)


def _training_query_blob(train_file: Path) -> str:
    return _question_blob(_load_jsonl(train_file))


def _raw_query_blob(raw_files: Iterable[Path]) -> str:
    records: List[Dict[str, Any]] = []
    for path in raw_files:
        records.extend(_load_jsonl(path))
    return _question_blob(records)


def _company_in_blob(row: Dict[str, Any], blob: str) -> bool:
    return any(
        token and token in blob
        for token in (
            row.get("company_name"),
            row.get("security_code"),
        )
    )


def _page_text(document: Dict[str, Any], page: Any) -> str:
    try:
        page_number = int(page)
    except (TypeError, ValueError):
        return ""
    for page_payload in (document.get("content") or {}).get("pages") or []:
        try:
            observed = int(page_payload.get("page"))
        except (TypeError, ValueError):
            continue
        if observed == page_number:
            return str(page_payload.get("text") or "")
    return ""


def _table_window(document: Dict[str, Any], table: Dict[str, Any]) -> str:
    markdown = str(table.get("markdown") or "")
    page_text = _page_text(document, table.get("page"))
    if not page_text:
        return markdown

    marker = ""
    for line in markdown.splitlines():
        line = line.strip()
        if line and not set(line) <= {"|", "-", ":", " "}:
            marker = line[:80]
            break

    index = page_text.find(marker) if marker else -1
    if index < 0:
        first_cell = next((cell.get("raw_value") for cell in table.get("cell_records") or [] if cell.get("raw_value")), None)
        index = page_text.find(str(first_cell)) if first_cell else -1

    if index >= 0:
        return page_text[max(0, index - 500): index + min(len(markdown), 1200)]
    return f"{page_text[:1200]}\n{markdown}"


def _infer_unit(raw_unit_hint: Any, table_window: str, metric_key: str) -> Optional[str]:
    compact = _compact_text(table_window[:1800])
    unit_pattern = (
        r"(?:金额)?单位[:：为]*"
        r"(?:人民币)?"
        r"(百万元|千万元|万元|千元|亿元|元)"
    )
    unit_match = re.search(unit_pattern, compact)
    if unit_match:
        return unit_match.group(1)

    raw_hint = str(raw_unit_hint or "").strip()
    if raw_hint and "%" not in raw_hint and "％" not in raw_hint:
        return raw_hint

    for unit in (
        "人民币百万元",
        "百万元",
        "人民币千万元",
        "千万元",
        "人民币万元",
        "万元人民币",
        "单位:万元",
        "万元",
        "人民币千元",
        "单位:千元",
        "千元",
        "亿元",
        "人民币元",
        "单位:元",
        "元",
    ):
        if unit in compact:
            return unit.replace("单位:", "")

    # Annual-report R&D tables often have bad '%' cell hints because the same
    # table contains ratio rows. In the absence of a nearby unit declaration,
    # treat the amount rows as already denominated in yuan.
    if metric_key == "rd_investment":
        return None
    return None


def _is_numeric_raw(raw_value: Any) -> bool:
    text = str(raw_value or "").strip()
    if not text or not any(char.isdigit() for char in text):
        return False
    if len(text) > 80 or "%" in text or "％" in text or "百分点" in text:
        return False
    return parse_numeric_value(text) is not None


def _bad_column(col_text: str, metric_key: str) -> bool:
    compact = _compact_text(col_text)
    if any(term in compact for term in ("2023", "2022", "2021", "上年", "上期", "上年度", "上期期末")):
        return True
    if any(term in compact for term in ("增减", "变化", "比例", "比上年", "同比", "%", "百分点")):
        return True
    return False


def _column_score(col_text: str, metric_key: str) -> Optional[float]:
    if _bad_column(col_text, metric_key):
        return None
    compact = _compact_text(col_text)
    if "2024" in compact:
        return 5.0
    if metric_key == "total_assets" and any(term in compact for term in ("期末", "本年末", "年末", "期末余额")):
        return 4.0
    if any(term in compact for term in ("本期", "本年度", "本年", "本报告期", "合计", "本集团", "本公司")):
        return 3.0
    if not compact:
        return 1.0
    return None


@dataclass(frozen=True)
class MetricSpec:
    key: str
    metric_name: str
    question_metric: str
    row_terms: tuple[str, ...]
    fallback_row_terms: tuple[str, ...] = ()
    bad_row_terms: tuple[str, ...] = ()
    context_terms: tuple[str, ...] = ()
    expected_section: str | None = None


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="net_profit",
        metric_name="归属于上市公司股东的净利润",
        question_metric="归属于上市公司股东的净利润",
        row_terms=("归属于上市公司股东的净利润", "归属于母公司股东的净利润"),
        fallback_row_terms=("净利润",),
        bad_row_terms=("扣除非经常性", "少数股东", "每股", "率", "现金流量"),
        context_terms=("主要会计数据", "主要财务指标", "公司简介和主要财务指标", "利润表"),
        expected_section="主要会计数据",
    ),
    MetricSpec(
        key="rd_investment",
        metric_name="研发投入合计",
        question_metric="研发投入合计",
        row_terms=("研发投入合计", "研发投入总额"),
        bad_row_terms=("占营业收入", "比例", "资本化的比重", "人员", "人数", "变化幅度"),
        context_terms=("研发投入", "研发费用"),
        expected_section="研发投入情况表",
    ),
    MetricSpec(
        key="total_assets",
        metric_name="资产总额",
        question_metric="资产总额",
        row_terms=("总资产", "资产总额", "资产总计"),
        bad_row_terms=("负债", "净资产", "收益率", "每股", "增减", "同比"),
        context_terms=("主要会计数据", "主要财务指标", "公司简介和主要财务指标", "资产负债表"),
        expected_section="主要会计数据",
    ),
    MetricSpec(
        key="operating_cash_flow",
        metric_name="经营活动产生的现金流量净额",
        question_metric="经营活动产生的现金流量净额",
        row_terms=("经营活动产生的现金流量净额", "经营活动现金流量净额"),
        fallback_row_terms=("经营性现金净流量", "经营现金流量净额"),
        bad_row_terms=("每股", "比率", "占比", "收益率"),
        context_terms=("主要会计数据", "主要财务指标", "公司简介和主要财务指标", "现金流量表"),
        expected_section="主要会计数据",
    ),
)


def _row_score(row_text: str, spec: MetricSpec) -> Optional[float]:
    compact = _compact_text(row_text)
    if any(term in compact for term in spec.bad_row_terms):
        return None
    if any(term in compact for term in spec.row_terms):
        return 20.0
    if any(term in compact for term in spec.fallback_row_terms):
        return 10.0
    return None


def _context_score(table_window: str, spec: MetricSpec) -> float:
    compact = _compact_text(table_window)
    score = 0.0
    if any(term in compact for term in spec.context_terms):
        score += 8.0
    if "母公司" in compact and spec.key != "rd_investment":
        score -= 4.0
    return score


def _extract_metric(document: Dict[str, Any], spec: MetricSpec) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for table in (document.get("content") or {}).get("structured_tables") or []:
        window = _table_window(document, table)
        for cell in table.get("cell_records") or []:
            raw_value = cell.get("raw_value")
            if not _is_numeric_raw(raw_value):
                continue
            row_text = " ".join(str(item) for item in (cell.get("matched_row_headers") or []))
            col_text = " ".join(str(item) for item in (cell.get("matched_col_headers") or []))
            row_score = _row_score(row_text, spec)
            if row_score is None:
                continue
            col_score = _column_score(col_text, spec.key)
            if col_score is None:
                continue
            unit = _infer_unit(cell.get("unit_hint"), window, spec.key)
            value = parse_numeric_value(str(raw_value), unit_hint=unit)
            if value is None:
                continue

            score = row_score + col_score + _context_score(window, spec)
            try:
                page = int(cell.get("page") or table.get("page"))
            except (TypeError, ValueError):
                page = None
            if page is not None:
                if spec.key != "rd_investment" and page <= 15:
                    score += 4.0
                if spec.key == "rd_investment" and page <= 80:
                    score += 2.0

            match = {
                "metric_key": spec.key,
                "metric_name": spec.metric_name,
                "value": value,
                "raw_value": raw_value,
                "unit": unit or "元",
                "page": page,
                "table_id": table.get("table_id"),
                "row_idx": cell.get("row_idx"),
                "col_idx": cell.get("col_idx"),
                "row_headers": cell.get("matched_row_headers") or [],
                "col_headers": cell.get("matched_col_headers") or [],
                "match_score": round(score, 4),
                "table_snippet": str(table.get("markdown") or "")[:700],
            }
            if best is None or match["match_score"] > best["match_score"]:
                best = match
    return best


def _load_document(documents_dir: Path, doc_id: str) -> Optional[Dict[str, Any]]:
    path = documents_dir / f"{doc_id}.json"
    if not path.exists():
        return None
    return _read_json(path)


def _build_question_answer(
    *,
    dataset_name: str,
    row: Dict[str, Any],
    match: Dict[str, Any],
    spec: MetricSpec,
    company_index: int,
    metric_index: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    qid = f"holdout-number-{company_index:02d}-{metric_index:02d}-{spec.key}"
    company = row["company_name"]
    doc_id = row["doc_id"]
    page = int(match["page"])
    text = f"{company}2024年年报中的{spec.question_metric}是多少元？"
    metadata = {
        "source_dataset": dataset_name,
        "source_record_type": "question",
        "quality_tier": "auto_extracted_gold",
        "review_required": True,
        "scenario_family": "number_metric_company_holdout",
        "generation_method": "structured_table_extraction",
        "holdout_policy": "exclude companies appearing in training/reranker_distill/processed/qwen3_reranker_sft_train.jsonl query text",
        "metric_key": spec.key,
        "source_table_id": match.get("table_id"),
        "source_row_idx": match.get("row_idx"),
        "source_col_idx": match.get("col_idx"),
        "source_row_headers": match.get("row_headers"),
        "source_col_headers": match.get("col_headers"),
        "source_raw_value": match.get("raw_value"),
        "source_unit": match.get("unit"),
        "match_score": match.get("match_score"),
    }
    common = {
        "kind": "number",
        "capability": "single_doc_fact",
        "difficulty": "medium",
        "doc_ids": [doc_id],
        "company_name": company,
        "stock_code": row.get("security_code"),
        "report_year": 2024,
        "report_type": "annual_report",
        "period": "本期",
        "metric_name": spec.metric_name,
        "currency": "CNY",
        "unit": "元",
        "evidence_type": "table",
        "group_id": f"holdout-industry-{company_index:02d}",
        "group_name": row.get("major_industry") or row.get("industry_l1"),
        "should_refuse": False,
        "annotation_status": "auto_extracted",
        "notes": f"自动从结构化表格抽取；复核 {spec.metric_name}、2024 期间、单位和页码。",
        "metadata": metadata,
    }
    question = {
        "id": qid,
        "text": text,
        "mentioned_companies": [],
        "industry_l1": row.get("industry_l1"),
        "gold_value": match["value"],
        "gold_pages": [page],
        "gold_chunk_ids": [],
        "expected_filters": {"company_name": company, "metric_name": spec.metric_name},
        **common,
    }
    answer = {
        "question_id": qid,
        "question_text": text,
        "value": match["value"],
        "gold_pages": [page],
        "gold_chunk_ids": [],
        "references": [
            {
                "pdf_sha1": doc_id,
                "page_index": page - 1,
                "chunk_id": None,
                "section_name": spec.expected_section,
                "evidence_type": "table",
            }
        ],
        **common,
    }
    answer["metadata"] = dict(metadata, source_record_type="answer")
    return question, answer


def _write_review_checklist(path: Path, selected: List[Dict[str, Any]], questions: List[Dict[str, Any]]) -> None:
    question_by_company_metric = {
        (question["company_name"], question["metadata"]["metric_key"]): question
        for question in questions
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "company_name",
                "doc_id",
                "metric_key",
                "question_id",
                "question_text",
                "gold_value",
                "gold_pages",
                "raw_value",
                "row_headers",
                "col_headers",
                "match_score",
                "review_status",
                "notes",
            ],
        )
        writer.writeheader()
        for item in selected:
            row = item["manifest_row"]
            for spec in METRICS:
                question = question_by_company_metric[(row["company_name"], spec.key)]
                writer.writerow(
                    {
                        "company_name": row["company_name"],
                        "doc_id": row["doc_id"],
                        "metric_key": spec.key,
                        "question_id": question["id"],
                        "question_text": question["text"],
                        "gold_value": question["gold_value"],
                        "gold_pages": "|".join(str(page) for page in question["gold_pages"]),
                        "raw_value": question["metadata"]["source_raw_value"],
                        "row_headers": "|".join(question["metadata"].get("source_row_headers") or []),
                        "col_headers": "|".join(question["metadata"].get("source_col_headers") or []),
                        "match_score": question["metadata"]["match_score"],
                        "review_status": "pending",
                        "notes": "复核指标口径是否为 2024 年度/期末合并口径，且单位已换算为元。",
                    }
                )


def build_dataset(
    *,
    dataset_dir: Path,
    train_file: Path,
    output_dir: Path,
    company_count: int,
) -> Dict[str, Any]:
    manifest_rows = _load_manifest(dataset_dir / "document_manifest.csv")
    documents_dir = dataset_dir / "databases_ser_tab" / "chunked_reports"
    train_blob = _training_query_blob(train_file)
    raw_blob = _raw_query_blob(RAW_QUERY_FILES)

    train_holdout_rows = [row for row in manifest_rows if not _company_in_blob(row, train_blob)]
    raw_holdout_rows = [row for row in manifest_rows if not _company_in_blob(row, raw_blob)]

    candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for row in train_holdout_rows:
        document = _load_document(documents_dir, row["doc_id"])
        if not document:
            rejected.append({"doc_id": row["doc_id"], "company_name": row["company_name"], "reason": "missing_document"})
            continue
        matches = {spec.key: _extract_metric(document, spec) for spec in METRICS}
        missing = [key for key, value in matches.items() if value is None]
        if missing:
            rejected.append({
                "doc_id": row["doc_id"],
                "company_name": row["company_name"],
                "reason": "missing_metric",
                "missing_metric_keys": missing,
            })
            continue
        candidates.append({"manifest_row": row, "matches": matches})

    if len(candidates) < company_count:
        raise RuntimeError(f"Only found {len(candidates)} complete holdout companies; need {company_count}.")

    # Prefer broad industry coverage, then stronger extraction scores.
    def candidate_score(item: Dict[str, Any]) -> tuple[int, float, str]:
        matches = item["matches"]
        min_score = min(float(match["match_score"]) for match in matches.values())
        return (0, -min_score, item["manifest_row"]["company_name"])

    by_industry: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        industry = candidate["manifest_row"].get("major_industry") or candidate["manifest_row"].get("industry_l1") or "unknown"
        by_industry.setdefault(industry, []).append(candidate)
    for bucket in by_industry.values():
        bucket.sort(key=candidate_score)

    selected: List[Dict[str, Any]] = []
    while len(selected) < company_count:
        progressed = False
        for industry in sorted(by_industry):
            bucket = by_industry[industry]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= company_count:
                break
        if not progressed:
            break

    dataset_name = "number_holdout_metrics80"
    questions: List[Dict[str, Any]] = []
    answers: List[Dict[str, Any]] = []
    for company_index, item in enumerate(selected, start=1):
        row = item["manifest_row"]
        for metric_index, spec in enumerate(METRICS, start=1):
            question, answer = _build_question_answer(
                dataset_name=dataset_name,
                row=row,
                match=item["matches"][spec.key],
                spec=spec,
                company_index=company_index,
                metric_index=metric_index,
            )
            questions.append(question)
            answers.append(answer)

    metadata = {
        "owner": "FinaRAG",
        "quality_tier": "auto_extracted_gold",
        "review_required": True,
        "holdout_policy": "company names/security codes excluded from qwen3_reranker_sft_train query text",
        "train_file": str(train_file.relative_to(ROOT) if train_file.is_relative_to(ROOT) else train_file),
        "raw_teacher_query_holdout_feasibility": {
            "raw_query_files": [str(path.relative_to(ROOT)) for path in RAW_QUERY_FILES],
            "raw_holdout_company_count": len(raw_holdout_rows),
            "note": "Using all raw teacher/candidate queries leaves too few companies for a 20-company holdout in this corpus.",
        },
        "metric_keys": [spec.key for spec in METRICS],
        "selected_companies": [
            {
                "company_name": item["manifest_row"]["company_name"],
                "doc_id": item["manifest_row"]["doc_id"],
                "security_code": item["manifest_row"]["security_code"],
                "industry": item["manifest_row"].get("major_industry"),
            }
            for item in selected
        ],
        "candidate_company_count": len(candidates),
        "rejected_company_count": len(rejected),
    }
    question_set = FinanceEvalQuestionSet(
        schema_version="finance_eval_v1",
        dataset_name=dataset_name,
        questions=questions,
        metadata=metadata,
    )
    answer_set = FinanceGoldAnswerSet(
        schema_version="finance_eval_v1",
        dataset_name=dataset_name,
        answers=answers,
        metadata=metadata,
    )
    manifest = FinanceEvalManifest(
        schema_version="finance_eval_v1",
        dataset_name=dataset_name,
        description="Company-holdout number benchmark with four non-revenue financial metrics per company.",
        source_corpora=[str(dataset_dir.relative_to(ROOT) if dataset_dir.is_relative_to(ROOT) else dataset_dir)],
        question_count=len(questions),
        answer_count=len(answers),
        scoring_profile="finance_atomic_answer_rag_three_layer_v3",
        strata_summary={
            "all": len(questions),
            "kind/number": len(questions),
            **{f"metric/{spec.key}": company_count for spec in METRICS},
            "companies": company_count,
        },
        metadata=metadata,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    questions_path = output_dir / "questions.json"
    answers_path = output_dir / "answers_gold.json"
    manifest_path = output_dir / "manifest.json"
    review_path = output_dir / "review_checklist.csv"
    candidates_path = output_dir / "candidate_companies.json"
    selected_path = output_dir / "selected_companies.json"
    rejected_path = output_dir / "rejected_companies.json"

    _write_json(questions_path, question_set.model_dump(by_alias=True))
    _write_json(answers_path, answer_set.model_dump())
    _write_json(manifest_path, manifest.model_dump())
    _write_json(candidates_path, candidates)
    _write_json(selected_path, selected)
    _write_json(rejected_path, rejected)
    _write_review_checklist(review_path, selected, questions)

    alignment = validate_dataset_alignment(load_question_set(questions_path), load_gold_answer_set(answers_path))
    return {
        "questions_path": str(questions_path),
        "answers_path": str(answers_path),
        "manifest_path": str(manifest_path),
        "review_checklist_path": str(review_path),
        "candidate_companies_path": str(candidates_path),
        "selected_companies_path": str(selected_path),
        "rejected_companies_path": str(rejected_path),
        "alignment": alignment,
        "selected_company_count": len(selected),
        "question_count": len(questions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a company-holdout number benchmark for reranker evaluation.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--company-count", type=int, default=20)
    args = parser.parse_args()

    summary = build_dataset(
        dataset_dir=args.dataset_dir,
        train_file=args.train_file,
        output_dir=args.output_dir,
        company_count=args.company_count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
