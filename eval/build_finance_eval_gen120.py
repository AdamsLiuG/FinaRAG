from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "finance_eval_benchmark_v2"
CORE200_QUESTIONS = BENCHMARK_DIR / "core200" / "questions.json"
SOURCE_DATASET_DIR = ROOT / "data" / "top10_industries_2024_20each"
SOURCE_MANIFEST = SOURCE_DATASET_DIR / "document_manifest.csv"
REPORT_CHUNKS = SOURCE_DATASET_DIR / "metadata_store" / "report_chunk.jsonl"
OUTPUT_DIR = BENCHMARK_DIR / "gen120"


CAPABILITIES = [
    "generation_summary",
    "generation_risk_synthesis",
    "generation_strategy_synthesis",
    "generation_table_text_reasoning",
    "generation_cross_doc_compare",
    "generation_evidence_based_classification",
]


CAPABILITY_NOTES = {
    "generation_summary": "要求模型基于经营讨论类证据做2-3点归纳，规则只负责定位证据。",
    "generation_risk_synthesis": "要求模型把年报风险因素分组概括，不能用字段抽取直接回答。",
    "generation_strategy_synthesis": "要求模型综合主题战略、业务布局和具体举措。",
    "generation_table_text_reasoning": "要求模型结合表格背景和经营叙述解释一致性或原因。",
    "generation_cross_doc_compare": "要求模型对两家公司证据做共同点与差异对比。",
    "generation_evidence_based_classification": "要求模型在给定分类框架中做证据约束判断并说明依据。",
}


KEYWORD_GROUPS = {
    "generation_summary": ["经营情况", "管理层讨论", "主营业务", "营业收入", "利润", "增长", "下降"],
    "generation_risk_synthesis": ["风险", "不确定", "挑战", "压力", "可能面临", "风险因素"],
    "generation_strategy_synthesis": ["战略", "布局", "研发", "数字化", "人工智能", "绿色", "出海", "国产替代"],
    "generation_table_text_reasoning": ["营业收入", "主要会计数据", "财务指标", "经营情况", "主营业务", "利润表"],
    "generation_cross_doc_compare": ["战略", "业务", "人工智能", "数字化", "绿色", "研发", "产品"],
    "generation_evidence_based_classification": ["增长", "成本", "效率", "风险", "转型", "战略", "研发", "市场"],
}


STRATEGY_TAG_FALLBACKS = ["数字化转型", "人工智能", "绿色转型", "出海", "国产替代"]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_manifest() -> Dict[str, Dict[str, Any]]:
    with SOURCE_MANIFEST.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["doc_id"]: row for row in reader}


def _load_report_chunks(doc_ids: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    wanted = set(doc_ids)
    chunks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with REPORT_CHUNKS.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            doc_id = str(row.get("doc_id") or row.get("report_id") or "")
            if doc_id in wanted:
                chunks[doc_id].append(row)
    return chunks


def _normalize_pages(values: Iterable[Any]) -> List[int]:
    pages = []
    for value in values:
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page > 0 and page not in pages:
            pages.append(page)
    return pages[:4]


def _score_chunk(chunk: Dict[str, Any], keywords: List[str], extra_keywords: Optional[List[str]] = None) -> int:
    text = f"{chunk.get('section_name') or ''}\n{chunk.get('section_title') or ''}\n{chunk.get('search_text') or ''}"
    score = 0
    for keyword in keywords + list(extra_keywords or []):
        if keyword and keyword in text:
            score += 3
    if chunk.get("has_table_context"):
        score += 1
    page = chunk.get("page")
    if isinstance(page, int) and page <= 5:
        score -= 1
    return score


def _pick_pages(
    chunks_by_doc: Dict[str, List[Dict[str, Any]]],
    doc_id: str,
    capability: str,
    fallback_pages: Optional[List[int]] = None,
    extra_keywords: Optional[List[str]] = None,
) -> List[int]:
    scored = []
    for chunk in chunks_by_doc.get(doc_id, []):
        page = chunk.get("page")
        if not isinstance(page, int):
            continue
        score = _score_chunk(chunk, KEYWORD_GROUPS[capability], extra_keywords=extra_keywords)
        if score > 0:
            scored.append((score, page))
    scored.sort(key=lambda item: (-item[0], item[1]))
    pages = _normalize_pages(page for _, page in scored)
    return pages or _normalize_pages(fallback_pages or [2])


def _references(
    doc_ids: List[str],
    gold_pages: List[int],
    reference_doc_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for index, page in enumerate(gold_pages):
        if reference_doc_ids and index < len(reference_doc_ids):
            doc_id = reference_doc_ids[index]
        else:
            doc_id = doc_ids[min(index, len(doc_ids) - 1)]
        refs.append(
            {
                "pdf_sha1": doc_id,
                "page_index": max(0, int(page) - 1),
                "chunk_id": None,
                "section_name": None,
                "evidence_type": "narrative",
            }
        )
    return refs


def _manifest_for_doc(doc_id: str, manifest: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row = manifest.get(doc_id, {})
    return {
        "company_name": row.get("company_name"),
        "stock_code": row.get("security_code"),
        "industry_l1": row.get("industry_l1") or row.get("major_industry"),
        "group_name": row.get("industry_l1") or row.get("major_industry"),
    }


def _strategy_tag(index: int, doc_id: str, chunks_by_doc: Dict[str, List[Dict[str, Any]]]) -> str:
    for chunk in chunks_by_doc.get(doc_id, []):
        tags = [tag for tag in chunk.get("strategy_tags") or [] if tag]
        if tags:
            return tags[index % len(tags)]
    return STRATEGY_TAG_FALLBACKS[index % len(STRATEGY_TAG_FALLBACKS)]


def _base_question_payload(
    *,
    question_id: str,
    text: str,
    capability: str,
    difficulty: str,
    doc_ids: List[str],
    company_name: Optional[str],
    mentioned_companies: Optional[List[str]],
    gold_pages: List[int],
    metric_name: str,
    manifest: Dict[str, Dict[str, Any]],
    value: str,
    notes: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
    reference_doc_ids: Optional[List[str]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    first_doc = doc_ids[0]
    manifest_info = _manifest_for_doc(first_doc, manifest)
    metadata = {
        "source_dataset": "finance_eval_benchmark_v2_gen120",
        "scenario_family": capability,
        "generation_method": "template_from_core200_and_report_metadata",
        "requires_generation": True,
        "rules_role": "evidence_only",
        "answer_style": "2-4 sentence evidence-grounded synthesis",
        "review_required": True,
        **(extra_metadata or {}),
    }
    question = {
        "id": question_id,
        "text": text,
        "kind": "text",
        "capability": capability,
        "difficulty": difficulty,
        "doc_ids": doc_ids,
        "company_name": company_name,
        "mentioned_companies": mentioned_companies or [],
        "stock_code": manifest_info["stock_code"] if company_name else None,
        "report_year": 2024,
        "report_type": "annual_report",
        "period": None,
        "metric_name": metric_name,
        "currency": None,
        "unit": None,
        "section_name": None,
        "industry_l1": manifest_info["industry_l1"],
        "group_id": None,
        "group_name": manifest_info["group_name"],
        "group_slot": None,
        "evidence_type": "narrative_synthesis",
        "gold_value": value,
        "gold_pages": gold_pages,
        "gold_chunk_ids": [],
        "should_refuse": False,
        "expected_filters": {
            "doc_ids": doc_ids,
            "report_year": 2024,
            "report_type": "annual_report",
        },
        "annotation_status": "draft_generation_required",
        "notes": notes,
        "metadata": metadata,
    }
    answer = {
        "question_id": question_id,
        "question_text": text,
        "kind": "text",
        "value": value,
        "doc_ids": doc_ids,
        "gold_pages": gold_pages,
        "gold_chunk_ids": [],
        "references": _references(doc_ids, gold_pages, reference_doc_ids=reference_doc_ids),
        "company_name": company_name,
        "stock_code": manifest_info["stock_code"] if company_name else None,
        "report_year": 2024,
        "report_type": "annual_report",
        "period": None,
        "metric_name": metric_name,
        "currency": None,
        "unit": None,
        "evidence_type": "narrative_synthesis",
        "capability": capability,
        "difficulty": difficulty,
        "group_id": None,
        "group_name": manifest_info["group_name"],
        "should_refuse": False,
        "annotation_status": "draft_generation_required",
        "notes": notes,
        "metadata": metadata,
    }
    return question, answer


def _single_doc_pool(core_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    pool = []
    for question in core_questions:
        company_name = question.get("company_name")
        doc_ids = question.get("doc_ids") or []
        if not company_name or len(doc_ids) != 1:
            continue
        key = (company_name, doc_ids[0])
        if key in seen:
            continue
        seen.add(key)
        pool.append(question)
    return pool


def _pair_pool(core_questions: List[Dict[str, Any]], single_pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs = [
        question
        for question in core_questions
        if len(question.get("mentioned_companies") or []) >= 2 and len(question.get("doc_ids") or []) >= 2
    ]
    for index in range(0, min(20, len(single_pool) - 1), 2):
        first = single_pool[index]
        second = single_pool[index + 1]
        pairs.append(
            {
                "mentioned_companies": [first["company_name"], second["company_name"]],
                "doc_ids": [first["doc_ids"][0], second["doc_ids"][0]],
                "gold_pages": [*(first.get("gold_pages") or [2])[:1], *(second.get("gold_pages") or [2])[:1]],
            }
        )
    unique_pairs = []
    seen = set()
    for pair in pairs:
        companies = tuple(pair.get("mentioned_companies") or [])
        doc_ids = tuple(pair.get("doc_ids") or [])
        key = (companies, doc_ids)
        if len(companies) < 2 or len(doc_ids) < 2 or key in seen:
            continue
        seen.add(key)
        unique_pairs.append(pair)
    return unique_pairs


def _take_cycled(pool: List[Dict[str, Any]], offset: int, count: int) -> List[Dict[str, Any]]:
    if not pool:
        raise RuntimeError("Cannot sample from an empty pool.")
    return [pool[(offset + index) % len(pool)] for index in range(count)]


def _build_records() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    core_questions = _load_json(CORE200_QUESTIONS)["questions"]
    manifest = _load_manifest()
    single_pool = _single_doc_pool(core_questions)
    pair_pool = _pair_pool(core_questions, single_pool)
    doc_ids = {
        doc_id
        for question in single_pool
        for doc_id in (question.get("doc_ids") or [])
    } | {
        doc_id
        for question in pair_pool
        for doc_id in (question.get("doc_ids") or [])
    }
    chunks_by_doc = _load_report_chunks(doc_ids)

    questions: List[Dict[str, Any]] = []
    answers: List[Dict[str, Any]] = []

    def add(question: Dict[str, Any], answer: Dict[str, Any]) -> None:
        questions.append(question)
        answers.append(answer)

    for index, source in enumerate(_take_cycled(single_pool, 0, 20), start=1):
        company = source["company_name"]
        doc_id = source["doc_ids"][0]
        pages = _pick_pages(chunks_by_doc, doc_id, "generation_summary", source.get("gold_pages"))
        value = f"参考答案应基于{company}2024年年报证据，归纳经营表现或业务变化的2-3个主要原因，并区分财务数据背景与管理层文字解释。"
        q, a = _base_question_payload(
            question_id=f"gen120-summary-{index:03d}",
            text=f"请结合{company}2024年年报中的经营情况或管理层讨论证据，概括公司年度经营表现变化的主要原因，列出2-3点。",
            capability="generation_summary",
            difficulty="medium",
            doc_ids=[doc_id],
            company_name=company,
            mentioned_companies=[],
            gold_pages=pages,
            metric_name="经营表现原因归纳",
            manifest=manifest,
            value=value,
            notes=CAPABILITY_NOTES["generation_summary"],
            extra_metadata={"must_cover": ["经营表现变化", "管理层解释", "2-3个原因"]},
        )
        add(q, a)

    for index, source in enumerate(_take_cycled(single_pool, 20, 20), start=1):
        company = source["company_name"]
        doc_id = source["doc_ids"][0]
        pages = _pick_pages(chunks_by_doc, doc_id, "generation_risk_synthesis", source.get("gold_pages"))
        value = f"参考答案应基于{company}2024年年报披露内容，按业务、市场、财务、合规或经营环境等维度概括主要风险，不得添加外部行业常识。"
        q, a = _base_question_payload(
            question_id=f"gen120-risk-{index:03d}",
            text=f"请根据{company}2024年年报披露的风险或挑战，按来源归纳2-3类主要风险，并说明每类风险的含义。",
            capability="generation_risk_synthesis",
            difficulty="medium",
            doc_ids=[doc_id],
            company_name=company,
            mentioned_companies=[],
            gold_pages=pages,
            metric_name="主要风险归纳",
            manifest=manifest,
            value=value,
            notes=CAPABILITY_NOTES["generation_risk_synthesis"],
            extra_metadata={"must_cover": ["风险分类", "风险含义", "证据约束"]},
        )
        add(q, a)

    for index, source in enumerate(_take_cycled(single_pool, 40, 20), start=1):
        company = source["company_name"]
        doc_id = source["doc_ids"][0]
        tag = _strategy_tag(index, doc_id, chunks_by_doc)
        pages = _pick_pages(chunks_by_doc, doc_id, "generation_strategy_synthesis", source.get("gold_pages"), [tag])
        value = f"参考答案应围绕{company}年报中的“{tag}”相关证据，概括具体业务布局、研发投入、产品/服务举措或管理动作。"
        q, a = _base_question_payload(
            question_id=f"gen120-strategy-{index:03d}",
            text=f"{company}2024年年报中围绕“{tag}”有哪些具体举措或业务布局？请结合年报证据概括。",
            capability="generation_strategy_synthesis",
            difficulty="hard",
            doc_ids=[doc_id],
            company_name=company,
            mentioned_companies=[],
            gold_pages=pages,
            metric_name=f"{tag}战略举措",
            manifest=manifest,
            value=value,
            notes=CAPABILITY_NOTES["generation_strategy_synthesis"],
            extra_metadata={"strategy_tag": tag, "must_cover": ["具体举措", "业务布局", "证据页"]},
        )
        add(q, a)

    for index, source in enumerate(_take_cycled(single_pool, 60, 20), start=1):
        company = source["company_name"]
        doc_id = source["doc_ids"][0]
        pages = _pick_pages(chunks_by_doc, doc_id, "generation_table_text_reasoning", source.get("gold_pages"))
        value = f"参考答案应结合{company}年报中的营业收入或主要财务指标表格，以及经营情况文字说明，判断数据表现与业务叙述之间的对应关系。"
        q, a = _base_question_payload(
            question_id=f"gen120-table-text-{index:03d}",
            text=f"请结合{company}2024年年报中的营业收入相关表格和经营情况说明，解释财务表现与业务叙述之间是否相互支撑。",
            capability="generation_table_text_reasoning",
            difficulty="hard",
            doc_ids=[doc_id],
            company_name=company,
            mentioned_companies=[],
            gold_pages=pages,
            metric_name="表格与文本综合解释",
            manifest=manifest,
            value=value,
            notes=CAPABILITY_NOTES["generation_table_text_reasoning"],
            extra_metadata={"must_cover": ["营业收入或财务指标", "经营情况说明", "对应关系"]},
        )
        add(q, a)

    for index, source in enumerate(pair_pool[:20], start=1):
        companies = source["mentioned_companies"][:2]
        docs = source["doc_ids"][:2]
        tag = _strategy_tag(index, docs[0], chunks_by_doc)
        fallbacks = list(source.get("gold_pages") or [2, 2])
        while len(fallbacks) < len(docs):
            fallbacks.append(2)
        page_doc_pairs = []
        for doc_id, fallback in zip(docs, fallbacks):
            picked_pages = _pick_pages(chunks_by_doc, doc_id, "generation_cross_doc_compare", [fallback], [tag])
            pair = (doc_id, picked_pages[0])
            if pair not in page_doc_pairs:
                page_doc_pairs.append(pair)
        pages = [page for _, page in page_doc_pairs]
        reference_doc_ids = [doc_id for doc_id, _ in page_doc_pairs]
        value = f"参考答案应比较{companies[0]}和{companies[1]}年报中关于“{tag}”或业务发展的共同点与差异，分别覆盖两家公司证据。"
        q, a = _base_question_payload(
            question_id=f"gen120-cross-doc-{index:03d}",
            text=f"对比{companies[0]}和{companies[1]}2024年年报中关于“{tag}”或业务发展的表述，概括共同点和差异。",
            capability="generation_cross_doc_compare",
            difficulty="hard",
            doc_ids=docs,
            company_name=None,
            mentioned_companies=companies,
            gold_pages=pages,
            metric_name=f"{tag}跨公司对比",
            manifest=manifest,
            value=value,
            notes=CAPABILITY_NOTES["generation_cross_doc_compare"],
            extra_metadata={"strategy_tag": tag, "must_cover": ["共同点", "差异", companies[0], companies[1]]},
            reference_doc_ids=reference_doc_ids,
        )
        add(q, a)

    for index, source in enumerate(_take_cycled(single_pool, 80, 20), start=1):
        company = source["company_name"]
        doc_id = source["doc_ids"][0]
        pages = _pick_pages(chunks_by_doc, doc_id, "generation_evidence_based_classification", source.get("gold_pages"))
        value = f"参考答案应把{company}的年度经营叙述归入给定四类之一，并用年报证据说明为什么该类比其他类别更合适。"
        q, a = _base_question_payload(
            question_id=f"gen120-classification-{index:03d}",
            text=(
                f"根据{company}2024年年报证据，将公司的年度经营叙述主要归类为"
                "“增长驱动”“成本/效率改善”“风险压力应对”或“战略转型布局”中的哪一类？请说明依据。"
            ),
            capability="generation_evidence_based_classification",
            difficulty="hard",
            doc_ids=[doc_id],
            company_name=company,
            mentioned_companies=[],
            gold_pages=pages,
            metric_name="证据约束分类判断",
            manifest=manifest,
            value=value,
            notes=CAPABILITY_NOTES["generation_evidence_based_classification"],
            extra_metadata={
                "allowed_labels": ["增长驱动", "成本/效率改善", "风险压力应对", "战略转型布局"],
                "must_cover": ["分类标签", "选择依据", "排除或弱化其他类别的理由"],
            },
        )
        add(q, a)

    return questions, answers


def _strata_summary(questions: List[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    counter["all"] = len(questions)
    for question in questions:
        counter[f"kind/{question['kind']}"] += 1
        counter[f"capability/{question['capability']}"] += 1
        counter[f"difficulty/{question['difficulty']}"] += 1
        counter[f"scenario/{question['metadata']['scenario_family']}"] += 1
        if question.get("industry_l1"):
            counter[f"industry/{question['industry_l1']}"] += 1
    return dict(sorted(counter.items()))


def _split_payloads(records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dev = [record for index, record in enumerate(records) if index % 2 == 0]
    test = [record for index, record in enumerate(records) if index % 2 == 1]
    return dev, test


def _write_review_checklist(path: Path, questions: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "question_id",
        "split",
        "scenario_family",
        "kind",
        "capability",
        "difficulty",
        "group_name",
        "company_name",
        "mentioned_companies",
        "doc_ids",
        "gold_pages",
        "annotation_status",
        "review_priority",
        "review_focus",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, question in enumerate(questions):
            writer.writerow(
                {
                    "question_id": question["id"],
                    "split": "dev" if index % 2 == 0 else "test",
                    "scenario_family": question["metadata"]["scenario_family"],
                    "kind": question["kind"],
                    "capability": question["capability"],
                    "difficulty": question["difficulty"],
                    "group_name": question.get("group_name") or "",
                    "company_name": question.get("company_name") or "",
                    "mentioned_companies": "|".join(question.get("mentioned_companies") or []),
                    "doc_ids": "|".join(question.get("doc_ids") or []),
                    "gold_pages": "|".join(str(page) for page in question.get("gold_pages") or []),
                    "annotation_status": question["annotation_status"],
                    "review_priority": "high",
                    "review_focus": "复核问题是否必须生成模型综合回答、gold_pages 是否覆盖关键证据、参考答案要点是否足够具体。",
                    "notes": question.get("notes") or "",
                }
            )


def build_gen120() -> None:
    questions, answers = _build_records()
    if len(questions) != 120 or len(answers) != 120:
        raise RuntimeError(f"Expected 120 questions/answers, got {len(questions)} and {len(answers)}.")

    capability_counts = Counter(question["capability"] for question in questions)
    for capability in CAPABILITIES:
        if capability_counts[capability] != 20:
            raise RuntimeError(f"Expected 20 records for {capability}, got {capability_counts[capability]}.")

    question_payload = {
        "schema_version": "finance_eval_v1",
        "dataset_name": "finance_eval_benchmark_v2_gen120",
        "questions": questions,
        "metadata": {
            "owner": "FinaRAG",
            "quality_tier": "draft_generation_required",
            "review_required": True,
            "purpose": "补充必须调用生成模型的综合、归纳、解释和对比题。",
        },
    }
    answer_payload = {
        "schema_version": "finance_eval_v1",
        "dataset_name": "finance_eval_benchmark_v2_gen120",
        "answers": answers,
        "metadata": {
            "owner": "FinaRAG",
            "quality_tier": "draft_generation_required",
            "review_required": True,
            "answer_policy": "value 字段为参考答案/评分要点，正式使用前建议人工或教师模型细化为完整 gold answer。",
        },
    }
    manifest = {
        "schema_version": "finance_eval_v1",
        "dataset_name": "finance_eval_benchmark_v2_gen120",
        "description": "120 generation-required finance RAG questions, 20 for each synthesis/compare/reasoning capability.",
        "source_corpora": [
            "finance_eval_benchmark_v2/core200",
            "top10_industries_2024_20each",
            "top10_industries_2024_20each/metadata_store/report_chunk.jsonl",
        ],
        "question_count": len(questions),
        "answer_count": len(answers),
        "scoring_profile": "finance_generation_required_v1",
        "strata_summary": _strata_summary(questions),
        "metadata": {
            "requires_generation": True,
            "rules_role": "evidence_only",
            "recommended_run_note": "Use kind=text support; direct-answer rules should not produce final answers for this split.",
            "capabilities": CAPABILITIES,
        },
    }

    _write_json(OUTPUT_DIR / "questions.json", question_payload)
    _write_json(OUTPUT_DIR / "answers_gold.json", answer_payload)
    _write_json(OUTPUT_DIR / "dataset_manifest.json", manifest)
    _write_review_checklist(OUTPUT_DIR / "review_checklist.csv", questions)

    dev_questions, test_questions = _split_payloads(questions)
    dev_answers, test_answers = _split_payloads(answers)
    _write_json(OUTPUT_DIR / "splits" / "dev" / "questions.json", {**question_payload, "questions": dev_questions})
    _write_json(OUTPUT_DIR / "splits" / "test" / "questions.json", {**question_payload, "questions": test_questions})
    _write_json(OUTPUT_DIR / "splits" / "dev" / "answers_gold.json", {**answer_payload, "answers": dev_answers})
    _write_json(OUTPUT_DIR / "splits" / "test" / "answers_gold.json", {**answer_payload, "answers": test_answers})


if __name__ == "__main__":
    build_gen120()
