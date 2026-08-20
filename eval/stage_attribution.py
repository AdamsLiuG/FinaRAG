"""F3 错误归因 v1：把错误样本归因到 pipeline 阶段（基于 gold + references 可判定分支）。

与 error_analysis 的区别：error_analysis 按题型/对错统计；本模块产出
阶段 × 错误率矩阵，为迭代优先级提供数据依据（"先修占比最大的阶段"）。

v1 归因树（短路判定，顺序即优先级）：
  pred.error 存在      → routing（公司路由失败）/ parse（解析失败）/ generation（其他异常）
  should_refuse 且未拒答 → refusal_policy（该拒不拒）
  should_refuse 且拒答   → correct_refusal（正确，非错误）
  非 refuse 题但 N/A    → over_refusal（不该拒而拒）
  value 非 N/A 且无引用  → ungrounded_generation（无证据作答）
  value == gold value    → 引用有交集 correct / 无交集 citation_error（答案对引用错）
  value != gold value    → 引用有交集 generation_error（证据对答案错）/
                           无交集 evidence_mislocalization（证据错位）
  gold 引用集合为空且值不匹配 → unknown（无位置信息，无法区分检索/生成，缺口显式）

v2 升级路径：D3 per-question trace 落地后，evidence_mislocalization 可细分
retrieval_miss / rerank_sink；pdf_parsing / text_splitter 需文档级比对（未覆盖）。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.error_analysis import _normalize_answer
from eval.metrics import load_answers_bundle

STAGES = [
    "routing",
    "parse",
    "refusal_policy",
    "over_refusal",
    "ungrounded_generation",
    "citation_error",
    "generation_error",
    "evidence_mislocalization",
]
NON_ERROR_STAGES = {"correct", "correct_refusal", "unknown"}


def _reference_keys(entry: Dict) -> Set[Tuple[str, int]]:
    """引用页键集合 (pdf_sha1, page_index)；缺字段条目跳过。"""
    keys: Set[Tuple[str, int]] = set()
    for ref in entry.get("references") or []:
        if not isinstance(ref, dict):
            continue  # 非法结构跳过（脏数据防御）
        sha = str(ref.get("pdf_sha1") or "")
        page = ref.get("page_index")
        if sha and isinstance(page, int):
            keys.add((sha, page))
    return keys


def _classify_error_message(error_message: str) -> str:
    """异常样本的阶段归类（关键词沿用 error_analysis，语义不变）。"""
    lowered = str(error_message).lower()
    if "company name found" in lowered or "ambiguous" in lowered:
        return "routing"
    if "parse" in lowered or "docling" in lowered or "pdf" in lowered:
        return "parse"
    return "generation"


def attribute_case(pred: Dict, gold: Optional[Dict]) -> str:
    """单题归因：返回 stage / correct / correct_refusal / unknown。"""
    if pred.get("error"):
        return _classify_error_message(pred["error"])

    pred_value = _normalize_answer(pred.get("value"))
    pred_refs = _reference_keys(pred)
    refused = pred_value in {"", "n/a", "none", "null"}

    if gold is None:
        # 无 gold 对照：只判结构性问题，值正确性不可判
        if refused:
            return "over_refusal"
        if not pred_refs:
            return "ungrounded_generation"
        return "unknown"

    gold_value = _normalize_answer(gold.get("value"))
    gold_refs = _reference_keys(gold)

    if gold.get("should_refuse"):
        return "correct_refusal" if refused else "refusal_policy"
    if refused:
        return "over_refusal"
    if not pred_refs:
        return "ungrounded_generation"

    value_matches = pred_value == gold_value
    refs_overlap = bool(pred_refs & gold_refs)

    if value_matches:
        # gold 无引用时值层面可判即 correct（无位置基准，不判引用错）
        return "correct" if (refs_overlap or not gold_refs) else "citation_error"
    if not gold_refs:
        # 无 gold 位置信息，检索/生成不可区分——缺口显式而非猜测
        return "unknown"
    return "generation_error" if refs_overlap else "evidence_mislocalization"


def attribution_matrix(preds: List[Dict], golds: List[Dict]) -> Dict:
    """阶段 × 错误率矩阵。pred 与 gold 按 question_text 对齐（pred 无 question_id）。"""
    gold_by_question = {
        gold.get("question_text"): gold for gold in golds if gold.get("question_text")
    }
    stage_counter: Counter = Counter()
    capability_stages: Dict[str, Counter] = defaultdict(Counter)
    examples: Dict[str, List[Dict]] = defaultdict(list)
    unmatched = 0

    for pred in preds:
        gold = gold_by_question.get(pred.get("question_text"))
        if gold is None:
            unmatched += 1
            continue
        stage = attribute_case(pred, gold)
        stage_counter[stage] += 1
        capability = gold.get("capability") or "unknown"
        capability_stages[capability][stage] += 1
        if stage not in NON_ERROR_STAGES and len(examples[stage]) < 3:
            examples[stage].append({
                "question_text": pred.get("question_text"),
                "kind": pred.get("kind"),
                "pred_value": pred.get("value"),
                "gold_value": gold.get("value"),
                "pred_references": [
                    {"pdf_sha1": r.get("pdf_sha1"), "page_index": r.get("page_index")}
                    for r in (pred.get("references") or [])[:3]
                ],
            })

    n = sum(stage_counter.values())
    return {
        "n": n,
        "unmatched_predictions": unmatched,
        "stage_counts": dict(stage_counter),
        "stage_rates": {
            stage: round(stage_counter[stage] / n, 4) if n else 0.0
            for stage in STAGES + sorted(NON_ERROR_STAGES)
            if stage_counter[stage]
        },
        "by_capability": {
            capability: dict(stages)
            for capability, stages in sorted(capability_stages.items())
        },
        "examples": dict(examples),
    }


def main():
    parser = argparse.ArgumentParser(
        description="FinaRAG 阶段归因矩阵：把错误样本归因到 pipeline 阶段"
    )
    parser.add_argument("--pred", type=Path, required=True, help="预测 answers bundle")
    parser.add_argument("--gold", type=Path, required=True, help="参考 answers_gold.json")
    parser.add_argument("--output", type=Path, default=None, help="结果 JSON 落盘路径")
    args = parser.parse_args()

    preds, _ = load_answers_bundle(args.pred)
    golds, _ = load_answers_bundle(args.gold)
    report = attribution_matrix(preds, golds)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"归因矩阵已写入 {args.output}")
    print(payload)


if __name__ == "__main__":
    main()
