"""口径一致性评测（LLM-Wiki 原子库价值验证）。

Level 1（无 API 依赖，可离线跑）：
    - 意图命中率：变体查询的 QueryPlan.metric 是否命中预期原子标准名；
    - 原子挂载率：QueryPlan.metric_atom 是否非空且字段完整；
    - 扩展口径检查：指标变体的 search_queries 不得含口径放宽词（如裸"收入"）；
    - 分层路由检查：定义类变体是否路由到 metric_atom_store。

Level 2（端到端，需 API key + 数据集，可选）：
    - 口径一致率：变体对答案的 citation 指向 evidence_targets 的比例；
    - 答案一致率：同原子变体对的数值答案是否一致。
    依赖完整 pipeline，评测脚本留接口，按环境条件启用。

用法：
    python eval/metric_consistency/run_metric_consistency.py \
        --variants eval/metric_consistency/variants.jsonl --report report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metric_atoms import get_default_store  # noqa: E402
from src.query_rewrite import QuestionRewriter  # noqa: E402

# 口径放宽词黑名单：这些词出现在 search_queries 中且不带限定词即视为口径污染
_CALIBER_POLLUTION_CHECKS = {
    "营业收入": ("收入", ("营业收入", "主营业务收入")),
}


def check_level1(variants: List[Dict]) -> Dict:
    rewriter = QuestionRewriter()
    store = get_default_store()
    metric_total = 0
    definition_total = 0
    intent_total = 0
    intent_hits = 0
    atom_mounted = 0
    pollution_violations: List[str] = []
    definition_routes = 0
    playbook_intent_hits = 0
    playbook_metric_mounted = 0

    for variant in variants:
        plan = rewriter.rewrite(variant["question"], schema=variant.get("schema", "number"))
        expect_route = variant.get("expect_route")
        if expect_route:
            definition_total += 1
            if plan.route == expect_route:
                definition_routes += 1
            continue

        expect_intent = variant.get("expect_intent")
        if expect_intent:
            # 剧本变体：同一意图的不同问法应命中同一剧本，且必需指标引用完整挂载
            intent_total += 1
            if plan.intent == expect_intent:
                playbook_intent_hits += 1
            payload = plan.playbook or {}
            expected_metrics = set(variant.get("required_metrics") or [])
            if expected_metrics and expected_metrics == set(payload.get("required_metrics") or []):
                playbook_metric_mounted += 1
            elif not expected_metrics and payload.get("playbook_id"):
                playbook_metric_mounted += 1
            continue

        metric_total += 1
        if plan.metric == variant["standard_name"]:
            intent_hits += 1
        payload = plan.metric_atom or {}
        if payload.get("atom_id") == variant["atom_id"] and payload.get("definition"):
            atom_mounted += 1

        metric = variant["standard_name"]
        if metric in _CALIBER_POLLUTION_CHECKS:
            bare_term, allowed_terms = _CALIBER_POLLUTION_CHECKS[metric]
            for query in plan.search_queries:
                if bare_term in query and not any(term in query for term in allowed_terms):
                    pollution_violations.append(query)
                    break

    atom_by_id = {atom.atom_id: atom for atom in store.all_atoms()}
    return {
        "level": 1,
        "total_variants": metric_total + definition_total + intent_total,
        "metric_variants": metric_total,
        "definition_variants": definition_total,
        "intent_variants": intent_total,
        "intent_hit_rate": round(intent_hits / metric_total, 4) if metric_total else None,
        "atom_mounted_rate": round(atom_mounted / metric_total, 4) if metric_total else None,
        "caliber_pollution_violations": pollution_violations,
        "definition_route_rate": (
            round(definition_routes / definition_total, 4) if definition_total else None
        ),
        "playbook_intent_hit_rate": (
            round(playbook_intent_hits / intent_total, 4) if intent_total else None
        ),
        "playbook_metric_mount_rate": (
            round(playbook_metric_mounted / intent_total, 4) if intent_total else None
        ),
        "atom_count": len(atom_by_id),
        "notes": "Level1 检查不依赖 API 与数据集；Level2（口径一致率/答案一致率）需端到端环境。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="口径一致性评测（Level1 离线可跑）")
    parser.add_argument(
        "--variants",
        default=str(Path(__file__).parent / "variants.jsonl"),
    )
    parser.add_argument("--report", default=str(Path(__file__).parent / "report_level1.json"))
    args = parser.parse_args()

    variant_path = Path(args.variants)
    if not variant_path.exists():
        print(f"variants not found: {variant_path}，先运行 gen_variants.py 生成")
        sys.exit(1)
    variants = [json.loads(line) for line in variant_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    report = check_level1(variants)
    report_path = Path(args.report)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
