"""④ trace 反哺知识层：自适应检索运行经验 → 知识资产进化建议。

只出建议不自动改库（Builder 永不自批）：建议经人工确认 → 编辑 yaml →
compile_atoms.py --update → 既有等价测试与门禁守护。
三条规则：
R1 高命中词扩 aliases：query 命中率 > 80% 且样本 ≥3 → 检索侧常靠它命中，
   查询改写侧应预展开（aliases 命中即免补检一轮）。
R2 零命中 target 复核：某 evidence_target 触发补检后 new_hits 全 0 →
   文档可能从不这么写，表述需复核。
R3 无触发无操作：原子从未触发补检 → 知识已足够，不是缺口。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

if __package__ in {None, ""} or __package__ == "scripts":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.knowledge_loop import cluster_candidate_atoms, normalize_trace_entry

MIN_HIT_RATE_FOR_ALIAS = 0.8
MIN_SAMPLES_FOR_SUGGESTION = 3


def load_trace(path: Path) -> Tuple[List[Dict], int]:
    """JSONL 读取；脏行跳过并计数（缺口显式）。"""
    entries, skipped = [], 0
    if not path.exists():
        return entries, skipped
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    return entries, skipped


def _query_stats(entries: List[Dict]) -> Dict[str, Dict]:
    """query → {hits, total}：new_hits>0 记一次命中。"""
    stats: Dict[str, Dict] = defaultdict(lambda: {"hits": 0, "total": 0})
    for raw in entries:
        entry = normalize_trace_entry(raw)
        queries = entry.get("queries") or []
        hit = (entry.get("new_hits") or 0) > 0
        for query in queries:
            stats[query]["total"] += 1
            if hit:
                stats[query]["hits"] += 1
    return stats


def per_atom_stats(entries: List[Dict]) -> List[Dict]:
    rows = []
    by_atom: Dict[str, List[Dict]] = defaultdict(list)
    for raw in entries:
        entry = normalize_trace_entry(raw)
        by_atom[entry.get("atom_id") or "-"].append(entry)
    for atom_id, atom_entries in sorted(by_atom.items()):
        n = len(atom_entries)
        rows.append({
            "atom_id": atom_id,
            "triggers": n,
            "trigger_dist": dict(Counter(e.get("trigger") for e in atom_entries)),
            "strategy_dist": dict(Counter(e.get("strategy") for e in atom_entries)),
            "rounds_total": sum(e.get("rounds") or 0 for e in atom_entries),
            "gap_closed_rate": round(
                sum(1 for e in atom_entries if e.get("gap_closed")) / n, 4),
            "outcome_dist": dict(Counter(e.get("outcome") for e in atom_entries)),
        })
    return rows


def alias_expansion_suggestions(entries: List[Dict], store) -> List[Dict]:
    """R1/R2（R3 无操作）。建议含 evidence 计数，供人工裁决。"""
    norm_entries = [normalize_trace_entry(e) for e in entries]
    suggestions = []
    query_stats = _query_stats(norm_entries)
    atom_ids = {e.get("atom_id") for e in norm_entries if e.get("atom_id")}
    for atom_id in sorted(atom_ids):
        atom = store.get(atom_id)
        if atom is None:
            continue
        for query, stat in sorted(query_stats.items()):
            if stat["total"] < MIN_SAMPLES_FOR_SUGGESTION:
                continue  # 样本不足不出建议
            hit_rate = stat["hits"] / stat["total"]
            if hit_rate >= MIN_HIT_RATE_FOR_ALIAS and query not in atom.aliases:
                suggestions.append({
                    "rule": "R1_expand_alias",
                    "atom_id": atom_id,
                    "term": query,
                    "evidence": f"补检命中率 {hit_rate:.0%}（{stat['hits']}/{stat['total']}）",
                })
            if hit_rate == 0.0 and query in atom.evidence_targets:
                suggestions.append({
                    "rule": "R2_review_target",
                    "atom_id": atom_id,
                    "term": query,
                    "evidence": f"补检零命中（0/{stat['total']}）——文档可能无此表述",
                })
    return suggestions


def render_report(
    rows: List[Dict],
    suggestions: List[Dict],
    skipped: int,
    candidate_suggestions: Optional[List[Dict]] = None,
) -> str:
    all_suggestions = [*suggestions, *(candidate_suggestions or [])]
    lines = ["# Adaptive Trace 分析报告", "",
             f"- trace 条数：{sum(r['triggers'] for r in rows)}",
             f"- 脏行跳过：{skipped}", "", "## 每原子统计", "",
             "| atom | 触发 | 收敛率 | outcome 分布 |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['atom_id']} | {row['triggers']} | "
                     f"{row['gap_closed_rate']} | {row['outcome_dist']} |")
    lines += ["", "## 建议（人工确认后入库）", ""]
    if not all_suggestions:
        lines.append("（无——样本不足或知识已覆盖）")
    for suggestion in all_suggestions:
        lines.append(f"- **{suggestion['rule']}** {suggestion['atom_id']}: "
                     f"`{suggestion['term']}` —— {suggestion['evidence']}")
    lines += ["", "> 入库流程：人工确认 → 编辑 config/metric_atoms.yaml → "
              "python scripts/compile_atoms.py --update（等价测试与门禁守护）"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="adaptive trace → 知识进化建议")
    parser.add_argument("--trace", type=Path, default=Path("logs/adaptive_trace.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/reports/adaptive_analysis.md"))
    parser.add_argument("--draft-output", type=Path, default=None, help="可选候选新原子草稿 YAML 输出路径")
    args = parser.parse_args()
    entries, skipped = load_trace(args.trace)
    rows = per_atom_stats(entries)
    from src.metric_atoms import get_default_store

    store = get_default_store()
    suggestions = alias_expansion_suggestions(entries, store)
    clustered = cluster_candidate_atoms(entries, store, min_samples=MIN_SAMPLES_FOR_SUGGESTION)
    cand_suggestions = clustered["suggestions"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_report(rows, suggestions, skipped, candidate_suggestions=cand_suggestions),
        encoding="utf-8",
    )
    if args.draft_output and clustered["draft_yaml"]:
        args.draft_output.parent.mkdir(parents=True, exist_ok=True)
        args.draft_output.write_text(clustered["draft_yaml"], encoding="utf-8")
    total_suggestions = len(suggestions) + len(cand_suggestions)
    print(f"报告已写入 {args.output}（{len(entries)} 条 trace，{total_suggestions} 条建议）")


if __name__ == "__main__":
    main()
