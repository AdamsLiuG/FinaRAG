"""原子对抗质检 CLI：LLM 多维打分 → 门禁裁决 → Markdown 报告供人工校准。

用法（需 API key 环境）：
    python scripts/review_metric_atoms.py --review-model qwen-max --mining-model qwen-plus
    python scripts/review_metric_atoms.py --atoms MET-001 MET-009 --output reports/atom_review.md

质检模型必须与提炼模型不同（生产者不自批），相同则拒绝执行。
退出码：0 门禁通过 / 2 模型相同被拒 / 3 门禁未通过（有原子需返修或拒绝）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metric_atom_review import (  # noqa: E402
    SameModelReviewError,
    aggregate_review,
    assert_reviewer_differs,
    render_review_report,
    review_atom_with_llm,
)
from src.metric_atoms import MetricAtomStore  # noqa: E402


def atom_payload(atom, store: MetricAtomStore) -> dict:
    return {
        "atom_id": atom.atom_id,
        "standard_name": atom.standard_name,
        "aliases": list(atom.aliases),
        "definition": atom.definition,
        "caliber_notes": atom.caliber_notes,
        "unit": atom.unit,
        "statement_location": atom.statement_location,
        "pitfalls": atom.pitfalls,
        "confusable_names": [
            conf.standard_name
            for ref in atom.confusable_with
            if (conf := store.get(ref)) is not None
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="对抗质检已入库的指标原子")
    parser.add_argument("--provider", default="qwen")
    parser.add_argument("--review-model", default=None, help="质检模型（须与提炼模型不同）")
    parser.add_argument("--mining-model", default="", help="提炼模型（用于校验生产者不自批）")
    parser.add_argument("--atoms", nargs="*", default=None, help="指定 atom_id，默认全库")
    parser.add_argument("--output", default="reports/atom_review.md")
    parser.add_argument("--dry-run", action="store_true", help="只列出受检原子，不调用 LLM")
    args = parser.parse_args()

    try:
        assert_reviewer_differs(args.mining_model or "", args.review_model or "")
    except SameModelReviewError as exc:
        print(f"拒绝执行：{exc}")
        sys.exit(2)

    store = MetricAtomStore.load()
    atoms = store.all_atoms()
    if args.atoms:
        wanted = set(args.atoms)
        atoms = [atom for atom in atoms if atom.atom_id in wanted]
        missing = wanted - {atom.atom_id for atom in atoms}
        if missing:
            print(f"未找到原子：{sorted(missing)}")
            sys.exit(1)

    if args.dry_run:
        for atom in atoms:
            print(f"{atom.atom_id}\t{atom.standard_name}")
        print(f"\n共 {len(atoms)} 个原子待质检（dry-run，未调用 LLM）")
        sys.exit(0)

    from src.api_requests import APIProcessor

    print(f"质检 {len(atoms)} 个原子（provider={args.provider} model={args.review_model}）...")
    processor = APIProcessor(provider=args.provider)
    reviews = []
    for atom in atoms:
        review = review_atom_with_llm(processor, atom_payload(atom, store), model=args.review_model)
        reviews.append(review)
        scores = review.get("scores") or {}
        total = sum(scores.values()) if scores else "-"
        print(f"[{str(review.get('verdict')):7s}] {atom.atom_id} {atom.standard_name} 总分={total}")

    summary = aggregate_review(reviews)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_review_report(reviews), encoding="utf-8")
    print(f"\n报告 -> {output}")
    print(f"门禁：{'通过' if summary['gate_passed'] else '未通过'}；需处理：{summary['blocking']}")
    sys.exit(0 if summary["gate_passed"] else 3)


if __name__ == "__main__":
    main()
