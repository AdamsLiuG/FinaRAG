"""口径一致性变体查询集生成器。

从指标原子库自动生成变体查询对：同一原子的不同 aliases 表述应命中同一指标、
生成同口径扩展查询——这是 LLM-Wiki 原子库的口径一致性契约。

用法：
    python eval/metric_consistency/gen_variants.py --company 宁德时代 --year 2024 \
        --output eval/metric_consistency/variants.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metric_atoms import MetricAtom, get_default_store  # noqa: E402


def _representative_aliases(atom: MetricAtom, max_variants: int = 3) -> list[str]:
    """每个原子取代表性表述：标准名 + 最长/最短别名，覆盖长短变体。"""
    terms = [atom.standard_name, *atom.aliases]
    unique = list(dict.fromkeys(terms))
    if len(unique) <= max_variants:
        return unique
    by_len = sorted(unique, key=len, reverse=True)
    picked = {unique[0], by_len[0], by_len[-1]}
    for term in by_len:
        if len(picked) >= max_variants:
            break
        picked.add(term)
    return [term for term in unique if term in picked]


def build_variants(company: str, year: int) -> list[dict]:
    store = get_default_store()
    variants = []
    for atom in store.all_atoms():
        for alias in _representative_aliases(atom):
            variants.append(
                {
                    "atom_id": atom.atom_id,
                    "standard_name": atom.standard_name,
                    "question": f"{company}{year}年年报的{alias}是多少？",
                    "schema": "number",
                    "company": company,
                    "year": year,
                }
            )
            # 定义类变体：验证分层路由（definition_qa 直达原子）
            if alias == atom.standard_name:
                variants.append(
                    {
                        "atom_id": atom.atom_id,
                        "standard_name": atom.standard_name,
                        "question": f"{alias}是什么口径？和易混淆指标有什么区别？",
                        "schema": "text",
                        "company": company,
                        "year": year,
                        "expect_route": "metric_atom_store",
                    }
                )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="生成口径一致性变体查询集")
    parser.add_argument("--company", default="示例公司", help="公司名（按数据集实际填写）")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--output", default=str(Path(__file__).parent / "variants.jsonl"))
    args = parser.parse_args()

    variants = build_variants(args.company, args.year)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for variant in variants:
            fh.write(json.dumps(variant, ensure_ascii=False) + "\n")
    print(f"generated {len(variants)} variants -> {output}")


if __name__ == "__main__":
    main()
