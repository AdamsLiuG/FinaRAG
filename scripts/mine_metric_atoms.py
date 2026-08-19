"""蒸馏扩原子 CLI（P3a）：LLM 草稿 → 规则校验 → draft YAML 供人工校准。

用法（需 API key 环境）：
    python scripts/mine_metric_atoms.py --n-target 40 --output config/metric_atoms_draft.yaml
    python scripts/mine_metric_atoms.py --corpus path/to/年报摘要.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api_requests import APIProcessor  # noqa: E402
from src.metric_atom_mining import mine_atoms_with_llm, render_draft_yaml, validate_atom_drafts  # noqa: E402
from src.metric_atoms import MetricAtomStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 蒸馏挖掘指标原子草稿")
    parser.add_argument("--corpus", default="", help="可选语料文件路径（年报摘要/会计准则文本）")
    parser.add_argument("--n-target", type=int, default=40)
    parser.add_argument("--provider", default="qwen")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default="config/metric_atoms_draft.yaml")
    args = parser.parse_args()

    corpus_hint = ""
    if args.corpus:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            print(f"corpus not found: {corpus_path}")
            sys.exit(1)
        corpus_hint = corpus_path.read_text(encoding="utf-8")

    print(f"调用 LLM 挖掘 {args.n_target} 个候选原子草稿（provider={args.provider}）...")
    processor = APIProcessor(provider=args.provider)
    if args.model:
        processor.processor.default_model = args.model
    drafts = mine_atoms_with_llm(processor, corpus_hint=corpus_hint, n_target=args.n_target)
    print(f"LLM 返回 {len(drafts)} 个草稿")

    store = MetricAtomStore.load()
    results = validate_atom_drafts(drafts, store)
    rejected = {name for name, errors in results.items() if errors}
    accepted = [name for name, errors in results.items() if not errors]
    for name, errors in results.items():
        status = "PASS" if not errors else "REJECT"
        print(f"[{status}] {name}")
        for error in errors:
            print(f"    - {error}")

    output = Path(args.output)
    output.write_text(render_draft_yaml(drafts, accepted=accepted), encoding="utf-8")
    print(f"\n{len(accepted)} 个草稿通过规则校验 -> {output}（人工校准后合入 metric_atoms.yaml）")
    print(f"{len(rejected)} 个草稿被拒绝（口径放宽/与现有原子重叠/字段缺失）")


if __name__ == "__main__":
    main()
