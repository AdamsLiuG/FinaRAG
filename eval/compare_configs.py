from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.run_eval import evaluate_answers_file, run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run and compare multiple FinaRAG configs.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/test_set"))
    parser.add_argument("--configs", default="qwen_base,qwen_rerank,qwen_ser_rerank")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    reports = []
    for config_name in [item.strip() for item in args.configs.split(",") if item.strip()]:
        answers_file = run_pipeline(args.dataset_dir, config_name, None)
        report = evaluate_answers_file(answers_file)
        report["config"] = config_name
        reports.append(report)

    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(reports, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
