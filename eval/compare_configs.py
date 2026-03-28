from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.run_eval import evaluate_answers_file, run_pipeline


def build_markdown_table(reports: list[dict]) -> str:
    header = "| Config | Answer Rate | Citation Coverage | Retrieval Hit@K | Reference Exact Match |"
    divider = "| --- | --- | --- | --- | --- |"
    rows = [header, divider]
    for report in reports:
        metrics = report.get("metrics", {})
        rows.append(
            "| {config} | {answer_rate} | {citation_coverage} | {retrieval_hit_at_k} | {reference_exact_match} |".format(
                config=report.get("config"),
                answer_rate=metrics.get("answer_rate"),
                citation_coverage=metrics.get("citation_coverage"),
                retrieval_hit_at_k=metrics.get("retrieval_hit_at_k"),
                reference_exact_match=metrics.get("reference_exact_match"),
            )
        )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Run and compare multiple FinaRAG configs.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/test_set"))
    parser.add_argument("--configs", default="qwen_base,qwen_rerank,qwen_ser_rerank")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
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
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.markdown_output, "w", encoding="utf-8") as file:
            file.write(build_markdown_table(reports))


if __name__ == "__main__":
    main()
