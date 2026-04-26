from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.error_analysis import summarize_error_analysis
from eval.metrics import compare_answers, load_answers_bundle, summarize_answers
from src.pipeline import Pipeline, configs, load_run_config


ROOT = Path(__file__).resolve().parents[1]


def _resolve_named_config_path(config_name: str) -> Path | None:
    candidate = Path(config_name)
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        return None

    search_candidates = []
    if candidate.is_absolute():
        search_candidates.append(candidate)
    else:
        search_candidates.append(ROOT / candidate)
        search_candidates.append(ROOT / "config" / candidate.name)

    for path in search_candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _default_reference_answers(dataset_dir: Path) -> Path | None:
    candidates = [
        dataset_dir / "answers_max_nst_o3m.json",
        dataset_dir / "answers_1st_place_o3-mini.json",
        dataset_dir / "answers_1st_place_llama_70b.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _default_debug_bundle(answers_file: Path) -> Path | None:
    debug_candidate = answers_file.with_name(answers_file.stem + "_debug" + answers_file.suffix)
    return debug_candidate if debug_candidate.exists() else None


def evaluate_answers_file(answers_file: Path, reference_answers: Path | None = None) -> dict:
    pred_answers, payload = load_answers_bundle(answers_file)
    debug_payload = None
    debug_path = _default_debug_bundle(answers_file)
    if debug_path is not None:
        _, debug_payload = load_answers_bundle(debug_path)
    metrics = summarize_answers(pred_answers)

    if reference_answers is not None and reference_answers.exists():
        ref_answers, _ = load_answers_bundle(reference_answers)
        metrics.update(compare_answers(pred_answers, ref_answers, debug_payload=debug_payload))
        error_analysis = summarize_error_analysis(pred_answers, ref_answers, debug_payload=debug_payload)
    else:
        error_analysis = summarize_error_analysis(pred_answers, [], debug_payload=debug_payload)

    return {
        "answers_file": str(answers_file),
        "details": payload.get("details"),
        "debug_bundle": str(debug_path) if debug_path is not None else None,
        "metrics": metrics,
        "error_analysis": error_analysis,
    }


def run_pipeline(dataset_dir: Path, config_name: str | None, config_path: Path | None) -> Path:
    if config_path is not None:
        run_config = load_run_config(config_path)
    elif config_name is not None:
        yaml_path = _resolve_named_config_path(config_name)
        run_config = load_run_config(yaml_path) if yaml_path is not None else configs[config_name]
    else:
        raise ValueError("Either config_name or config_path is required when --run-pipeline is used.")

    pipeline = Pipeline(dataset_dir, run_config=run_config)
    return pipeline.process_questions()


def main():
    parser = argparse.ArgumentParser(description="Evaluate FinaRAG answers files or run the pipeline and evaluate the results.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/test_set"))
    parser.add_argument("--answers-file", type=Path, default=None)
    parser.add_argument("--reference-answers", type=Path, default=None)
    parser.add_argument("--config", default="qwen_base")
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--run-pipeline", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    answers_file = args.answers_file
    if args.run_pipeline:
        answers_file = run_pipeline(args.dataset_dir, args.config, args.config_path)
    if answers_file is None:
        raise ValueError("--answers-file is required when --run-pipeline is not used.")

    reference_answers = args.reference_answers or _default_reference_answers(args.dataset_dir)
    report = evaluate_answers_file(answers_file, reference_answers)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
