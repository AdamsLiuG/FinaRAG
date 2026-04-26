from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.export_finance_eval_bundle import export_finance_eval_bundle
from eval.finance_eval import evaluate_finance_answers
from eval.ragas_adapter import RagasRuntimeConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_DIR = ROOT / "data" / "finance_eval_benchmark_v1"
DEFAULT_PIPELINE_DATASET_DIR = ROOT / "data" / "top10_industries_2024_20each"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _default_debug_bundle(answers_file: Path) -> Path | None:
    debug_candidate = answers_file.with_name(answers_file.stem + "_debug" + answers_file.suffix)
    return debug_candidate if debug_candidate.exists() else None


def _default_export_paths(answers_file: Path) -> tuple[Path, Path, Path]:
    pred_answers_out = answers_file.with_name(answers_file.stem + ".finance_eval.json")
    debug_out = answers_file.with_name(answers_file.stem + ".finance_eval_debug.json")
    report_out = answers_file.with_name(answers_file.stem + ".finance_eval.report.json")
    return pred_answers_out, debug_out, report_out


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


def _resolve_run_config(config_name: str | None, config_path: Path | None):
    from src.pipeline import configs, load_run_config

    if config_path is not None:
        return load_run_config(config_path)
    if config_name is not None:
        yaml_path = _resolve_named_config_path(config_name)
        if yaml_path is not None:
            return load_run_config(yaml_path)
        return configs[config_name]
    raise ValueError("Either config_name or config_path is required when running the pipeline.")


def _build_ragas_config_from_args(args: argparse.Namespace) -> RagasRuntimeConfig:
    env_ragas_config = RagasRuntimeConfig.from_env()
    return RagasRuntimeConfig(
        enabled=False if args.disable_ragas else env_ragas_config.enabled,
        llm_provider=args.ragas_llm_provider or env_ragas_config.llm_provider,
        llm_model=args.ragas_llm_model or env_ragas_config.llm_model,
        llm_base_url=args.ragas_llm_base_url or env_ragas_config.llm_base_url,
        llm_api_key=args.ragas_llm_api_key or env_ragas_config.llm_api_key,
        llm_timeout=args.ragas_llm_timeout if args.ragas_llm_timeout is not None else env_ragas_config.llm_timeout,
        llm_max_retries=args.ragas_llm_max_retries if args.ragas_llm_max_retries is not None else env_ragas_config.llm_max_retries,
        llm_adapter=args.ragas_llm_adapter or env_ragas_config.llm_adapter,
        llm_force_stream=True if args.ragas_llm_force_stream else env_ragas_config.llm_force_stream,
        embedding_provider=args.ragas_embedding_provider or env_ragas_config.embedding_provider,
        embedding_model=args.ragas_embedding_model or env_ragas_config.embedding_model,
        embedding_device=args.ragas_embedding_device or env_ragas_config.embedding_device,
        embedding_base_url=args.ragas_embedding_base_url or env_ragas_config.embedding_base_url,
        embedding_api_key=args.ragas_embedding_api_key or env_ragas_config.embedding_api_key,
        context_limit=args.ragas_context_limit if args.ragas_context_limit is not None else env_ragas_config.context_limit,
    )


def run_pipeline_for_benchmark(
    *,
    dataset_dir: Path,
    questions_file: Path,
    config_name: str | None,
    config_path: Path | None,
    output_path: Path | None = None,
    resume: bool = False,
    retry_errors: bool = True,
) -> Path:
    from src.pipeline import Pipeline

    run_config = _resolve_run_config(config_name, config_path)
    resolved_questions_file = questions_file if questions_file.is_absolute() else ROOT / questions_file
    pipeline = Pipeline(dataset_dir, questions_file_name=str(resolved_questions_file), run_config=run_config)
    return pipeline.process_questions(
        output_path=output_path,
        resume=resume,
        retry_errors=retry_errors,
    )


def run_finance_benchmark(
    *,
    pipeline_dataset_dir: Path,
    questions_file: Path,
    gold_answers_file: Path,
    answers_file: Path | None = None,
    debug_file: Path | None = None,
    config_name: str | None = "qwen_base",
    config_path: Path | None = None,
    pred_answers_out: Path | None = None,
    eval_debug_out: Path | None = None,
    report_out: Path | None = None,
    use_embedding_similarity: bool = False,
    include_cases: bool = True,
    ragas_config: RagasRuntimeConfig | None = None,
    resume: bool = False,
    resume_file: Path | None = None,
    retry_errors: bool = True,
) -> Dict[str, Any]:
    if resume and answers_file is not None:
        raise ValueError("resume cannot be used together with answers_file because answers_file skips the pipeline.")
    if resume_file is not None and answers_file is not None:
        raise ValueError("resume_file cannot be used together with answers_file because answers_file skips the pipeline.")

    raw_answers_file = answers_file
    if raw_answers_file is None:
        raw_answers_file = run_pipeline_for_benchmark(
            dataset_dir=pipeline_dataset_dir,
            questions_file=questions_file,
            config_name=config_name,
            config_path=config_path,
            output_path=resume_file,
            resume=resume,
            retry_errors=retry_errors,
        )

    raw_debug_file = debug_file or _default_debug_bundle(raw_answers_file)
    default_pred_out, default_eval_debug_out, default_report_out = _default_export_paths(raw_answers_file)
    pred_answers_out = pred_answers_out or default_pred_out
    eval_debug_out = eval_debug_out or default_eval_debug_out
    report_out = report_out or default_report_out

    export_summary = export_finance_eval_bundle(
        questions_file=questions_file,
        answers_file=raw_answers_file,
        pred_answers_out=pred_answers_out,
        debug_out=eval_debug_out,
        debug_file=raw_debug_file,
    )

    evaluation_report = evaluate_finance_answers(
        questions_file=questions_file,
        gold_answers_file=gold_answers_file,
        pred_answers_file=pred_answers_out,
        debug_file=eval_debug_out,
        use_embedding_similarity=use_embedding_similarity,
        include_cases=include_cases,
        ragas_config=ragas_config,
    )
    _write_json(report_out, evaluation_report)

    return {
        "pipeline_dataset_dir": str(pipeline_dataset_dir),
        "questions_file": str(questions_file),
        "gold_answers_file": str(gold_answers_file),
        "raw_answers_file": str(raw_answers_file),
        "raw_debug_file": str(raw_debug_file) if raw_debug_file is not None else None,
        "pred_answers_file": str(pred_answers_out),
        "eval_debug_file": str(eval_debug_out),
        "evaluation_report_file": str(report_out),
        "export_summary": export_summary,
        "summary": evaluation_report.get("summary", {}),
        "ragas": evaluation_report.get("ragas", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the FinaRAG pipeline, export finance_eval bundles, and execute formal benchmark scoring in one command."
    )
    parser.add_argument("--pipeline-dataset-dir", type=Path, default=DEFAULT_PIPELINE_DATASET_DIR)
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_BENCHMARK_DIR / "questions.json")
    parser.add_argument("--gold-answers-file", type=Path, default=DEFAULT_BENCHMARK_DIR / "answers_gold.json")
    parser.add_argument("--answers-file", type=Path, default=None)
    parser.add_argument("--debug-file", type=Path, default=None)
    parser.add_argument("--config", default="qwen_base")
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the pipeline from an existing raw answers/debug bundle instead of starting a fresh answers file.",
    )
    parser.add_argument(
        "--resume-file",
        type=Path,
        default=None,
        help="Raw answers JSON path to resume into. Without --resume, this becomes the fresh raw answers output path.",
    )
    parser.add_argument("--pred-answers-out", type=Path, default=None)
    parser.add_argument("--eval-debug-out", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--use-embedding-similarity", action="store_true")
    parser.add_argument("--no-case-details", action="store_true")
    parser.add_argument("--disable-ragas", action="store_true")
    parser.add_argument("--ragas-llm-provider", default=None)
    parser.add_argument("--ragas-llm-model", default=None)
    parser.add_argument("--ragas-llm-base-url", default=None)
    parser.add_argument("--ragas-llm-api-key", default=None)
    parser.add_argument("--ragas-llm-timeout", type=float, default=None)
    parser.add_argument("--ragas-llm-max-retries", type=int, default=None)
    parser.add_argument("--ragas-llm-adapter", default=None)
    parser.add_argument("--ragas-llm-force-stream", action="store_true")
    parser.add_argument("--ragas-embedding-provider", default=None)
    parser.add_argument("--ragas-embedding-model", default=None)
    parser.add_argument("--ragas-embedding-device", default=None)
    parser.add_argument("--ragas-embedding-base-url", default=None)
    parser.add_argument("--ragas-embedding-api-key", default=None)
    parser.add_argument("--ragas-context-limit", type=int, default=None)
    args = parser.parse_args()

    ragas_config = _build_ragas_config_from_args(args)
    report = run_finance_benchmark(
        pipeline_dataset_dir=args.pipeline_dataset_dir,
        questions_file=args.questions_file,
        gold_answers_file=args.gold_answers_file,
        answers_file=args.answers_file,
        debug_file=args.debug_file,
        config_name=args.config,
        config_path=args.config_path,
        pred_answers_out=args.pred_answers_out,
        eval_debug_out=args.eval_debug_out,
        report_out=args.report_out,
        use_embedding_similarity=args.use_embedding_similarity,
        include_cases=not args.no_case_details,
        ragas_config=ragas_config,
        resume=args.resume,
        resume_file=args.resume_file,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
