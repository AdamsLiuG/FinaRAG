from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import display_path, load_yaml_mapping, resolve_repo_path, utc_now_iso, write_json  # noqa: E402


_STAGE_ORDER = ["collect", "score", "label", "split", "export"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full reranker distillation dataset build pipeline with a local /v1/rerank-compatible teacher."
    )
    parser.add_argument("--data-config-path", type=Path, default=None, help="Data-build YAML config path.")
    parser.add_argument("--split-config-path", type=Path, default=None, help="Split YAML config path.")
    parser.add_argument("--export-config-path", type=Path, default=None, help="Export YAML config path.")
    parser.add_argument("--python-bin", default=None, help="Python executable used to run stage scripts.")
    parser.add_argument("--summary-output-path", type=Path, default=None, help="Pipeline summary JSON path.")
    parser.add_argument("--start-stage", choices=_STAGE_ORDER, default="collect", help="First stage to run.")
    parser.add_argument("--end-stage", choices=_STAGE_ORDER, default="export", help="Last stage to run.")
    parser.add_argument("--resume", action="store_true", help="Resume collect/score stages from existing outputs.")
    parser.add_argument("--no-resume", action="store_true", help="Force collect/score stages to rebuild from scratch.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _default_data_config_path() -> Path:
    preferred = REPO_ROOT / "training/reranker_distill/configs/data_build.local_vllm_reranker.example.yaml"
    if preferred.exists():
        return preferred
    return REPO_ROOT / "training/reranker_distill/configs/data_build.example.yaml"


def _family_index(stage_family: str) -> int:
    return _STAGE_ORDER.index(stage_family)


def build_stage_commands(settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    python_bin = str(settings["python_bin"])
    data_config_path = str(settings["data_config_path"])
    split_config_path = str(settings["split_config_path"])
    export_config_path = str(settings["export_config_path"])

    resume_args: List[str] = []
    if settings["resume"] is True:
        resume_args = ["--resume"]
    elif settings["resume"] is False:
        resume_args = ["--no-resume"]

    commands = [
        {
            "name": "collect",
            "family": "collect",
            "argv": [
                python_bin,
                str(REPO_ROOT / "training/reranker_distill/scripts/collect_candidate_pool.py"),
                "--config-path",
                data_config_path,
                *resume_args,
            ],
        },
        {
            "name": "score",
            "family": "score",
            "argv": [
                python_bin,
                str(REPO_ROOT / "training/reranker_distill/scripts/score_with_teacher_reranker.py"),
                "--config-path",
                data_config_path,
                *resume_args,
            ],
        },
        {
            "name": "label",
            "family": "label",
            "argv": [
                python_bin,
                str(REPO_ROOT / "training/reranker_distill/scripts/build_pointwise_labels.py"),
                "--config-path",
                data_config_path,
            ],
        },
        {
            "name": "split",
            "family": "split",
            "argv": [
                python_bin,
                str(REPO_ROOT / "training/reranker_distill/scripts/split_train_dev_test.py"),
                "--config-path",
                split_config_path,
            ],
        },
        {
            "name": "export_train",
            "family": "export",
            "argv": [
                python_bin,
                str(REPO_ROOT / "training/reranker_distill/scripts/export_for_trainer.py"),
                "--config-path",
                export_config_path,
                "--input-path",
                str(settings["export_train_input_path"]),
                "--output-path",
                str(settings["export_train_output_path"]),
                "--stats-output-path",
                str(settings["export_train_stats_output_path"]),
            ],
        },
        {
            "name": "export_dev",
            "family": "export",
            "argv": [
                python_bin,
                str(REPO_ROOT / "training/reranker_distill/scripts/export_for_trainer.py"),
                "--config-path",
                export_config_path,
                "--input-path",
                str(settings["export_dev_input_path"]),
                "--output-path",
                str(settings["export_dev_output_path"]),
                "--stats-output-path",
                str(settings["export_dev_stats_output_path"]),
            ],
        },
        {
            "name": "export_test",
            "family": "export",
            "argv": [
                python_bin,
                str(REPO_ROOT / "training/reranker_distill/scripts/export_for_trainer.py"),
                "--config-path",
                export_config_path,
                "--input-path",
                str(settings["export_test_input_path"]),
                "--output-path",
                str(settings["export_test_output_path"]),
                "--stats-output-path",
                str(settings["export_test_stats_output_path"]),
            ],
        },
    ]

    start_index = _family_index(settings["start_stage"])
    end_index = _family_index(settings["end_stage"])
    return [
        command
        for command in commands
        if start_index <= _family_index(command["family"]) <= end_index
    ]


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    data_config_path = resolve_repo_path(REPO_ROOT, args.data_config_path or _default_data_config_path())
    split_config_path = resolve_repo_path(
        REPO_ROOT,
        args.split_config_path or REPO_ROOT / "training/reranker_distill/configs/split.example.yaml",
    )
    export_config_path = resolve_repo_path(
        REPO_ROOT,
        args.export_config_path or REPO_ROOT / "training/reranker_distill/configs/export.example.yaml",
    )
    if data_config_path is None or split_config_path is None or export_config_path is None:
        raise ValueError("data/split/export config paths are required.")

    data_config = load_yaml_mapping(data_config_path)
    split_config = load_yaml_mapping(split_config_path)
    export_config = load_yaml_mapping(export_config_path)

    if _family_index(args.start_stage) > _family_index(args.end_stage):
        raise ValueError("start-stage must be earlier than or equal to end-stage.")

    if args.resume:
        resume = True
    elif args.no_resume:
        resume = False
    else:
        resume_config = data_config.get("resume")
        resume = bool(resume_config) if resume_config is not None else True

    summary_output_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(
            args.summary_output_path,
            data_config.get("pipeline_stats_output_path"),
            "training/reranker_distill/manifests/pipeline_run_summary.json",
        ),
    )

    train_input_path = resolve_repo_path(REPO_ROOT, export_config.get("export_input_path"))
    train_output_path = resolve_repo_path(REPO_ROOT, export_config.get("export_output_path"))
    train_stats_path = resolve_repo_path(REPO_ROOT, export_config.get("export_stats_output_path"))
    dev_input_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(export_config.get("dev_export_input_path"), split_config.get("dev_output_path")),
    )
    dev_output_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(export_config.get("dev_export_output_path"), "training/reranker_distill/processed/pointwise_dev.jsonl"),
    )
    dev_stats_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(export_config.get("dev_export_stats_output_path"), "training/reranker_distill/manifests/export_dev_stats.json"),
    )
    test_input_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(export_config.get("test_export_input_path"), split_config.get("test_output_path")),
    )
    test_output_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(export_config.get("test_export_output_path"), "training/reranker_distill/processed/pointwise_test.jsonl"),
    )
    test_stats_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(export_config.get("test_export_stats_output_path"), "training/reranker_distill/manifests/export_test_stats.json"),
    )

    required_paths = [
        summary_output_path,
        train_input_path,
        train_output_path,
        train_stats_path,
        dev_input_path,
        dev_output_path,
        dev_stats_path,
        test_input_path,
        test_output_path,
        test_stats_path,
    ]
    if any(path is None for path in required_paths):
        raise ValueError("summary and export train/dev/test paths must all resolve successfully.")

    return {
        "python_bin": args.python_bin or sys.executable,
        "data_config_path": data_config_path,
        "split_config_path": split_config_path,
        "export_config_path": export_config_path,
        "summary_output_path": summary_output_path,
        "start_stage": args.start_stage,
        "end_stage": args.end_stage,
        "resume": resume,
        "dry_run": bool(args.dry_run),
        "export_train_input_path": train_input_path,
        "export_train_output_path": train_output_path,
        "export_train_stats_output_path": train_stats_path,
        "export_dev_input_path": dev_input_path,
        "export_dev_output_path": dev_output_path,
        "export_dev_stats_output_path": dev_stats_path,
        "export_test_input_path": test_input_path,
        "export_test_output_path": test_output_path,
        "export_test_stats_output_path": test_stats_path,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)
    commands = build_stage_commands(settings)

    summary = {
        "build_timestamp": utc_now_iso(),
        "data_config_path": display_path(settings["data_config_path"], REPO_ROOT),
        "split_config_path": display_path(settings["split_config_path"], REPO_ROOT),
        "export_config_path": display_path(settings["export_config_path"], REPO_ROOT),
        "python_bin": str(settings["python_bin"]),
        "start_stage": settings["start_stage"],
        "end_stage": settings["end_stage"],
        "resume": settings["resume"],
        "dry_run": settings["dry_run"],
        "stages": [],
    }

    for command in commands:
        stage_summary = {
            "name": command["name"],
            "family": command["family"],
            "argv": command["argv"],
        }
        started_at = time.perf_counter()
        if settings["dry_run"]:
            stage_summary["status"] = "dry_run"
            stage_summary["duration_seconds"] = 0.0
            summary["stages"].append(stage_summary)
            continue

        completed = subprocess.run(command["argv"], cwd=str(REPO_ROOT), check=False)
        stage_summary["returncode"] = int(completed.returncode)
        stage_summary["duration_seconds"] = round(time.perf_counter() - started_at, 3)
        stage_summary["status"] = "ok" if completed.returncode == 0 else "failed"
        summary["stages"].append(stage_summary)
        if completed.returncode != 0:
            summary["status"] = "failed"
            write_json(settings["summary_output_path"], summary)
            raise SystemExit(completed.returncode)

    summary["status"] = "dry_run" if settings["dry_run"] else "ok"
    write_json(settings["summary_output_path"], summary)


if __name__ == "__main__":
    main()
