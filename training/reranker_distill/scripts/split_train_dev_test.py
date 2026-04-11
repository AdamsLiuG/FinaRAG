from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import (  # noqa: E402
    build_split_group_key,
    deterministic_split_for_key,
    display_path,
    load_records,
    load_yaml_mapping,
    resolve_repo_path,
    utc_now_iso,
    write_json,
    write_records,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministically split reranker pointwise data.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--input-path", type=Path, default=None, help="Pointwise input path.")
    parser.add_argument("--train-output-path", type=Path, default=None, help="Train split output path.")
    parser.add_argument("--dev-output-path", type=Path, default=None, help="Dev split output path.")
    parser.add_argument("--test-output-path", type=Path, default=None, help="Test split output path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Stats JSON path.")
    parser.add_argument("--dev-ratio", type=float, default=None, help="Dev ratio.")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test ratio.")
    parser.add_argument("--split-salt", default=None, help="Hash salt for deterministic split.")
    parser.add_argument("--group-fields", nargs="*", default=None, help="Preferred split grouping fields.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def split_records(
    records: Iterable[Dict[str, Any]],
    *,
    dev_ratio: float,
    test_ratio: float,
    split_salt: str,
    group_fields: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    splits = {"train": [], "dev": [], "test": []}
    for record in records:
        group_key = build_split_group_key(record, group_fields)
        split_name = deterministic_split_for_key(
            group_key,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            salt=split_salt,
        )
        splits[split_name].append(record)
    return splits


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/reranker_distill/configs/split.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    input_path = resolve_repo_path(REPO_ROOT, _coalesce(args.input_path, config.get("input_path")))
    train_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.train_output_path, config.get("train_output_path")))
    dev_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.dev_output_path, config.get("dev_output_path")))
    test_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.test_output_path, config.get("test_output_path")))
    stats_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.stats_output_path, config.get("stats_output_path")))
    if input_path is None or train_output_path is None or dev_output_path is None or test_output_path is None or stats_output_path is None:
        raise ValueError("input/train/dev/test/stats paths are required.")

    return {
        "config_path": config_path,
        "input_path": input_path,
        "train_output_path": train_output_path,
        "dev_output_path": dev_output_path,
        "test_output_path": test_output_path,
        "stats_output_path": stats_output_path,
        "dev_ratio": float(_coalesce(args.dev_ratio, config.get("dev_ratio"), 0.1)),
        "test_ratio": float(_coalesce(args.test_ratio, config.get("test_ratio"), 0.1)),
        "split_salt": str(_coalesce(args.split_salt, config.get("split_salt"), "finarag_reranker_v1")),
        "group_fields": list(_coalesce(args.group_fields, config.get("group_fields"), ["query_id"]) or []),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)
    records = load_records(settings["input_path"])
    splits = split_records(
        records,
        dev_ratio=settings["dev_ratio"],
        test_ratio=settings["test_ratio"],
        split_salt=settings["split_salt"],
        group_fields=settings["group_fields"],
    )

    write_records(settings["train_output_path"], splits["train"])
    write_records(settings["dev_output_path"], splits["dev"])
    write_records(settings["test_output_path"], splits["test"])

    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "input_path": display_path(settings["input_path"], REPO_ROOT),
        "train_output_path": display_path(settings["train_output_path"], REPO_ROOT),
        "dev_output_path": display_path(settings["dev_output_path"], REPO_ROOT),
        "test_output_path": display_path(settings["test_output_path"], REPO_ROOT),
        "dev_ratio": settings["dev_ratio"],
        "test_ratio": settings["test_ratio"],
        "split_salt": settings["split_salt"],
        "group_fields": settings["group_fields"],
        "train_count": len(splits["train"]),
        "dev_count": len(splits["dev"]),
        "test_count": len(splits["test"]),
    }
    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
