from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import (  # noqa: E402
    display_path,
    load_records,
    load_yaml_mapping,
    resolve_repo_path,
    utc_now_iso,
    write_json,
    write_records,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export pointwise reranker labels into trainer-ready records.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--input-path", type=Path, default=None, help="Pointwise labels input path.")
    parser.add_argument("--output-path", type=Path, default=None, help="Trainer export output path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Stats JSON path.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def build_export_record(record: Dict[str, Any]) -> Dict[str, Any]:
    query = record.get("query") or record.get("question_text")
    passage = record.get("passage") or record.get("text")
    if not query or not passage:
        raise ValueError("missing_query_or_passage")

    return {
        "query": query,
        "passage": passage,
        "teacher_score": round(float(record.get("teacher_score") or 0.0), 4),
        "hard_label": int(record.get("hard_label") or 0),
        "meta": {
            "pair_id": record.get("pair_id"),
            "query_id": record.get("query_id"),
            "candidate_id": record.get("candidate_id"),
            "doc_id": record.get("doc_id"),
            "company_name": record.get("company_name"),
            "page": record.get("page"),
            "schema": record.get("schema"),
            "teacher_rank": int(record.get("teacher_rank") or 0),
            "base_score": round(float(record.get("base_score") or 0.0), 4),
            "section_name": record.get("section_name"),
            "retrieval_sources": list(record.get("retrieval_sources", [])),
            "is_hard_negative": bool(record.get("is_hard_negative", False)),
            "label_source": list(record.get("label_source", [])),
        },
    }


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/reranker_distill/configs/train.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    input_path = resolve_repo_path(REPO_ROOT, _coalesce(args.input_path, config.get("export_input_path")))
    output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.output_path, config.get("export_output_path")))
    stats_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.stats_output_path, config.get("export_stats_output_path")))
    if input_path is None or output_path is None or stats_output_path is None:
        raise ValueError("input/output/stats paths are required.")

    return {
        "config_path": config_path,
        "input_path": input_path,
        "output_path": output_path,
        "stats_output_path": stats_output_path,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)
    input_records = load_records(settings["input_path"])
    output_records = [build_export_record(record) for record in input_records]
    write_records(settings["output_path"], output_records)

    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "input_path": display_path(settings["input_path"], REPO_ROOT),
        "output_path": display_path(settings["output_path"], REPO_ROOT),
        "export_record_count": len(output_records),
    }
    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
