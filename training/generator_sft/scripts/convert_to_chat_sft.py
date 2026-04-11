from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import (  # noqa: E402
    compact_json_dumps,
    display_path,
    load_records,
    load_yaml_mapping,
    resolve_repo_path,
    utc_now_iso,
    write_json,
    write_records,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert filtered SFT samples into chat-format records.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--input-path", type=Path, default=None, help="Filtered teacher answer JSONL path.")
    parser.add_argument("--output-path", type=Path, default=None, help="Chat-format JSON/JSONL output path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Stats JSON path.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def build_chat_record(record: Dict[str, Any]) -> Dict[str, Any]:
    assistant_response_json = record.get("assistant_response_json")
    if not assistant_response_json:
        assistant_payload = record.get("assistant_response") or {}
        assistant_response_json = compact_json_dumps(assistant_payload)

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": str(record.get("system_prompt") or "").strip(),
        },
        {
            "role": "user",
            "content": str(record.get("user_prompt") or "").strip(),
        },
        {
            "role": "assistant",
            "content": assistant_response_json,
        },
    ]
    return {
        "messages": messages,
        "meta": {
            "sample_id": record.get("sample_id"),
            "query_id": record.get("query_id"),
            "schema": record.get("schema"),
            "company_name": record.get("company_name"),
            "doc_ids": list(record.get("doc_ids", [])),
            "source": record.get("source"),
            "accepted_checks": list(record.get("accepted_checks", [])),
            "retrieval_pages": list(record.get("retrieval_pages", [])),
            "should_refuse": bool(record.get("should_refuse", False)),
        },
    }


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/generator_sft/configs/filter.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    input_path = resolve_repo_path(REPO_ROOT, _coalesce(args.input_path, config.get("filtered_input_path") or config.get("input_path")))
    output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.output_path, config.get("chat_output_path")))
    stats_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.stats_output_path, config.get("chat_stats_output_path")))
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
    records = load_records(settings["input_path"])
    chat_records = [build_chat_record(record) for record in records]
    write_records(settings["output_path"], chat_records)

    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "input_path": display_path(settings["input_path"], REPO_ROOT),
        "output_path": display_path(settings["output_path"], REPO_ROOT),
        "chat_record_count": len(chat_records),
    }
    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
