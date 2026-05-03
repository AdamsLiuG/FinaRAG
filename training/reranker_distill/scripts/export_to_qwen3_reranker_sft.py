from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


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


QWEN3_RERANKER_SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
QWEN3_RERANKER_DEFAULT_INSTRUCTION = (
    "Given a Chinese financial annual report question, retrieve passages that directly "
    "answer the query with precise company, year, metric, and unit evidence."
)
QWEN3_RERANKER_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@dataclass
class BinaryLabelPolicy:
    label_source: str = "hybrid"
    positive_hard_labels: tuple[int, ...] = (2,)
    negative_hard_labels: tuple[int, ...] = (0,)
    positive_teacher_score_threshold: float = 0.85
    negative_teacher_score_threshold: float = 0.2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert reranker pointwise distill records into native Qwen3-Reranker yes/no SFT data."
    )
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--train-input-path", type=Path, default=None, help="Input train JSON/JSONL path.")
    parser.add_argument("--dev-input-path", type=Path, default=None, help="Input dev JSON/JSONL path.")
    parser.add_argument("--test-input-path", type=Path, default=None, help="Input test JSON/JSONL path.")
    parser.add_argument("--train-output-path", type=Path, default=None, help="Output train JSONL path.")
    parser.add_argument("--dev-output-path", type=Path, default=None, help="Output dev JSONL path.")
    parser.add_argument("--test-output-path", type=Path, default=None, help="Output test JSONL path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Summary stats JSON path.")
    parser.add_argument(
        "--label-source",
        choices=("hard_label", "teacher_score", "hybrid"),
        default=None,
        help="How binary yes/no targets are derived.",
    )
    parser.add_argument("--instruction", default=None, help="Instruction text injected into the reranker prompt.")
    parser.add_argument("--system-prompt", default=None, help="System prompt injected into the reranker prompt.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _parse_int_tuple(value: Any, default: Sequence[int]) -> tuple[int, ...]:
    if value in (None, ""):
        return tuple(int(item) for item in default)
    if isinstance(value, (list, tuple, set)):
        return tuple(int(item) for item in value)
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def build_qwen3_reranker_prompt_prefix(
    query: str,
    instruction: str | None = None,
    system_prompt: str | None = None,
) -> str:
    instruction_text = str(instruction or QWEN3_RERANKER_DEFAULT_INSTRUCTION).strip()
    system_text = str(system_prompt or QWEN3_RERANKER_SYSTEM_PROMPT).strip()
    query_text = str(query or "").strip()
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<Instruct>: {instruction_text}\n"
        f"<Query>: {query_text}\n"
        "<Document>: "
    )


def build_qwen3_reranker_prompt(
    query: str,
    passage: str,
    instruction: str | None = None,
    system_prompt: str | None = None,
) -> str:
    return (
        build_qwen3_reranker_prompt_prefix(
            query=query,
            instruction=instruction,
            system_prompt=system_prompt,
        )
        + str(passage or "").strip()
        + QWEN3_RERANKER_PROMPT_SUFFIX
    )


def resolve_binary_target(
    record: Dict[str, Any],
    policy: BinaryLabelPolicy,
) -> Optional[Dict[str, Any]]:
    hard_label = record.get("hard_label")
    if policy.label_source in {"hard_label", "hybrid"} and hard_label not in (None, ""):
        normalized_hard_label = int(hard_label)
        if normalized_hard_label in policy.positive_hard_labels:
            return {"target": "yes", "binary_label": 1, "target_source": "hard_label"}
        if normalized_hard_label in policy.negative_hard_labels:
            return {"target": "no", "binary_label": 0, "target_source": "hard_label"}
        if policy.label_source == "hard_label":
            return None

    teacher_score = record.get("teacher_score")
    if policy.label_source in {"teacher_score", "hybrid"} and teacher_score not in (None, ""):
        normalized_teacher_score = float(teacher_score)
        if normalized_teacher_score >= policy.positive_teacher_score_threshold:
            return {"target": "yes", "binary_label": 1, "target_source": "teacher_score"}
        if normalized_teacher_score <= policy.negative_teacher_score_threshold:
            return {"target": "no", "binary_label": 0, "target_source": "teacher_score"}

    return None


def build_qwen3_reranker_sft_record(
    record: Dict[str, Any],
    *,
    instruction: str | None = None,
    system_prompt: str | None = None,
    policy: BinaryLabelPolicy | None = None,
) -> Optional[Dict[str, Any]]:
    query = str(record.get("query") or record.get("question_text") or "").strip()
    passage = str(record.get("passage") or record.get("text") or "").strip()
    if not query or not passage:
        raise ValueError("missing_query_or_passage")

    effective_policy = policy or BinaryLabelPolicy()
    target_payload = resolve_binary_target(record, effective_policy)
    if target_payload is None:
        return None

    instruction_text = str(instruction or record.get("instruction") or QWEN3_RERANKER_DEFAULT_INSTRUCTION).strip()
    system_text = str(system_prompt or record.get("system_prompt") or QWEN3_RERANKER_SYSTEM_PROMPT).strip()
    prompt = build_qwen3_reranker_prompt(
        query=query,
        passage=passage,
        instruction=instruction_text,
        system_prompt=system_text,
    )

    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    output_record = {
        "prompt": prompt,
        "target": target_payload["target"],
        "label": int(target_payload["binary_label"]),
        "query": query,
        "passage": passage,
        "instruction": instruction_text,
        "system_prompt": system_text,
        "meta": {
            **meta,
            "target_source": target_payload["target_source"],
            "teacher_score": record.get("teacher_score"),
            "hard_label": record.get("hard_label"),
            "label_source": record.get("label_source"),
        },
    }
    if "query_id" in record and "query_id" not in output_record["meta"]:
        output_record["meta"]["query_id"] = record.get("query_id")
    return output_record


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/reranker_distill/configs/sft_export.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    policy = BinaryLabelPolicy(
        label_source=str(_coalesce(args.label_source, config.get("label_source"), "hybrid")),
        positive_hard_labels=_parse_int_tuple(config.get("positive_hard_labels"), (2,)),
        negative_hard_labels=_parse_int_tuple(config.get("negative_hard_labels"), (0,)),
        positive_teacher_score_threshold=float(config.get("positive_teacher_score_threshold", 0.85)),
        negative_teacher_score_threshold=float(config.get("negative_teacher_score_threshold", 0.2)),
    )

    return {
        "config_path": config_path,
        "policy": policy,
        "instruction": str(_coalesce(args.instruction, config.get("instruction"), QWEN3_RERANKER_DEFAULT_INSTRUCTION)),
        "system_prompt": str(_coalesce(args.system_prompt, config.get("system_prompt"), QWEN3_RERANKER_SYSTEM_PROMPT)),
        "train_input_path": resolve_repo_path(REPO_ROOT, _coalesce(args.train_input_path, config.get("train_input_path"))),
        "dev_input_path": resolve_repo_path(REPO_ROOT, _coalesce(args.dev_input_path, config.get("dev_input_path"))),
        "test_input_path": resolve_repo_path(REPO_ROOT, _coalesce(args.test_input_path, config.get("test_input_path"))),
        "train_output_path": resolve_repo_path(REPO_ROOT, _coalesce(args.train_output_path, config.get("train_output_path"))),
        "dev_output_path": resolve_repo_path(REPO_ROOT, _coalesce(args.dev_output_path, config.get("dev_output_path"))),
        "test_output_path": resolve_repo_path(REPO_ROOT, _coalesce(args.test_output_path, config.get("test_output_path"))),
        "stats_output_path": resolve_repo_path(REPO_ROOT, _coalesce(args.stats_output_path, config.get("stats_output_path"))),
    }


def _export_split(
    input_path: Path | None,
    output_path: Path | None,
    *,
    instruction: str,
    system_prompt: str,
    policy: BinaryLabelPolicy,
) -> Optional[Dict[str, Any]]:
    if input_path is None or output_path is None:
        return None
    if not input_path.exists():
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "skipped": "missing_input",
        }

    input_records = load_records(input_path)
    output_records: List[Dict[str, Any]] = []
    skipped_ambiguous = 0

    for record in input_records:
        converted = build_qwen3_reranker_sft_record(
            record,
            instruction=instruction,
            system_prompt=system_prompt,
            policy=policy,
        )
        if converted is None:
            skipped_ambiguous += 1
            continue
        output_records.append(converted)

    write_records(output_path, output_records)
    positive_count = sum(1 for record in output_records if int(record.get("label") or 0) == 1)
    negative_count = len(output_records) - positive_count

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_record_count": len(input_records),
        "export_record_count": len(output_records),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "skipped_ambiguous_count": skipped_ambiguous,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)

    split_stats = {
        "train": _export_split(
            settings["train_input_path"],
            settings["train_output_path"],
            instruction=settings["instruction"],
            system_prompt=settings["system_prompt"],
            policy=settings["policy"],
        ),
        "dev": _export_split(
            settings["dev_input_path"],
            settings["dev_output_path"],
            instruction=settings["instruction"],
            system_prompt=settings["system_prompt"],
            policy=settings["policy"],
        ),
        "test": _export_split(
            settings["test_input_path"],
            settings["test_output_path"],
            instruction=settings["instruction"],
            system_prompt=settings["system_prompt"],
            policy=settings["policy"],
        ),
    }

    stats_output_path = settings["stats_output_path"]
    if stats_output_path is not None:
        write_json(
            stats_output_path,
            {
                "build_timestamp": utc_now_iso(),
                "config_path": display_path(settings["config_path"], REPO_ROOT),
                "instruction": settings["instruction"],
                "system_prompt": settings["system_prompt"],
                "label_policy": {
                    "label_source": settings["policy"].label_source,
                    "positive_hard_labels": list(settings["policy"].positive_hard_labels),
                    "negative_hard_labels": list(settings["policy"].negative_hard_labels),
                    "positive_teacher_score_threshold": settings["policy"].positive_teacher_score_threshold,
                    "negative_teacher_score_threshold": settings["policy"].negative_teacher_score_threshold,
                },
                "splits": split_stats,
            },
        )


if __name__ == "__main__":
    main()
