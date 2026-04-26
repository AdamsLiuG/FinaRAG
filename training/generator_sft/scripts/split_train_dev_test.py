from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    parser = argparse.ArgumentParser(description="Deterministically split generator SFT records into train/dev/test.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--input-path", type=Path, default=None, help="Chat-format input JSON/JSONL path.")
    parser.add_argument("--core-output-path", type=Path, default=None, help="Output path for filtered core_single_doc records.")
    parser.add_argument("--aux-output-path", type=Path, default=None, help="Output path for aux_multidoc/auxiliary records.")
    parser.add_argument("--split-manifest-output-path", type=Path, default=None, help="Output JSONL path for split manifest.")
    parser.add_argument("--train-output-path", type=Path, default=None, help="Train split output path.")
    parser.add_argument("--dev-output-path", type=Path, default=None, help="Dev split output path.")
    parser.add_argument("--test-output-path", type=Path, default=None, help="Test split output path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Stats JSON path.")
    parser.add_argument("--dev-ratio", type=float, default=None, help="Dev ratio.")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test ratio.")
    parser.add_argument("--split-salt", default=None, help="Hash salt for deterministic splitting.")
    parser.add_argument("--group-fields", nargs="*", default=None, help="Preferred grouping fields.")
    parser.add_argument("--core-schemas", nargs="*", default=None, help="Schemas eligible for core_single_doc split.")
    parser.add_argument("--core-only", action="store_true", help="Only split core_single_doc records into train/dev/test.")
    parser.add_argument("--no-core-only", action="store_true", help="Disable core_single_doc pre-filtering even if enabled in config.")
    parser.add_argument("--strict-doc-holdout", action="store_true", help="Force train/dev/test grouping by single doc_id.")
    parser.add_argument("--no-strict-doc-holdout", action="store_true", help="Disable strict doc holdout.")
    parser.add_argument("--llamafactory-dataset-dir", type=Path, default=None, help="Optional LLaMA Factory export directory.")
    parser.add_argument("--llamafactory-dataset-prefix", default=None, help="Dataset name prefix for LLaMA Factory export.")
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


def _record_meta(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("meta")
    return meta if isinstance(meta, dict) else {}


def _record_value(record: Dict[str, Any], field_name: str) -> Any:
    if field_name in record and record[field_name] not in (None, "", []):
        return record[field_name]
    meta = _record_meta(record)
    return meta.get(field_name)


def _stringify_doc_ids(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item) for item in value if item not in (None, "")]


def _primary_doc_id(record: Dict[str, Any]) -> str | None:
    doc_ids = _stringify_doc_ids(_record_value(record, "doc_ids"))
    return doc_ids[0] if len(doc_ids) == 1 else None


def _is_core_single_doc(record: Dict[str, Any], core_schemas: List[str]) -> bool:
    schema = str(_record_value(record, "schema") or "")
    doc_ids = _stringify_doc_ids(_record_value(record, "doc_ids"))
    return len(doc_ids) == 1 and schema in set(core_schemas)


def _build_manifest_record(
    record: Dict[str, Any],
    *,
    split_name: str,
    split_family: str,
    reason: str,
    group_key: str,
    split_salt: str,
) -> Dict[str, Any]:
    return {
        "sample_id": _record_value(record, "sample_id"),
        "query_id": _record_value(record, "query_id"),
        "schema": _record_value(record, "schema"),
        "company_name": _record_value(record, "company_name"),
        "doc_ids": _stringify_doc_ids(_record_value(record, "doc_ids")),
        "primary_doc_id": _primary_doc_id(record),
        "split_name": split_name,
        "split_family": split_family,
        "reason": reason,
        "group_key": group_key,
        "split_salt": split_salt,
        "build_timestamp": utc_now_iso(),
    }


def _validate_doc_holdout(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    doc_sets = {
        split_name: {
            _primary_doc_id(record)
            for record in records
            if _primary_doc_id(record)
        }
        for split_name, records in splits.items()
    }
    train_dev = len(doc_sets["train"] & doc_sets["dev"])
    train_test = len(doc_sets["train"] & doc_sets["test"])
    dev_test = len(doc_sets["dev"] & doc_sets["test"])
    if train_dev or train_test or dev_test:
        raise ValueError(
            "Strict doc_id holdout violated: "
            f"train/dev={train_dev}, train/test={train_test}, dev/test={dev_test}"
        )
    return {
        "train_dev_doc_overlap": train_dev,
        "train_test_doc_overlap": train_test,
        "dev_test_doc_overlap": dev_test,
        "train_doc_count": len(doc_sets["train"]),
        "dev_doc_count": len(doc_sets["dev"]),
        "test_doc_count": len(doc_sets["test"]),
    }


def _core_group_key(record: Dict[str, Any], *, strict_doc_holdout: bool, group_fields: List[str]) -> str:
    if strict_doc_holdout:
        primary_doc_id = _primary_doc_id(record)
        if not primary_doc_id:
            raise ValueError("strict_doc_holdout requires exactly one doc_id per core_single_doc record.")
        return primary_doc_id
    return build_split_group_key(record, group_fields)


def _sanitize_for_llamafactory(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (str, float)):
        return value
    if isinstance(value, int):
        if abs(value) > (2**63 - 1):
            return str(value)
        return value
    if isinstance(value, list):
        return [_sanitize_for_llamafactory(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_for_llamafactory(item) for key, item in value.items()}
    return value


def _write_llamafactory_exports(dataset_dir: Path, dataset_prefix: str, splits: Dict[str, List[Dict[str, Any]]]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = {}
    for split_name, records in splits.items():
        file_name = f"{dataset_prefix}_{split_name}.json"
        sanitized_records = [
            {
                "messages": _sanitize_for_llamafactory(list(record.get("messages") or [])),
            }
            for record in records
        ]
        write_records(dataset_dir / file_name, sanitized_records)
        dataset_info[f"{dataset_prefix}_{split_name}"] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }

    (dataset_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/generator_sft/configs/split.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    input_path = resolve_repo_path(REPO_ROOT, _coalesce(args.input_path, config.get("input_path")))
    core_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.core_output_path, config.get("core_output_path")))
    aux_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.aux_output_path, config.get("aux_output_path")))
    split_manifest_output_path = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.split_manifest_output_path, config.get("split_manifest_output_path")),
    )
    train_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.train_output_path, config.get("train_output_path")))
    dev_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.dev_output_path, config.get("dev_output_path")))
    test_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.test_output_path, config.get("test_output_path")))
    stats_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.stats_output_path, config.get("stats_output_path")))
    llamafactory_dataset_dir = resolve_repo_path(
        REPO_ROOT,
        _coalesce(args.llamafactory_dataset_dir, config.get("llamafactory_dataset_dir")),
    )
    if input_path is None or train_output_path is None or dev_output_path is None or test_output_path is None or stats_output_path is None:
        raise ValueError("input/train/dev/test/stats paths are required.")

    core_only_config = config.get("core_only")
    core_only = True if args.core_only else False if args.no_core_only else bool(core_only_config if core_only_config is not None else False)
    strict_doc_holdout_config = config.get("strict_doc_holdout")
    strict_doc_holdout = (
        True if args.strict_doc_holdout
        else False if args.no_strict_doc_holdout
        else bool(strict_doc_holdout_config if strict_doc_holdout_config is not None else core_only)
    )

    group_fields = list(
        _coalesce(args.group_fields, config.get("group_fields"), ["doc_ids", "company_name", "query_id"]) or []
    )
    core_schemas = [
        str(value)
        for value in (_coalesce(args.core_schemas, config.get("core_schemas"), ["name", "number", "boolean"]) or [])
        if str(value).strip()
    ]

    if core_only and core_output_path is None:
        core_output_path = input_path.with_name("core_single_doc.chat.v2.jsonl")
    if core_only and aux_output_path is None:
        aux_output_path = input_path.with_name("aux_multidoc.chat.v2.jsonl")
    if core_only and split_manifest_output_path is None:
        split_manifest_output_path = (stats_output_path.parent / "split_manifest.v2.jsonl").resolve()

    return {
        "config_path": config_path,
        "input_path": input_path,
        "core_output_path": core_output_path,
        "aux_output_path": aux_output_path,
        "split_manifest_output_path": split_manifest_output_path,
        "train_output_path": train_output_path,
        "dev_output_path": dev_output_path,
        "test_output_path": test_output_path,
        "stats_output_path": stats_output_path,
        "dev_ratio": float(_coalesce(args.dev_ratio, config.get("dev_ratio"), 0.1)),
        "test_ratio": float(_coalesce(args.test_ratio, config.get("test_ratio"), 0.1)),
        "split_salt": str(_coalesce(args.split_salt, config.get("split_salt"), "finarag_generator_v1")),
        "group_fields": group_fields,
        "core_only": core_only,
        "strict_doc_holdout": strict_doc_holdout,
        "core_schemas": core_schemas,
        "llamafactory_dataset_dir": llamafactory_dataset_dir,
        "llamafactory_dataset_prefix": str(
            _coalesce(args.llamafactory_dataset_prefix, config.get("llamafactory_dataset_prefix"), "finarag_generator")
        ),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)
    records = load_records(settings["input_path"])

    manifest_records: List[Dict[str, Any]] = []
    aux_reason_counter = Counter()

    if settings["core_only"]:
        core_records: List[Dict[str, Any]] = []
        aux_records: List[Dict[str, Any]] = []
        for record in records:
            if _is_core_single_doc(record, settings["core_schemas"]):
                group_key = _core_group_key(
                    record,
                    strict_doc_holdout=settings["strict_doc_holdout"],
                    group_fields=settings["group_fields"],
                )
                core_records.append(record)
                manifest_records.append(
                    _build_manifest_record(
                        record,
                        split_name="pending",
                        split_family="core_single_doc",
                        reason="core_single_doc",
                        group_key=group_key,
                        split_salt=settings["split_salt"],
                    )
                )
                continue

            doc_ids = _stringify_doc_ids(_record_value(record, "doc_ids"))
            reason = "multidoc" if len(doc_ids) > 1 else "non_core_schema_or_missing_doc"
            aux_records.append(record)
            aux_reason_counter[reason] += 1
            manifest_records.append(
                _build_manifest_record(
                    record,
                    split_name="aux",
                    split_family="aux_multidoc",
                    reason=reason,
                    group_key=f"aux:{_record_value(record, 'query_id') or _record_value(record, 'sample_id') or ''}",
                    split_salt=settings["split_salt"],
                )
            )

        splits = {"train": [], "dev": [], "test": []}
        for record in core_records:
            group_key = _core_group_key(
                record,
                strict_doc_holdout=settings["strict_doc_holdout"],
                group_fields=settings["group_fields"],
            )
            split_name = deterministic_split_for_key(
                group_key,
                dev_ratio=settings["dev_ratio"],
                test_ratio=settings["test_ratio"],
                salt=settings["split_salt"],
            )
            splits[split_name].append(record)

        manifest_index = {
            (
                str(item.get("sample_id") or ""),
                str(item.get("query_id") or ""),
            ): item
            for item in manifest_records
        }
        for split_name, split_records_list in splits.items():
            for record in split_records_list:
                key = (
                    str(_record_value(record, "sample_id") or ""),
                    str(_record_value(record, "query_id") or ""),
                )
                manifest_index[key]["split_name"] = split_name

        if settings["core_output_path"] is not None:
            write_records(settings["core_output_path"], core_records)
        if settings["aux_output_path"] is not None:
            write_records(settings["aux_output_path"], aux_records)
        if settings["split_manifest_output_path"] is not None:
            write_records(settings["split_manifest_output_path"], manifest_records)
        doc_holdout_stats = _validate_doc_holdout(splits) if settings["strict_doc_holdout"] else {}
    else:
        splits = split_records(
            records,
            dev_ratio=settings["dev_ratio"],
            test_ratio=settings["test_ratio"],
            split_salt=settings["split_salt"],
            group_fields=settings["group_fields"],
        )
        for split_name, split_records_list in splits.items():
            for record in split_records_list:
                group_key = build_split_group_key(record, settings["group_fields"])
                manifest_records.append(
                    _build_manifest_record(
                        record,
                        split_name=split_name,
                        split_family="legacy_all",
                        reason="legacy_split",
                        group_key=group_key,
                        split_salt=settings["split_salt"],
                    )
                )
        if settings["split_manifest_output_path"] is not None:
            write_records(settings["split_manifest_output_path"], manifest_records)
        doc_holdout_stats = {}

    write_records(settings["train_output_path"], splits["train"])
    write_records(settings["dev_output_path"], splits["dev"])
    write_records(settings["test_output_path"], splits["test"])

    if settings["llamafactory_dataset_dir"] is not None:
        _write_llamafactory_exports(
            settings["llamafactory_dataset_dir"],
            settings["llamafactory_dataset_prefix"],
            splits,
        )

    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "input_path": display_path(settings["input_path"], REPO_ROOT),
        "core_output_path": display_path(settings["core_output_path"], REPO_ROOT),
        "aux_output_path": display_path(settings["aux_output_path"], REPO_ROOT),
        "split_manifest_output_path": display_path(settings["split_manifest_output_path"], REPO_ROOT),
        "train_output_path": display_path(settings["train_output_path"], REPO_ROOT),
        "dev_output_path": display_path(settings["dev_output_path"], REPO_ROOT),
        "test_output_path": display_path(settings["test_output_path"], REPO_ROOT),
        "dev_ratio": settings["dev_ratio"],
        "test_ratio": settings["test_ratio"],
        "split_salt": settings["split_salt"],
        "group_fields": settings["group_fields"],
        "core_only": settings["core_only"],
        "strict_doc_holdout": settings["strict_doc_holdout"],
        "core_schemas": settings["core_schemas"],
        "train_count": len(splits["train"]),
        "dev_count": len(splits["dev"]),
        "test_count": len(splits["test"]),
        "core_record_count": sum(len(records_list) for records_list in splits.values()) if settings["core_only"] else None,
        "aux_record_count": sum(aux_reason_counter.values()) if settings["core_only"] else None,
        "aux_reason_distribution": dict(aux_reason_counter) if settings["core_only"] else {},
        "manifest_record_count": len(manifest_records),
        "doc_holdout_validation": doc_holdout_stats,
        "llamafactory_dataset_dir": display_path(settings["llamafactory_dataset_dir"], REPO_ROOT),
    }
    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
