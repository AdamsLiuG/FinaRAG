from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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
    parser = argparse.ArgumentParser(description="Build a mixed generator SFT dataset from multiple chat JSONL pools.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--all-output-path", type=Path, default=None, help="Override merged all-output JSONL path.")
    parser.add_argument("--stats-output-path", type=Path, default=None, help="Override stats JSON path.")
    parser.add_argument("--allow-shortfall", action="store_true", help="Allow selections to undershoot target count.")
    parser.add_argument("--no-allow-shortfall", action="store_true", help="Disallow selection shortfalls.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _record_meta(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("meta")
    return meta if isinstance(meta, dict) else {}


def _record_value(record: Dict[str, Any], field_name: str) -> Any:
    if field_name in record and record[field_name] not in (None, "", []):
        return record[field_name]
    return _record_meta(record).get(field_name)


def _normalize_dedupe_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalize_dedupe_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _normalize_dedupe_value(item)) for key, item in value.items()))
    return "" if value is None else value


def _dedupe_key(record: Dict[str, Any], fields: Iterable[str]) -> Tuple[Any, ...]:
    return tuple(_normalize_dedupe_value(_record_value(record, field_name)) for field_name in fields)


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, list):
        actual_values = list(actual)
        if isinstance(expected, list):
            return any(item in expected for item in actual_values)
        return expected in actual_values
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def _matches_filters(record: Dict[str, Any], match: Dict[str, Any], exclude: Dict[str, Any]) -> bool:
    for field_name, expected in match.items():
        if not _value_matches(_record_value(record, field_name), expected):
            return False
    for field_name, blocked in exclude.items():
        if _value_matches(_record_value(record, field_name), blocked):
            return False
    return True


def _stable_rank(record: Dict[str, Any], *, random_seed: str, selection_name: str, dedupe_fields: Iterable[str]) -> str:
    key_payload = {
        "selection_name": selection_name,
        "dedupe_key": _dedupe_key(record, dedupe_fields),
        "random_seed": random_seed,
    }
    return hashlib.sha256(json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _annotate_record(
    record: Dict[str, Any],
    *,
    mix_tag: str,
    bucket_name: str,
    mix_group: str,
    mix_subgroup: str,
    index: int,
) -> Dict[str, Any]:
    cloned = copy.deepcopy(record)
    meta = dict(_record_meta(cloned))
    meta["mix_tag"] = mix_tag
    meta["mix_bucket"] = bucket_name
    meta["mix_group"] = mix_group
    meta["mix_subgroup"] = mix_subgroup
    meta["mix_group_index"] = index
    cloned["meta"] = meta
    return cloned


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = resolve_repo_path(REPO_ROOT, args.config_path) if args.config_path is not None else None
    config = load_yaml_mapping(config_path)

    mix_tag = str(_coalesce(None, config.get("mix_tag"), "generator_mix"))
    random_seed = str(_coalesce(None, config.get("random_seed"), mix_tag))
    allow_shortfall_config = config.get("allow_shortfall")
    allow_shortfall = (
        True
        if args.allow_shortfall
        else False
        if args.no_allow_shortfall
        else bool(allow_shortfall_config if allow_shortfall_config is not None else False)
    )

    dedupe_fields = [
        str(value)
        for value in (_coalesce(None, config.get("dedupe_fields"), ["sample_id", "query_id", "variant_type"]) or [])
        if str(value).strip()
    ]
    bucket_outputs_config = config.get("bucket_outputs") or {}
    if not isinstance(bucket_outputs_config, dict) or not bucket_outputs_config:
        raise ValueError("bucket_outputs must be a non-empty mapping.")

    bucket_outputs = {
        str(bucket_name): resolve_repo_path(REPO_ROOT, bucket_path)
        for bucket_name, bucket_path in bucket_outputs_config.items()
    }
    if any(path is None for path in bucket_outputs.values()):
        raise ValueError("All bucket output paths must be provided.")

    all_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.all_output_path, config.get("all_output_path")))
    stats_output_path = resolve_repo_path(REPO_ROOT, _coalesce(args.stats_output_path, config.get("stats_output_path")))
    if all_output_path is None or stats_output_path is None:
        raise ValueError("all_output_path and stats_output_path are required.")

    groups_config = config.get("groups") or []
    if not isinstance(groups_config, list) or not groups_config:
        raise ValueError("groups must be a non-empty list.")

    groups: List[Dict[str, Any]] = []
    for item in groups_config:
        if not isinstance(item, dict):
            raise ValueError("Each group config must be a mapping.")
        input_path = resolve_repo_path(REPO_ROOT, item.get("input_path"))
        if input_path is None:
            raise ValueError(f"Group {item.get('name')!r} is missing input_path.")
        bucket_name = str(item.get("bucket") or "").strip()
        if bucket_name not in bucket_outputs:
            raise ValueError(f"Group {item.get('name')!r} references unknown bucket: {bucket_name!r}")
        groups.append(
            {
                "name": str(item.get("name") or "").strip(),
                "bucket": bucket_name,
                "mix_group": str(item.get("mix_group") or bucket_name),
                "input_path": input_path,
                "count": max(0, int(item.get("count") or 0)),
                "match": dict(item.get("match") or {}),
                "exclude": dict(item.get("exclude") or {}),
            }
        )
    if any(not group["name"] for group in groups):
        raise ValueError("Each group must define a non-empty name.")

    return {
        "config_path": config_path,
        "mix_tag": mix_tag,
        "random_seed": random_seed,
        "allow_shortfall": allow_shortfall,
        "dedupe_fields": dedupe_fields,
        "bucket_outputs": bucket_outputs,
        "all_output_path": all_output_path,
        "stats_output_path": stats_output_path,
        "groups": groups,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)

    source_cache: Dict[Path, List[Dict[str, Any]]] = {}
    bucket_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_records: List[Dict[str, Any]] = []
    used_keys = set()
    selection_stats: Dict[str, Dict[str, Any]] = {}
    bucket_counts = Counter()

    for group in settings["groups"]:
        input_path = group["input_path"]
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input file for group {group['name']!r}: {input_path}")
        records = source_cache.setdefault(input_path, load_records(input_path))
        matching_records = [record for record in records if _matches_filters(record, group["match"], group["exclude"])]
        matching_records.sort(
            key=lambda record: _stable_rank(
                record,
                random_seed=settings["random_seed"],
                selection_name=group["name"],
                dedupe_fields=settings["dedupe_fields"],
            )
        )

        selected_records: List[Dict[str, Any]] = []
        for record in matching_records:
            key = _dedupe_key(record, settings["dedupe_fields"])
            if key in used_keys:
                continue
            selected_records.append(record)
            used_keys.add(key)
            if len(selected_records) >= group["count"]:
                break

        shortfall = max(0, group["count"] - len(selected_records))
        selection_stats[group["name"]] = {
            "bucket": group["bucket"],
            "mix_group": group["mix_group"],
            "input_path": display_path(input_path, REPO_ROOT),
            "requested_count": group["count"],
            "candidate_count": len(matching_records),
            "selected_count": len(selected_records),
            "shortfall": shortfall,
            "match": group["match"],
            "exclude": group["exclude"],
        }

        if shortfall and not settings["allow_shortfall"]:
            raise ValueError(
                f"Selection {group['name']!r} shortfall: requested={group['count']}, selected={len(selected_records)}"
            )

        for index, record in enumerate(selected_records):
            annotated = _annotate_record(
                record,
                mix_tag=settings["mix_tag"],
                bucket_name=group["bucket"],
                mix_group=group["mix_group"],
                mix_subgroup=group["name"],
                index=index,
            )
            bucket_records[group["bucket"]].append(annotated)
            all_records.append(annotated)
            bucket_counts[group["bucket"]] += 1

    for bucket_name, output_path in settings["bucket_outputs"].items():
        write_records(output_path, bucket_records.get(bucket_name, []))
    write_records(settings["all_output_path"], all_records)

    stats = {
        "build_timestamp": utc_now_iso(),
        "config_path": display_path(settings["config_path"], REPO_ROOT),
        "mix_tag": settings["mix_tag"],
        "random_seed": settings["random_seed"],
        "allow_shortfall": settings["allow_shortfall"],
        "dedupe_fields": settings["dedupe_fields"],
        "bucket_outputs": {name: display_path(path, REPO_ROOT) for name, path in settings["bucket_outputs"].items()},
        "all_output_path": display_path(settings["all_output_path"], REPO_ROOT),
        "bucket_counts": dict(bucket_counts),
        "all_record_count": len(all_records),
        "selection_stats": selection_stats,
    }
    write_json(settings["stats_output_path"], stats)


if __name__ == "__main__":
    main()
