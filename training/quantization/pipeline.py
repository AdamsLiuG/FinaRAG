from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import load_yaml_mapping  # noqa: E402
from training.quantization.common import (  # noqa: E402
    SUPPORTED_TASK_TYPES,
    build_stage_manifest,
    coalesce,
    default_stage_manifest_path,
    resolve_path,
    sanitize_name,
    write_stage_manifest,
)
from training.quantization.merge_lora import run_merge_stage  # noqa: E402
from training.quantization.quantize_awq import (  # noqa: E402
    SUPPORTED_QUANTIZATION_BACKENDS,
    run_quantize_stage,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run merge/export plus AWQ quantization end to end.")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional YAML config path.")
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_QUANTIZATION_BACKENDS,
        default=None,
        help="Choose auto, AutoAWQ, or llmcompressor for AWQ 4-bit.",
    )
    parser.add_argument(
        "--task-type",
        choices=SUPPORTED_TASK_TYPES,
        default=None,
        help="Choose generator or reranker AWQ defaults.",
    )
    parser.add_argument("--base-model-path", type=Path, default=None, help="Base model directory path.")
    parser.add_argument("--adapter-path", type=Path, default=None, help="LoRA adapter directory path.")
    parser.add_argument("--merged-output-dir", type=Path, default=None, help="Merged model output directory.")
    parser.add_argument("--quantized-output-dir", type=Path, default=None, help="Quantized model output directory.")
    parser.add_argument("--calib-path", type=Path, default=None, help="Calibration dataset path.")
    parser.add_argument("--template", type=str, default=None, help="LLaMA-Factory template name.")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to trust remote code in both stages.",
    )
    parser.add_argument("--llamafactory-cli", type=str, default=None, help="Path or name of llamafactory-cli.")
    parser.add_argument("--export-device", type=str, default=None, help="LLaMA-Factory export device.")
    parser.add_argument("--export-size", type=int, default=None, help="LLaMA-Factory export shard size in GB.")
    parser.add_argument(
        "--export-legacy-format",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to enable legacy export format.",
    )
    parser.add_argument("--dataset-format", type=str, default=None, help="Calibration dataset format.")
    parser.add_argument("--max-calib-samples", type=int, default=None, help="Maximum calibration sample count.")
    parser.add_argument("--max-calib-length", type=int, default=None, help="Maximum calibration token length.")
    parser.add_argument("--w-bit", type=int, default=None, help="AWQ weight bit width.")
    parser.add_argument("--q-group-size", type=int, default=None, help="AWQ quantization group size.")
    parser.add_argument(
        "--zero-point",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to enable AWQ zero-point quantization.",
    )
    parser.add_argument("--version", type=str, default=None, help="AWQ version string.")
    parser.add_argument("--device-map", type=str, default=None, help="AutoAWQ device_map.")
    parser.add_argument("--dtype", type=str, default=None, help="Optional torch dtype override.")
    parser.add_argument("--shard-size", type=str, default=None, help="Quantized output shard size.")
    parser.add_argument(
        "--overwrite-merged-output-dir",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replace the merged output directory if it already exists.",
    )
    parser.add_argument(
        "--overwrite-quantized-output-dir",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replace the quantized output directory if it already exists.",
    )
    parser.add_argument("--manifest-path", type=Path, default=None, help="Write pipeline manifest JSON here.")
    return parser


def _default_output_dirs(adapter_path: Path) -> Dict[str, Path]:
    adapter_name = sanitize_name(adapter_path.name)
    base_dir = REPO_ROOT / "training/quantization/artifacts"
    return {
        "merged_output_dir": base_dir / f"{adapter_name}_merged",
        "quantized_output_dir": base_dir / f"{adapter_name}_awq_int4",
    }


def _resolve_stage_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_yaml_mapping(args.config_path)

    backend = str(coalesce(args.backend, config.get("backend"), "auto"))
    task_type = str(coalesce(args.task_type, config.get("task_type"), "generator"))
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported task_type: {task_type}. Expected one of {SUPPORTED_TASK_TYPES}.")
    base_model_path = resolve_path(coalesce(args.base_model_path, config.get("base_model_path")))
    adapter_path = resolve_path(coalesce(args.adapter_path, config.get("adapter_path")))
    calib_path = resolve_path(coalesce(args.calib_path, config.get("calib_path")))

    if base_model_path is None:
        raise ValueError("--base-model-path is required.")
    if adapter_path is None:
        raise ValueError("--adapter-path is required.")

    defaults = _default_output_dirs(adapter_path)
    merged_output_dir = resolve_path(
        coalesce(args.merged_output_dir, config.get("merged_output_dir"), defaults["merged_output_dir"])
    )
    quantized_output_dir = resolve_path(
        coalesce(args.quantized_output_dir, config.get("quantized_output_dir"), defaults["quantized_output_dir"])
    )
    trust_remote_code = bool(coalesce(args.trust_remote_code, config.get("trust_remote_code"), True))
    manifest_path = resolve_path(coalesce(args.manifest_path, config.get("manifest_path")))

    return {
        "backend": backend,
        "task_type": task_type,
        "base_model_path": base_model_path,
        "adapter_path": adapter_path,
        "merged_output_dir": merged_output_dir,
        "quantized_output_dir": quantized_output_dir,
        "calib_path": calib_path,
        "template": str(coalesce(args.template, config.get("template"), "default")),
        "trust_remote_code": trust_remote_code,
        "llamafactory_cli": str(coalesce(args.llamafactory_cli, config.get("llamafactory_cli"), "llamafactory-cli")),
        "export_device": str(coalesce(args.export_device, config.get("export_device"), "cpu")),
        "export_size": int(coalesce(args.export_size, config.get("export_size"), 5)),
        "export_legacy_format": bool(coalesce(args.export_legacy_format, config.get("export_legacy_format"), False)),
        "dataset_format": str(coalesce(args.dataset_format, config.get("dataset_format"), "auto")),
        "max_calib_samples": int(coalesce(args.max_calib_samples, config.get("max_calib_samples"), 128)),
        "max_calib_length": int(coalesce(args.max_calib_length, config.get("max_calib_length"), 3072)),
        "w_bit": int(coalesce(args.w_bit, config.get("w_bit"), 4)),
        "q_group_size": int(coalesce(args.q_group_size, config.get("q_group_size"), 128)),
        "zero_point": bool(coalesce(args.zero_point, config.get("zero_point"), True)),
        "version": str(coalesce(args.version, config.get("version"), "GEMM")),
        "device_map": str(coalesce(args.device_map, config.get("device_map"), "auto")),
        "dtype": coalesce(args.dtype, config.get("dtype"), None),
        "shard_size": str(coalesce(args.shard_size, config.get("shard_size"), "5GB")),
        "overwrite_merged_output_dir": bool(
            coalesce(args.overwrite_merged_output_dir, config.get("overwrite_merged_output_dir"), False)
        ),
        "overwrite_quantized_output_dir": bool(
            coalesce(args.overwrite_quantized_output_dir, config.get("overwrite_quantized_output_dir"), False)
        ),
        "manifest_path": manifest_path,
    }


def run_pipeline_stage(
    *,
    backend: str = "auto",
    task_type: str = "generator",
    base_model_path: Path | str,
    adapter_path: Path | str,
    merged_output_dir: Path | str,
    quantized_output_dir: Path | str,
    calib_path: Path | str | None = None,
    template: str = "default",
    trust_remote_code: bool = True,
    llamafactory_cli: str = "llamafactory-cli",
    export_device: str = "cpu",
    export_size: int = 5,
    export_legacy_format: bool = False,
    dataset_format: str = "auto",
    max_calib_samples: int = 128,
    max_calib_length: int = 3072,
    w_bit: int = 4,
    q_group_size: int = 128,
    zero_point: bool = True,
    version: str = "GEMM",
    device_map: str = "auto",
    dtype: str | None = None,
    shard_size: str = "5GB",
    overwrite_merged_output_dir: bool = False,
    overwrite_quantized_output_dir: bool = False,
    manifest_path: Path | str | None = None,
) -> Dict[str, Any]:
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported task_type: {task_type}. Expected one of {SUPPORTED_TASK_TYPES}.")
    merged_output_dir = Path(merged_output_dir)
    quantized_output_dir = Path(quantized_output_dir)
    manifest_path = Path(manifest_path) if manifest_path is not None else default_stage_manifest_path(
        quantized_output_dir, filename="pipeline_manifest.json"
    )

    merge_result = run_merge_stage(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        output_dir=merged_output_dir,
        template=template,
        trust_remote_code=trust_remote_code,
        export_device=export_device,
        export_size=export_size,
        export_legacy_format=export_legacy_format,
        overwrite_output_dir=overwrite_merged_output_dir,
        llamafactory_cli=llamafactory_cli,
    )
    quantize_result = run_quantize_stage(
        backend=backend,
        task_type=task_type,
        model_path=merge_result["output_dir"],
        output_dir=quantized_output_dir,
        calib_path=calib_path,
        dataset_format=dataset_format,
        max_calib_samples=max_calib_samples,
        max_calib_length=max_calib_length,
        w_bit=w_bit,
        q_group_size=q_group_size,
        zero_point=zero_point,
        version=version,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        shard_size=shard_size,
        overwrite_output_dir=overwrite_quantized_output_dir,
    )

    payload = build_stage_manifest(
        stage="merge_then_awq",
        inputs={
            "backend": backend,
            "task_type": task_type,
            "base_model_path": Path(base_model_path),
            "adapter_path": Path(adapter_path),
            "calib_path": Path(calib_path) if calib_path is not None else None,
            "template": template,
            "trust_remote_code": trust_remote_code,
            "llamafactory_cli": llamafactory_cli,
        },
        outputs={
            "merged_output_dir": Path(merge_result["output_dir"]),
            "quantized_output_dir": Path(quantize_result["output_dir"]),
            "merge_manifest_path": Path(merge_result["manifest_path"]),
            "awq_manifest_path": Path(quantize_result["manifest_path"]),
        },
        extra={
            "quant_config": quantize_result["quant_config"],
            "calibration_sample_count": quantize_result["calibration_sample_count"],
        },
    )
    write_stage_manifest(manifest_path, payload)
    return {
        "merged_output_dir": merge_result["output_dir"],
        "quantized_output_dir": quantize_result["output_dir"],
        "manifest_path": str(manifest_path),
        "merge_result": merge_result,
        "quantize_result": quantize_result,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    stage_kwargs = _resolve_stage_kwargs(args)
    result = run_pipeline_stage(**stage_kwargs)
    print(result["quantized_output_dir"])


if __name__ == "__main__":
    main()
