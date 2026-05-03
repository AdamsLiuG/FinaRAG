from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import load_yaml_mapping  # noqa: E402
from training.quantization.common import (  # noqa: E402
    build_stage_manifest,
    coalesce,
    default_stage_manifest_path,
    ensure_command_available,
    ensure_existing_path,
    prepare_output_path,
    resolve_path,
    run_command,
    validate_model_directory,
    write_stage_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a base model and LoRA adapter into a full export.")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional YAML config path.")
    parser.add_argument("--base-model-path", type=Path, default=None, help="Base model directory path.")
    parser.add_argument("--adapter-path", type=Path, default=None, help="LoRA adapter directory path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Merged model output directory.")
    parser.add_argument("--template", type=str, default=None, help="LLaMA-Factory template name.")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to pass trust_remote_code to the export config.",
    )
    parser.add_argument("--export-device", type=str, default=None, help="LLaMA-Factory export device.")
    parser.add_argument("--export-size", type=int, default=None, help="LLaMA-Factory export shard size in GB.")
    parser.add_argument(
        "--export-legacy-format",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to enable legacy export format.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replace the merged output directory if it already exists.",
    )
    parser.add_argument("--llamafactory-cli", type=str, default=None, help="Path or name of llamafactory-cli.")
    parser.add_argument("--config-output-path", type=Path, default=None, help="Persist generated export YAML here.")
    parser.add_argument("--manifest-path", type=Path, default=None, help="Write stage manifest JSON here.")
    return parser


def build_merge_export_payload(
    *,
    base_model_path: Path,
    adapter_path: Path,
    export_dir: Path,
    template: str | None,
    trust_remote_code: bool,
    export_device: str,
    export_size: int,
    export_legacy_format: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model_name_or_path": str(base_model_path),
        "adapter_name_or_path": str(adapter_path),
        "trust_remote_code": bool(trust_remote_code),
        "export_dir": str(export_dir),
        "export_size": int(export_size),
        "export_device": str(export_device),
        "export_legacy_format": bool(export_legacy_format),
    }
    if template:
        payload["template"] = template
    return payload


def write_merge_export_config(
    *,
    config_path: Path,
    base_model_path: Path | str,
    adapter_path: Path | str,
    export_dir: Path | str,
    template: str | None,
    trust_remote_code: bool = True,
    export_device: str = "cpu",
    export_size: int = 5,
    export_legacy_format: bool = False,
) -> Path:
    payload = build_merge_export_payload(
        base_model_path=Path(base_model_path),
        adapter_path=Path(adapter_path),
        export_dir=Path(export_dir),
        template=template,
        trust_remote_code=trust_remote_code,
        export_device=export_device,
        export_size=export_size,
        export_legacy_format=export_legacy_format,
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path


def build_merge_command(llamafactory_cli: str, config_path: Path) -> list[str]:
    return [str(llamafactory_cli), "export", str(config_path)]


def build_merge_export_config_payload_for_manifest(
    *,
    base_model_path: Path,
    adapter_path: Path,
    output_dir: Path,
    template: str | None,
    trust_remote_code: bool,
    export_device: str,
    export_size: int,
    export_legacy_format: bool,
) -> Dict[str, Any]:
    return build_merge_export_payload(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        export_dir=output_dir,
        template=template,
        trust_remote_code=trust_remote_code,
        export_device=export_device,
        export_size=export_size,
        export_legacy_format=export_legacy_format,
    )


def _resolve_stage_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_yaml_mapping(args.config_path)

    base_model_path = resolve_path(
        coalesce(args.base_model_path, config.get("base_model_path") or config.get("model_name_or_path"))
    )
    adapter_path = resolve_path(
        coalesce(args.adapter_path, config.get("adapter_path") or config.get("adapter_name_or_path"))
    )
    output_dir = resolve_path(coalesce(args.output_dir, config.get("output_dir") or config.get("export_dir")))
    template = coalesce(args.template, config.get("template"), "default")
    trust_remote_code = bool(coalesce(args.trust_remote_code, config.get("trust_remote_code"), True))
    export_device = str(coalesce(args.export_device, config.get("export_device"), "cpu"))
    export_size = int(coalesce(args.export_size, config.get("export_size"), 5))
    export_legacy_format = bool(coalesce(args.export_legacy_format, config.get("export_legacy_format"), False))
    overwrite_output_dir = bool(coalesce(args.overwrite_output_dir, config.get("overwrite_output_dir"), False))
    llamafactory_cli = str(coalesce(args.llamafactory_cli, config.get("llamafactory_cli"), "llamafactory-cli"))
    config_output_path = resolve_path(coalesce(args.config_output_path, config.get("config_output_path")))
    manifest_path = resolve_path(coalesce(args.manifest_path, config.get("manifest_path")))

    if base_model_path is None:
        raise ValueError("--base-model-path is required.")
    if adapter_path is None:
        raise ValueError("--adapter-path is required.")
    if output_dir is None:
        raise ValueError("--output-dir is required.")

    return {
        "base_model_path": base_model_path,
        "adapter_path": adapter_path,
        "output_dir": output_dir,
        "template": template,
        "trust_remote_code": trust_remote_code,
        "export_device": export_device,
        "export_size": export_size,
        "export_legacy_format": export_legacy_format,
        "overwrite_output_dir": overwrite_output_dir,
        "llamafactory_cli": llamafactory_cli,
        "config_output_path": config_output_path,
        "manifest_path": manifest_path,
    }


def run_merge_stage(
    *,
    base_model_path: Path | str,
    adapter_path: Path | str,
    output_dir: Path | str,
    template: str = "default",
    trust_remote_code: bool = True,
    export_device: str = "cpu",
    export_size: int = 5,
    export_legacy_format: bool = False,
    overwrite_output_dir: bool = False,
    llamafactory_cli: str = "llamafactory-cli",
    config_output_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> Dict[str, Any]:
    base_model_path = ensure_existing_path(Path(base_model_path), label="base model directory")
    adapter_path = ensure_existing_path(Path(adapter_path), label="adapter directory")
    output_dir = prepare_output_path(Path(output_dir), overwrite=overwrite_output_dir)
    config_output_path = Path(config_output_path) if config_output_path is not None else None
    manifest_path = Path(manifest_path) if manifest_path is not None else default_stage_manifest_path(
        output_dir, filename="merge_manifest.json"
    )
    config_payload = build_merge_export_config_payload_for_manifest(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        output_dir=output_dir,
        template=template,
        trust_remote_code=trust_remote_code,
        export_device=export_device,
        export_size=export_size,
        export_legacy_format=export_legacy_format,
    )

    resolved_cli = ensure_command_available(llamafactory_cli)
    if config_output_path is not None:
        config_output_path.parent.mkdir(parents=True, exist_ok=True)
        config_path = write_merge_export_config(
            config_path=config_output_path,
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            export_dir=output_dir,
            template=template,
            trust_remote_code=trust_remote_code,
            export_device=export_device,
            export_size=export_size,
            export_legacy_format=export_legacy_format,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="finarag-merge-") as temp_dir:
            config_path = write_merge_export_config(
                config_path=Path(temp_dir) / "merge_export.yaml",
                base_model_path=base_model_path,
                adapter_path=adapter_path,
                export_dir=output_dir,
                template=template,
                trust_remote_code=trust_remote_code,
                export_device=export_device,
                export_size=export_size,
                export_legacy_format=export_legacy_format,
            )
            command = build_merge_command(resolved_cli, config_path)
            run_command(command)
        artifacts = validate_model_directory(output_dir, stage_name="merge")
        payload = build_stage_manifest(
            stage="merge_lora",
            inputs={
                "base_model_path": base_model_path,
                "adapter_path": adapter_path,
                "template": template,
                "trust_remote_code": trust_remote_code,
                "export_device": export_device,
                "export_size": export_size,
                "export_legacy_format": export_legacy_format,
                "llamafactory_cli": resolved_cli,
            },
            outputs={
                "output_dir": output_dir,
                "artifacts": artifacts,
            },
        extra={
            "config_path": None,
            "config_payload": config_payload,
            "command": build_merge_command(resolved_cli, config_path),
        },
    )
    write_stage_manifest(manifest_path, payload)
    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "config_path": None,
        "artifacts": artifacts,
        "command": build_merge_command(resolved_cli, config_path),
    }

    command = build_merge_command(resolved_cli, config_path)
    run_command(command)
    artifacts = validate_model_directory(output_dir, stage_name="merge")
    payload = build_stage_manifest(
        stage="merge_lora",
        inputs={
            "base_model_path": base_model_path,
            "adapter_path": adapter_path,
            "template": template,
            "trust_remote_code": trust_remote_code,
            "export_device": export_device,
            "export_size": export_size,
            "export_legacy_format": export_legacy_format,
            "llamafactory_cli": resolved_cli,
        },
        outputs={
            "output_dir": output_dir,
            "artifacts": artifacts,
        },
        extra={
            "config_path": config_path,
            "config_payload": config_payload,
            "command": command,
        },
    )
    write_stage_manifest(manifest_path, payload)
    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "config_path": str(config_path),
        "artifacts": artifacts,
        "command": command,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    stage_kwargs = _resolve_stage_kwargs(args)
    result = run_merge_stage(**stage_kwargs)
    print(result["output_dir"])


if __name__ == "__main__":
    main()
