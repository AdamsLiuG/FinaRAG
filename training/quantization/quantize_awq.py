from __future__ import annotations

import argparse
import contextlib
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoConfig  # noqa: E402

try:
    from compressed_tensors.offload import disable_onloading, set_onload_device  # noqa: E402
except ImportError:  # pragma: no cover - optional dependency for llmcompressor path
    disable_onloading = None
    set_onload_device = None

try:
    from compressed_tensors.offload.cache.base import OffloadCache  # noqa: E402
except ImportError:  # pragma: no cover - optional dependency for llmcompressor path
    OffloadCache = ()

from training.common import load_yaml_mapping  # noqa: E402
from training.quantization.common import (  # noqa: E402
    SUPPORTED_TASK_TYPES,
    build_awq_quant_config,
    build_stage_manifest,
    coalesce,
    default_stage_manifest_path,
    discover_default_calibration_path,
    ensure_existing_path,
    load_calibration_texts,
    prepare_output_path,
    resolve_path,
    validate_model_directory,
    write_stage_manifest,
)

SUPPORTED_QUANTIZATION_BACKENDS = ("auto", "autoawq", "llmcompressor")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize a merged model with AWQ 4-bit.")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional YAML config path.")
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_QUANTIZATION_BACKENDS,
        default="auto",
        help="Choose auto, AutoAWQ, or llmcompressor for AWQ 4-bit.",
    )
    parser.add_argument(
        "--task-type",
        choices=SUPPORTED_TASK_TYPES,
        default=None,
        help="Choose generator or reranker calibration defaults.",
    )
    parser.add_argument("--model-path", type=Path, default=None, help="Merged model directory path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Quantized model output directory.")
    parser.add_argument("--calib-path", type=Path, default=None, help="Calibration dataset path.")
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
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to trust remote code when loading the merged model.",
    )
    parser.add_argument("--dtype", type=str, default=None, help="Optional torch dtype override.")
    parser.add_argument("--shard-size", type=str, default=None, help="Quantized output shard size.")
    parser.add_argument(
        "--overwrite-output-dir",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replace the quantized output directory if it already exists.",
    )
    parser.add_argument("--manifest-path", type=Path, default=None, help="Write stage manifest JSON here.")
    return parser


def _resolve_torch_dtype(dtype_name: str | None) -> Any:
    if dtype_name in (None, "", "auto"):
        return None
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    resolved = mapping.get(str(dtype_name).strip().lower())
    if resolved is None:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return resolved


def _autoawq_supported_model_types() -> set[str]:
    try:
        from awq.models.auto import AWQ_CAUSAL_LM_MODEL_MAP
    except ImportError:
        return set()
    return set(AWQ_CAUSAL_LM_MODEL_MAP.keys())


def _llmcompressor_available() -> bool:
    return find_spec("llmcompressor") is not None


def resolve_quantization_backend(
    *,
    model_path: Path,
    requested_backend: str,
    trust_remote_code: bool,
    task_type: str | None = None,
) -> tuple[str, str]:
    if requested_backend not in SUPPORTED_QUANTIZATION_BACKENDS:
        raise ValueError(
            f"Unsupported backend: {requested_backend}. Expected one of {SUPPORTED_QUANTIZATION_BACKENDS}."
        )

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
    model_type = str(config.model_type)

    if requested_backend == "autoawq" and model_type == "qwen3_5":
        raise ValueError(
            "AutoAWQ does not support model_type=qwen3_5. "
            "For Qwen3.5-style models, use the llmcompressor backend instead."
        )

    if requested_backend == "autoawq":
        autoawq_model_types = _autoawq_supported_model_types()
        if model_type not in autoawq_model_types:
            raise ValueError(
                f"AutoAWQ does not support model_type={model_type}. "
                "For Qwen3.5-style models, use the llmcompressor backend instead."
            )
        return "autoawq", model_type

    if requested_backend == "llmcompressor":
        if not _llmcompressor_available():
            raise ValueError(
                "The llmcompressor backend is not installed. Install it from source, for example: "
                "`pip install git+https://github.com/vllm-project/llm-compressor.git`"
            )
        return "llmcompressor", model_type

    if task_type == "reranker" and _llmcompressor_available():
        return "llmcompressor", model_type

    if model_type == "qwen3_5":
        if _llmcompressor_available():
            return "llmcompressor", model_type
        raise ValueError(
            "model_type=qwen3_5 is not supported by AutoAWQ. "
            "Install llmcompressor from source and rerun with `--backend llmcompressor` or default `--backend auto`: "
            "`pip install git+https://github.com/vllm-project/llm-compressor.git`"
        )

    autoawq_model_types = _autoawq_supported_model_types()
    if model_type in autoawq_model_types:
        return "autoawq", model_type

    raise ValueError(
        f"No supported AWQ backend is available for model_type={model_type}. "
        f"AutoAWQ supports: {sorted(autoawq_model_types)}"
    )


def _build_manifest_payload(
    *,
    backend: str,
    model_type: str,
    task_type: str,
    llmcompressor_ignore: list[str],
    model_path: Path,
    resolved_calib_path: Path,
    dataset_format: str,
    max_calib_samples: int,
    max_calib_length: int,
    device_map: str,
    trust_remote_code: bool,
    dtype: str | None,
    shard_size: str,
    output_dir: Path,
    artifacts: Dict[str, Any],
    quant_config: Dict[str, Any],
    calibration_sample_count: int,
) -> Dict[str, Any]:
    return build_stage_manifest(
        stage="quantize_awq",
        inputs={
            "backend": backend,
            "model_type": model_type,
            "task_type": task_type,
            "llmcompressor_ignore": llmcompressor_ignore,
            "model_path": model_path,
            "calib_path": resolved_calib_path,
            "dataset_format": dataset_format,
            "max_calib_samples": max_calib_samples,
            "max_calib_length": max_calib_length,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "shard_size": shard_size,
        },
        outputs={
            "output_dir": output_dir,
            "artifacts": artifacts,
        },
        extra={
            "quant_config": quant_config,
            "calibration_sample_count": calibration_sample_count,
        },
    )


def build_llmcompressor_ignore_patterns(*, model_type: str, task_type: str) -> list[str]:
    ignore_patterns = ["lm_head"]
    if model_type == "qwen3_5" and task_type == "generator":
        # The current generator calibration flow is text-only. For Qwen3.5
        # checkpoints that bundle a visual tower, leave that branch untouched.
        ignore_patterns.append(r"re:^model\.visual(\..*)?$")
    return ignore_patterns


def build_llmcompressor_sequential_targets(*, model_type: str, task_type: str) -> list[str] | None:
    if model_type == "qwen3_5" and task_type == "generator":
        return ["Qwen3_5DecoderLayer"]
    return None


def prepare_llmcompressor_model_for_oneshot(model: Any, *, model_type: str, task_type: str) -> None:
    if model_type == "qwen3_5" and task_type == "generator" and set_onload_device is not None:
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None:
            if isinstance(getattr(lm_head, "_parameters", None), OffloadCache):
                set_onload_device(lm_head, "cpu")
                return

            if hasattr(lm_head, "to"):
                lm_head.to("cpu")
            hook = getattr(lm_head, "_hf_hook", None)
            if hook is not None and hasattr(hook, "execution_device"):
                hook.execution_device = "cpu"


@contextlib.contextmanager
def patch_llmcompressor_disable_lm_head_for_offloaded_weights():
    try:
        import torch
        from llmcompressor.utils import helpers as llm_helpers
    except ImportError:
        yield
        return

    original_disable_lm_head = llm_helpers.disable_lm_head

    @contextlib.contextmanager
    def safe_disable_lm_head(model: torch.nn.Module):
        _, lm_head = llm_helpers.get_embeddings(model)
        if (
            lm_head is None
            or not isinstance(lm_head, torch.nn.Linear)
            or disable_onloading is None
            or not isinstance(getattr(lm_head, "_parameters", None), OffloadCache)
        ):
            with original_disable_lm_head(model):
                yield
            return

        # Avoid OffloadCache onloading the ignored lm_head weight back to GPU just
        # to construct the meta-device dummy used during calibration.
        with disable_onloading():
            dummy_weight = lm_head.weight.to("meta")

        def dummy_forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.to("meta") @ dummy_weight.T

        with contextlib.ExitStack() as stack:
            lm_head_forward = dummy_forward.__get__(lm_head)
            stack.enter_context(llm_helpers.patch_attr(lm_head, "forward", lm_head_forward))

            if hasattr(model, "_hf_hook"):
                stack.enter_context(llm_helpers.patch_attr(model._hf_hook, "io_same_device", False))

            yield

    llm_helpers.disable_lm_head = safe_disable_lm_head
    try:
        yield
    finally:
        llm_helpers.disable_lm_head = original_disable_lm_head


def _run_quantize_with_autoawq(
    *,
    model_path: Path,
    output_dir: Path,
    calibration_texts: list[str],
    trust_remote_code: bool,
    device_map: str,
    dtype: str | None,
    w_bit: int,
    q_group_size: int,
    zero_point: bool,
    version: str,
    shard_size: str,
) -> Dict[str, Any]:
    try:
        from awq import AutoAWQForCausalLM
    except ImportError as exc:
        raise ImportError("AutoAWQ is not installed in the current environment.") from exc

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)

    model_kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "safetensors": True,
    }
    torch_dtype = _resolve_torch_dtype(dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoAWQForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    quant_config = build_awq_quant_config(
        w_bit=w_bit,
        q_group_size=q_group_size,
        zero_point=zero_point,
        version=version,
    )
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_texts)
    model.save_quantized(str(output_dir), safetensors=True, shard_size=shard_size)
    tokenizer.save_pretrained(str(output_dir))
    return quant_config


def _run_quantize_with_llmcompressor(
    *,
    model_path: Path,
    output_dir: Path,
    calibration_texts: list[str],
    model_type: str,
    task_type: str,
    trust_remote_code: bool,
    device_map: str,
    dtype: str | None,
    max_calib_length: int,
) -> Dict[str, Any]:
    try:
        from datasets import Dataset
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier
    except ImportError as exc:
        raise ImportError(
            "The llmcompressor backend requires `datasets`, `llmcompressor`, and its dependencies to be installed."
        ) from exc

    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
    torch_dtype = _resolve_torch_dtype(dtype) or "auto"
    if model_type == "qwen3_5":
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    calibration_dataset = Dataset.from_dict({"text": calibration_texts})
    ignore_patterns = build_llmcompressor_ignore_patterns(
        model_type=model_type,
        task_type=task_type,
    )
    sequential_targets = build_llmcompressor_sequential_targets(
        model_type=model_type,
        task_type=task_type,
    )
    prepare_llmcompressor_model_for_oneshot(
        model,
        model_type=model_type,
        task_type=task_type,
    )
    recipe = [AWQModifier(ignore=ignore_patterns, scheme="W4A16_ASYM", targets=["Linear"])]
    with patch_llmcompressor_disable_lm_head_for_offloaded_weights():
        oneshot(
            model=model,
            tokenizer=tokenizer,
            recipe=recipe,
            dataset=calibration_dataset,
            output_dir=str(output_dir),
            num_calibration_samples=len(calibration_texts),
            max_seq_length=max_calib_length,
            batch_size=1,
            text_column="text",
            trust_remote_code_model=trust_remote_code,
            sequential_targets=sequential_targets,
            sequential_offload_device="cpu",
        )
    return build_awq_quant_config()


def _resolve_stage_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_yaml_mapping(args.config_path)

    backend = str(coalesce(args.backend, config.get("backend"), "auto"))
    task_type = str(coalesce(args.task_type, config.get("task_type"), "generator"))
    model_path = resolve_path(coalesce(args.model_path, config.get("model_path")))
    output_dir = resolve_path(coalesce(args.output_dir, config.get("output_dir")))
    calib_path = resolve_path(coalesce(args.calib_path, config.get("calib_path")))
    dataset_format = str(coalesce(args.dataset_format, config.get("dataset_format"), "auto"))
    max_calib_samples = int(coalesce(args.max_calib_samples, config.get("max_calib_samples"), 128))
    max_calib_length = int(coalesce(args.max_calib_length, config.get("max_calib_length"), 3072))
    w_bit = int(coalesce(args.w_bit, config.get("w_bit"), 4))
    q_group_size = int(coalesce(args.q_group_size, config.get("q_group_size"), 128))
    zero_point = bool(coalesce(args.zero_point, config.get("zero_point"), True))
    version = str(coalesce(args.version, config.get("version"), "GEMM"))
    device_map = str(coalesce(args.device_map, config.get("device_map"), "auto"))
    trust_remote_code = bool(coalesce(args.trust_remote_code, config.get("trust_remote_code"), True))
    dtype = coalesce(args.dtype, config.get("dtype"), None)
    shard_size = str(coalesce(args.shard_size, config.get("shard_size"), "5GB"))
    overwrite_output_dir = bool(coalesce(args.overwrite_output_dir, config.get("overwrite_output_dir"), False))
    manifest_path = resolve_path(coalesce(args.manifest_path, config.get("manifest_path")))

    if model_path is None:
        raise ValueError("--model-path is required.")
    if output_dir is None:
        raise ValueError("--output-dir is required.")

    return {
        "backend": backend,
        "task_type": task_type,
        "model_path": model_path,
        "output_dir": output_dir,
        "calib_path": calib_path,
        "dataset_format": dataset_format,
        "max_calib_samples": max_calib_samples,
        "max_calib_length": max_calib_length,
        "w_bit": w_bit,
        "q_group_size": q_group_size,
        "zero_point": zero_point,
        "version": version,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "dtype": dtype,
        "shard_size": shard_size,
        "overwrite_output_dir": overwrite_output_dir,
        "manifest_path": manifest_path,
    }


def run_quantize_stage(
    *,
    backend: str = "auto",
    task_type: str = "generator",
    model_path: Path | str,
    output_dir: Path | str,
    calib_path: Path | str | None = None,
    dataset_format: str = "auto",
    max_calib_samples: int = 128,
    max_calib_length: int = 3072,
    w_bit: int = 4,
    q_group_size: int = 128,
    zero_point: bool = True,
    version: str = "GEMM",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    dtype: str | None = None,
    shard_size: str = "5GB",
    overwrite_output_dir: bool = False,
    manifest_path: Path | str | None = None,
) -> Dict[str, Any]:
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported task_type: {task_type}. Expected one of {SUPPORTED_TASK_TYPES}.")
    model_path = ensure_existing_path(Path(model_path), label="merged model directory")
    output_dir = prepare_output_path(Path(output_dir), overwrite=overwrite_output_dir)
    manifest_path = Path(manifest_path) if manifest_path is not None else default_stage_manifest_path(
        output_dir, filename="awq_manifest.json"
    )

    resolved_calib_path = (
        Path(calib_path) if calib_path is not None else discover_default_calibration_path(task_type=task_type)
    )
    if resolved_calib_path is None:
        raise FileNotFoundError(
            "No calibration dataset was provided and no default FinaRAG calibration dataset could be discovered."
        )
    resolved_calib_path = ensure_existing_path(resolved_calib_path, label="calibration dataset")
    resolved_backend, model_type = resolve_quantization_backend(
        model_path=model_path,
        requested_backend=backend,
        trust_remote_code=trust_remote_code,
        task_type=task_type,
    )
    llmcompressor_ignore = build_llmcompressor_ignore_patterns(
        model_type=model_type,
        task_type=task_type,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
    calibration_texts = load_calibration_texts(
        resolved_calib_path,
        dataset_format=dataset_format,
        max_samples=max_calib_samples,
        tokenizer=tokenizer,
        max_length=max_calib_length,
    )
    if resolved_backend == "autoawq":
        quant_config = _run_quantize_with_autoawq(
            model_path=model_path,
            output_dir=output_dir,
            calibration_texts=calibration_texts,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            dtype=dtype,
            w_bit=w_bit,
            q_group_size=q_group_size,
            zero_point=zero_point,
            version=version,
            shard_size=shard_size,
        )
    elif resolved_backend == "llmcompressor":
        quant_config = _run_quantize_with_llmcompressor(
            model_path=model_path,
            output_dir=output_dir,
            calibration_texts=calibration_texts,
            model_type=model_type,
            task_type=task_type,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            dtype=dtype,
            max_calib_length=max_calib_length,
        )
    else:
        raise ValueError(f"Unsupported resolved backend: {resolved_backend}")

    artifacts = validate_model_directory(output_dir, stage_name="awq")
    payload = _build_manifest_payload(
        backend=resolved_backend,
        model_type=model_type,
        task_type=task_type,
        llmcompressor_ignore=llmcompressor_ignore,
        model_path=model_path,
        resolved_calib_path=resolved_calib_path,
        dataset_format=dataset_format,
        max_calib_samples=max_calib_samples,
        max_calib_length=max_calib_length,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        shard_size=shard_size,
        output_dir=output_dir,
        artifacts=artifacts,
        quant_config=quant_config,
        calibration_sample_count=len(calibration_texts),
    )
    write_stage_manifest(manifest_path, payload)
    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "artifacts": artifacts,
        "quant_config": quant_config,
        "calibration_sample_count": len(calibration_texts),
        "task_type": task_type,
        "backend": resolved_backend,
        "model_type": model_type,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    stage_kwargs = _resolve_stage_kwargs(args)
    result = run_quantize_stage(**stage_kwargs)
    print(result["output_dir"])


if __name__ == "__main__":
    main()
