from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from training.common import display_path, load_records, utc_now_iso, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_TASK_TYPES = ("generator", "reranker")
_DEFAULT_CALIBRATION_CANDIDATES_BY_TASK = {
    "generator": (
        REPO_ROOT / "training/generator_sft/processed/train.chat.v2.jsonl",
        REPO_ROOT / "training/generator_sft/processed/train.chat.jsonl",
        REPO_ROOT / "training/generator_sft/llamafactory_data/finarag_generator_v2_train.json",
        REPO_ROOT / "training/generator_sft/llamafactory_data/finarag_generator_train.json",
    ),
    "reranker": (
        REPO_ROOT / "training/reranker_distill/processed/qwen3_reranker_sft_train.jsonl",
        REPO_ROOT / "training/reranker_distill/processed/pointwise_train.jsonl",
        REPO_ROOT / "training/reranker_distill/processed/pointwise_train_raw.jsonl",
    ),
}
_MODEL_WEIGHT_GLOBS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
)
_TOKENIZER_FILE_CANDIDATES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
_RERANKER_SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
_RERANKER_DEFAULT_INSTRUCTION = (
    "Given a Chinese financial annual report question, retrieve passages that directly "
    "answer the query with precise company, year, metric, and unit evidence."
)
_RERANKER_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def resolve_path(value: str | Path | None, *, repo_root: Path = REPO_ROOT) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def ensure_existing_path(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def prepare_output_path(path: Path, *, overwrite: bool) -> Path:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {path}. Pass --overwrite-output-dir to replace it."
            )
        if path == REPO_ROOT:
            raise ValueError(f"Refusing to remove repository root: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def find_model_weight_files(model_dir: Path) -> List[Path]:
    weight_files: List[Path] = []
    for pattern in _MODEL_WEIGHT_GLOBS:
        weight_files.extend(sorted(model_dir.glob(pattern)))
    return weight_files


def validate_model_directory(model_dir: Path, *, stage_name: str) -> Dict[str, Any]:
    ensure_existing_path(model_dir, label=f"{stage_name} output directory")
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{stage_name} output is missing config.json: {model_dir}")

    weight_files = find_model_weight_files(model_dir)
    if not weight_files:
        raise FileNotFoundError(f"{stage_name} output has no model weight files: {model_dir}")

    tokenizer_files = [name for name in _TOKENIZER_FILE_CANDIDATES if (model_dir / name).exists()]
    if not tokenizer_files:
        raise FileNotFoundError(f"{stage_name} output is missing tokenizer assets: {model_dir}")

    return {
        "config_path": str(config_path),
        "weight_files": [str(path) for path in weight_files],
        "tokenizer_files": tokenizer_files,
    }


def ensure_command_available(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(
            f"Command not found: {command}. Activate the correct environment first, "
            "or pass an explicit --llamafactory-cli path."
        )
    return resolved


def run_command(command: Sequence[str], *, cwd: Path = REPO_ROOT) -> None:
    try:
        subprocess.run(list(command), cwd=str(cwd), check=True)
    except subprocess.CalledProcessError as exc:
        rendered = " ".join(str(part) for part in command)
        raise RuntimeError(f"Command failed with exit code {exc.returncode}: {rendered}") from exc


def build_awq_quant_config(
    *,
    w_bit: int = 4,
    q_group_size: int = 128,
    zero_point: bool = True,
    version: str = "GEMM",
) -> Dict[str, Any]:
    return {
        "w_bit": int(w_bit),
        "q_group_size": int(q_group_size),
        "zero_point": bool(zero_point),
        "version": str(version),
    }


def discover_default_calibration_path(*, task_type: str = "generator", repo_root: Path = REPO_ROOT) -> Path | None:
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported task_type: {task_type}. Expected one of {SUPPORTED_TASK_TYPES}.")
    for candidate in _DEFAULT_CALIBRATION_CANDIDATES_BY_TASK[task_type]:
        resolved = candidate if candidate.is_absolute() else (repo_root / candidate).resolve()
        if resolved.exists():
            return resolved
    return None


def _flatten_message_content(content: Any) -> str:
    if content in (None, ""):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        flattened = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("value") or ""
                if text:
                    flattened.append(str(text))
            elif item not in (None, ""):
                flattened.append(str(item))
        return "\n".join(part for part in flattened if part)
    return str(content)


def _normalize_chat_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = (
            message.get("role")
            or message.get("from")
            or message.get("speaker")
            or message.get("author")
            or ""
        )
        role = str(role).strip().lower()
        if role in {"human", "user_prompt"}:
            role = "user"
        elif role in {"gpt", "assistant_response", "model"}:
            role = "assistant"
        elif role in {"system_prompt"}:
            role = "system"
        content = _flatten_message_content(message.get("content") if "content" in message else message.get("value"))
        if role and content.strip():
            normalized.append({"role": role, "content": content.strip()})
    return normalized


def record_to_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    if isinstance(record.get("messages"), list):
        normalized = _normalize_chat_messages(record["messages"])
        if normalized:
            return normalized

    if isinstance(record.get("conversations"), list):
        normalized = _normalize_chat_messages(record["conversations"])
        if normalized:
            return normalized

    user_parts: List[str] = []
    instruction = str(record.get("instruction") or "").strip()
    input_text = str(record.get("input") or "").strip()
    if instruction:
        user_parts.append(instruction)
    if input_text:
        user_parts.append(input_text)

    assistant_text = str(record.get("output") or record.get("response") or "").strip()
    system_text = str(record.get("system") or "").strip()

    normalized: List[Dict[str, str]] = []
    if system_text:
        normalized.append({"role": "system", "content": system_text})
    if user_parts:
        normalized.append({"role": "user", "content": "\n\n".join(user_parts)})
    if assistant_text:
        normalized.append({"role": "assistant", "content": assistant_text})
    return normalized


def render_messages_for_calibration(messages: Sequence[Dict[str, str]], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=False,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            pass
    return "\n\n".join(f"{item['role']}: {item['content']}" for item in messages if item.get("content"))


def build_reranker_calibration_prompt(
    *,
    query: str,
    passage: str,
    instruction: str | None = None,
    system_prompt: str | None = None,
) -> str:
    instruction_text = str(instruction or _RERANKER_DEFAULT_INSTRUCTION).strip()
    system_text = str(system_prompt or _RERANKER_SYSTEM_PROMPT).strip()
    query_text = str(query or "").strip()
    passage_text = str(passage or "").strip()
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<Instruct>: {instruction_text}\n"
        f"<Query>: {query_text}\n"
        f"<Document>: {passage_text}"
        f"{_RERANKER_PROMPT_SUFFIX}"
    )


def truncate_text_to_token_limit(text: str, tokenizer: Any, *, max_length: int | None) -> str:
    if not text or tokenizer is None or not max_length or max_length <= 0:
        return text
    if not hasattr(tokenizer, "encode"):
        return text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_length:
        return text
    token_ids = token_ids[:max_length]
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=False)
        except Exception:
            return text
    return text


def infer_dataset_format(records: Sequence[Dict[str, Any]], *, fallback: str = "llamafactory_json") -> str:
    if not records:
        return fallback
    record = records[0]
    if str(record.get("prompt") or "").strip():
        return "reranker_sft_jsonl"
    if isinstance(record.get("messages"), list):
        return "chat_jsonl"
    if isinstance(record.get("conversations"), list):
        return "llamafactory_json"
    if str(record.get("query") or record.get("question_text") or "").strip() and str(
        record.get("passage") or record.get("text") or ""
    ).strip():
        return "reranker_pointwise_jsonl"
    if any(key in record for key in ("instruction", "input", "output", "response")):
        return "llamafactory_json"
    return fallback


def load_calibration_texts(
    calib_path: Path,
    *,
    dataset_format: str = "auto",
    max_samples: int = 128,
    tokenizer: Any = None,
    max_length: int | None = None,
) -> List[str]:
    records = load_records(calib_path)
    format_name = infer_dataset_format(records) if dataset_format == "auto" else dataset_format
    supported_formats = {
        "chat_jsonl",
        "llamafactory_json",
        "reranker_sft_jsonl",
        "reranker_pointwise_jsonl",
    }
    if format_name not in supported_formats:
        raise ValueError(f"Unsupported dataset format: {format_name}. Expected one of {sorted(supported_formats)}.")

    texts: List[str] = []
    for record in records:
        if format_name in {"chat_jsonl", "llamafactory_json"}:
            messages = record_to_messages(record)
            if not messages:
                continue
            rendered = render_messages_for_calibration(messages, tokenizer)
        elif format_name == "reranker_sft_jsonl":
            prompt = str(record.get("prompt") or "").strip()
            target = str(record.get("target") or record.get("output") or "").strip()
            rendered = f"{prompt}{target}" if target else prompt
        else:
            query = str(record.get("query") or record.get("question_text") or "").strip()
            passage = str(record.get("passage") or record.get("text") or "").strip()
            if not query or not passage:
                continue
            rendered = build_reranker_calibration_prompt(
                query=query,
                passage=passage,
                instruction=record.get("instruction"),
                system_prompt=record.get("system_prompt"),
            )
        rendered = truncate_text_to_token_limit(rendered, tokenizer, max_length=max_length).strip()
        if not rendered:
            continue
        texts.append(rendered)
        if len(texts) >= max(1, int(max_samples)):
            break

    if not texts:
        raise ValueError(f"No usable calibration samples were found in {calib_path}.")
    return texts


def default_stage_manifest_path(output_dir: Path, *, filename: str) -> Path:
    return output_dir / filename


def write_stage_manifest(manifest_path: Path, payload: Dict[str, Any]) -> Path:
    write_json(manifest_path, payload)
    return manifest_path


def build_stage_manifest(
    *,
    stage: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    def _render(value: Any) -> Any:
        if isinstance(value, Path):
            return display_path(value, repo_root) or str(value)
        if isinstance(value, list):
            return [_render(item) for item in value]
        if isinstance(value, dict):
            return {key: _render(item) for key, item in value.items()}
        return value

    payload = {
        "stage": stage,
        "timestamp": utc_now_iso(),
        "inputs": _render(inputs),
        "outputs": _render(outputs),
    }
    if extra:
        payload["extra"] = _render(extra)
    return payload


def sanitize_name(value: str) -> str:
    allowed = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_.") or "artifact"
