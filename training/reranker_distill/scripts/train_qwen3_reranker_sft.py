from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import load_records, load_yaml_mapping, resolve_repo_path  # noqa: E402
from training.reranker_distill.scripts.export_to_qwen3_reranker_sft import (  # noqa: E402
    QWEN3_RERANKER_DEFAULT_INSTRUCTION,
    QWEN3_RERANKER_PROMPT_SUFFIX,
    QWEN3_RERANKER_SYSTEM_PROMPT,
    build_qwen3_reranker_prompt_prefix,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT trainer for native Qwen3-Reranker yes/no supervision.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--train-path", type=Path, default=None, help="Qwen3 reranker SFT train JSONL path.")
    parser.add_argument("--eval-path", type=Path, default=None, help="Qwen3 reranker SFT eval JSONL path.")
    parser.add_argument("--model-name-or-path", default=None, help="Base model name or path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--quantization-bits", type=int, default=None, help="Optional 4 or 8 for QLoRA.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and validate dataset paths without training.")
    return parser


def _coalesce(cli_value: Any, config_value: Any, default: Any = None) -> Any:
    return cli_value if cli_value is not None else (config_value if config_value is not None else default)


def _parse_target_modules(value: Any) -> Any:
    if value in (None, "", "all-linear"):
        return "all-linear"
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return "all-linear"


def _parse_string_list(value: Any, default: List[str]) -> List[str]:
    if value in (None, ""):
        return list(default)
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(default)


def build_supervised_example(
    record: Dict[str, Any],
    tokenizer: Any,
    cutoff_len: int,
    *,
    default_instruction: str | None = None,
    default_system_prompt: str | None = None,
) -> Dict[str, List[int]]:
    query = str(record.get("query") or "").strip()
    passage = str(record.get("passage") or "").strip()
    target = str(record.get("target") or "").strip()
    if not query or not passage or not target:
        raise ValueError("each SFT record must contain non-empty query, passage, and target fields")

    instruction = str(record.get("instruction") or default_instruction or QWEN3_RERANKER_DEFAULT_INSTRUCTION).strip()
    system_prompt = str(record.get("system_prompt") or default_system_prompt or QWEN3_RERANKER_SYSTEM_PROMPT).strip()

    prefix_text = build_qwen3_reranker_prompt_prefix(
        query=query,
        instruction=instruction,
        system_prompt=system_prompt,
    )
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(QWEN3_RERANKER_PROMPT_SUFFIX, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids:
        raise ValueError("target tokenization produced an empty sequence")

    document_budget = cutoff_len - len(prefix_ids) - len(suffix_ids) - len(target_ids)
    if document_budget <= 0:
        raise ValueError("cutoff_len is too small for the configured prompt prefix and target token")

    document_ids = tokenizer.encode(
        passage,
        add_special_tokens=False,
        truncation=True,
        max_length=document_budget,
    )
    input_ids = prefix_ids + document_ids + suffix_ids + target_ids
    labels = ([-100] * (len(prefix_ids) + len(document_ids) + len(suffix_ids))) + target_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _filter_supported_kwargs(
    kwargs: Dict[str, Any],
    supported_parameters: Iterable[str],
) -> Dict[str, Any]:
    supported = set(supported_parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def build_training_arguments_kwargs(
    settings: Dict[str, Any],
    *,
    has_eval_records: bool,
    world_size: int,
    supported_parameters: Iterable[str],
) -> Dict[str, Any]:
    supported = set(supported_parameters)
    eval_strategy_key = "eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"
    kwargs: Dict[str, Any] = {
        "output_dir": str(settings["output_dir"]),
        "overwrite_output_dir": True,
        "do_train": True,
        "do_eval": bool(has_eval_records),
        "learning_rate": settings["learning_rate"],
        "weight_decay": settings["weight_decay"],
        "num_train_epochs": settings["num_train_epochs"],
        "per_device_train_batch_size": settings["per_device_train_batch_size"],
        "per_device_eval_batch_size": settings["per_device_eval_batch_size"],
        "gradient_accumulation_steps": settings["gradient_accumulation_steps"],
        "warmup_ratio": settings["warmup_ratio"],
        "lr_scheduler_type": settings["lr_scheduler_type"],
        "logging_steps": settings["logging_steps"],
        "save_steps": settings["save_steps"],
        "eval_steps": settings["eval_steps"],
        eval_strategy_key: "steps" if has_eval_records else "no",
        "save_strategy": "steps",
        "save_total_limit": settings["save_total_limit"],
        "bf16": settings["bf16"],
        "fp16": settings["fp16"],
        "gradient_checkpointing": settings["gradient_checkpointing"],
        "max_grad_norm": settings["max_grad_norm"],
        "optim": settings["optim"],
        "report_to": "none",
        "remove_unused_columns": False,
    }
    if int(world_size) > 1:
        kwargs["ddp_find_unused_parameters"] = False
    return _filter_supported_kwargs(kwargs, supported)


def build_trainer_kwargs(
    *,
    model: Any,
    training_args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
    data_collator: Any,
    preprocess_logits_for_metrics: Any,
    compute_metrics: Any,
    supported_parameters: Iterable[str],
) -> Dict[str, Any]:
    supported = set(supported_parameters)
    kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
        "compute_metrics": compute_metrics,
    }
    if "tokenizer" in supported:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in supported:
        kwargs["processing_class"] = tokenizer
    return _filter_supported_kwargs(kwargs, supported)


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/reranker_distill/configs/sft_train.example.yaml"
    config_path = args.config_path or (default_config_path if default_config_path.exists() else None)
    config = load_yaml_mapping(config_path)

    train_path = resolve_repo_path(REPO_ROOT, _coalesce(args.train_path, config.get("train_path")))
    eval_path = resolve_repo_path(REPO_ROOT, _coalesce(args.eval_path, config.get("eval_path")))
    output_dir = resolve_repo_path(REPO_ROOT, _coalesce(args.output_dir, config.get("output_dir")))
    model_name_or_path = _coalesce(args.model_name_or_path, config.get("model_name_or_path"))
    if train_path is None or output_dir is None or not model_name_or_path:
        raise ValueError("train_path, output_dir, and model_name_or_path are required.")

    return {
        "config_path": config_path,
        "train_path": train_path,
        "eval_path": eval_path,
        "model_name_or_path": str(model_name_or_path),
        "output_dir": output_dir,
        "quantization_bits": _coalesce(args.quantization_bits, config.get("quantization_bits")),
        "trust_remote_code": bool(_coalesce(None, config.get("trust_remote_code"), True)),
        "cutoff_len": int(_coalesce(None, config.get("cutoff_len"), 2048)),
        "instruction": str(_coalesce(None, config.get("instruction"), QWEN3_RERANKER_DEFAULT_INSTRUCTION)),
        "system_prompt": str(_coalesce(None, config.get("system_prompt"), QWEN3_RERANKER_SYSTEM_PROMPT)),
        "lora_rank": int(_coalesce(None, config.get("lora_rank"), 16)),
        "lora_alpha": int(_coalesce(None, config.get("lora_alpha"), 32)),
        "lora_dropout": float(_coalesce(None, config.get("lora_dropout"), 0.05)),
        "target_modules": _parse_target_modules(config.get("target_modules")),
        "modules_to_save": _parse_string_list(config.get("modules_to_save"), []),
        "learning_rate": float(_coalesce(None, config.get("learning_rate"), 2.0e-4)),
        "weight_decay": float(_coalesce(None, config.get("weight_decay"), 0.0)),
        "num_train_epochs": float(_coalesce(None, config.get("num_train_epochs"), 3.0)),
        "per_device_train_batch_size": int(_coalesce(None, config.get("per_device_train_batch_size"), 8)),
        "per_device_eval_batch_size": int(_coalesce(None, config.get("per_device_eval_batch_size"), 8)),
        "gradient_accumulation_steps": int(_coalesce(None, config.get("gradient_accumulation_steps"), 2)),
        "warmup_ratio": float(_coalesce(None, config.get("warmup_ratio"), 0.05)),
        "lr_scheduler_type": str(_coalesce(None, config.get("lr_scheduler_type"), "cosine")),
        "logging_steps": int(_coalesce(None, config.get("logging_steps"), 10)),
        "save_steps": int(_coalesce(None, config.get("save_steps"), 200)),
        "eval_steps": int(_coalesce(None, config.get("eval_steps"), 200)),
        "save_total_limit": int(_coalesce(None, config.get("save_total_limit"), 3)),
        "seed": int(_coalesce(None, config.get("seed"), 42)),
        "bf16": bool(_coalesce(None, config.get("bf16"), True)),
        "fp16": bool(_coalesce(None, config.get("fp16"), False)),
        "gradient_checkpointing": bool(_coalesce(None, config.get("gradient_checkpointing"), True)),
        "max_grad_norm": float(_coalesce(None, config.get("max_grad_norm"), 1.0)),
        "optim": str(_coalesce(None, config.get("optim"), "adamw_torch")),
        "dry_run": bool(args.dry_run),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)
    train_records = load_records(settings["train_path"])
    eval_records = load_records(settings["eval_path"]) if settings["eval_path"] is not None and settings["eval_path"].exists() else []

    if settings["dry_run"]:
        print(
            {
                "train_path": str(settings["train_path"]),
                "eval_path": str(settings["eval_path"]) if settings["eval_path"] is not None else None,
                "train_record_count": len(train_records),
                "eval_record_count": len(eval_records),
                "model_name_or_path": settings["model_name_or_path"],
                "output_dir": str(settings["output_dir"]),
                "quantization_bits": settings["quantization_bits"],
            }
        )
        return

    try:
        import numpy as np
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependency. Install torch, transformers, peft, and optionally bitsandbytes first."
        ) from exc

    class _ListDataset(Dataset):
        def __init__(self, items: List[Dict[str, Any]]):
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            return self.items[index]

    class _RerankerDataCollator:
        def __init__(self, tokenizer: Any):
            self.tokenizer = tokenizer

        def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
            max_length = max(len(feature["input_ids"]) for feature in features)
            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []
            for feature in features:
                pad_length = max_length - len(feature["input_ids"])
                padded_input_ids.append(feature["input_ids"] + ([self.tokenizer.pad_token_id] * pad_length))
                padded_attention_mask.append(feature["attention_mask"] + ([0] * pad_length))
                padded_labels.append(feature["labels"] + ([-100] * pad_length))
            return {
                "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
                "labels": torch.tensor(padded_labels, dtype=torch.long),
            }

    set_seed(settings["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        settings["model_name_or_path"],
        trust_remote_code=settings["trust_remote_code"],
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": settings["trust_remote_code"],
    }
    if settings["bf16"]:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif settings["fp16"]:
        model_kwargs["torch_dtype"] = torch.float16

    quantization_bits = settings["quantization_bits"]
    if quantization_bits in {4, 8}:
        if quantization_bits == 4:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if settings["bf16"] else torch.float16,
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        settings["model_name_or_path"],
        **model_kwargs,
    )
    if settings["gradient_checkpointing"]:
        model.config.use_cache = False

    if quantization_bits in {4, 8}:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=settings["lora_rank"],
        lora_alpha=settings["lora_alpha"],
        lora_dropout=settings["lora_dropout"],
        bias="none",
        target_modules=settings["target_modules"],
        modules_to_save=settings["modules_to_save"] or None,
    )
    model = get_peft_model(model, lora_config)
    if settings["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    processed_train_records = [
        build_supervised_example(
            record,
            tokenizer,
            settings["cutoff_len"],
            default_instruction=settings["instruction"],
            default_system_prompt=settings["system_prompt"],
        )
        for record in train_records
    ]
    processed_eval_records = [
        build_supervised_example(
            record,
            tokenizer,
            settings["cutoff_len"],
            default_instruction=settings["instruction"],
            default_system_prompt=settings["system_prompt"],
        )
        for record in eval_records
    ]

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        labels = np.asarray(labels)
        predictions = np.asarray(predictions)
        valid_mask = labels != -100
        if not np.any(valid_mask):
            return {"token_accuracy": 0.0, "sample_accuracy": 0.0}

        token_accuracy = float((predictions[valid_mask] == labels[valid_mask]).mean())
        sample_matches: List[bool] = []
        for row_index in range(labels.shape[0]):
            row_positions = np.where(valid_mask[row_index])[0]
            if row_positions.size == 0:
                continue
            first_position = int(row_positions[0])
            sample_matches.append(bool(predictions[row_index, first_position] == labels[row_index, first_position]))
        sample_accuracy = float(sum(sample_matches) / len(sample_matches)) if sample_matches else 0.0
        return {
            "token_accuracy": token_accuracy,
            "sample_accuracy": sample_accuracy,
        }

    training_args_kwargs = build_training_arguments_kwargs(
        settings,
        has_eval_records=bool(processed_eval_records),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        supported_parameters=inspect.signature(TrainingArguments.__init__).parameters,
    )
    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = build_trainer_kwargs(
        model=model,
        training_args=training_args,
        train_dataset=_ListDataset(processed_train_records),
        eval_dataset=_ListDataset(processed_eval_records) if processed_eval_records else None,
        tokenizer=tokenizer,
        data_collator=_RerankerDataCollator(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if processed_eval_records else None,
        compute_metrics=compute_metrics if processed_eval_records else None,
        supported_parameters=inspect.signature(Trainer.__init__).parameters,
    )
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(settings["output_dir"])


if __name__ == "__main__":
    main()
