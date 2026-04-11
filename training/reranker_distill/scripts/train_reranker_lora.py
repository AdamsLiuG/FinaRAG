from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import load_records, load_yaml_mapping, resolve_repo_path  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a LoRA/QLoRA reranker student from exported pointwise data.")
    parser.add_argument("--config-path", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--train-path", type=Path, default=None, help="Trainer export train path.")
    parser.add_argument("--eval-path", type=Path, default=None, help="Trainer export eval path.")
    parser.add_argument("--model-name-or-path", default=None, help="Base reranker model name or path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--quantization-bits", type=int, default=None, help="Optional 4 or 8 for QLoRA.")
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


def _resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_path = REPO_ROOT / "training/reranker_distill/configs/train.example.yaml"
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
        "trust_remote_code": bool(_coalesce(None, config.get("trust_remote_code"), False)),
        "cutoff_len": int(_coalesce(None, config.get("cutoff_len"), 1024)),
        "lora_rank": int(_coalesce(None, config.get("lora_rank"), 16)),
        "lora_alpha": int(_coalesce(None, config.get("lora_alpha"), 32)),
        "lora_dropout": float(_coalesce(None, config.get("lora_dropout"), 0.05)),
        "target_modules": _parse_target_modules(config.get("target_modules")),
        "modules_to_save": _parse_string_list(config.get("modules_to_save"), ["score"]),
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
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = _resolve_settings(args)

    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependency. Install torch, transformers, datasets, peft, and optionally bitsandbytes first."
        ) from exc

    set_seed(settings["seed"])
    train_records = load_records(settings["train_path"])
    eval_records = load_records(settings["eval_path"]) if settings["eval_path"] is not None and settings["eval_path"].exists() else []

    tokenizer = AutoTokenizer.from_pretrained(
        settings["model_name_or_path"],
        trust_remote_code=settings["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs: Dict[str, Any] = {
        "num_labels": 1,
        "problem_type": "regression",
        "trust_remote_code": settings["trust_remote_code"],
    }
    if settings["bf16"]:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif settings["fp16"]:
        model_kwargs["torch_dtype"] = torch.float16

    quantization_bits = settings["quantization_bits"]
    if quantization_bits in {4, 8}:
        if quantization_bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if settings["bf16"] else torch.float16,
            )
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(
        settings["model_name_or_path"],
        **model_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if quantization_bits in {4, 8}:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=settings["lora_rank"],
        lora_alpha=settings["lora_alpha"],
        lora_dropout=settings["lora_dropout"],
        bias="none",
        target_modules=settings["target_modules"],
        modules_to_save=settings["modules_to_save"],
    )
    model = get_peft_model(model, lora_config)

    def preprocess_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        tokenized = tokenizer(
            batch["query"],
            batch["passage"],
            truncation="only_second",
            max_length=settings["cutoff_len"],
        )
        tokenized["labels"] = [float(value) for value in batch["teacher_score"]]
        return tokenized

    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(eval_records) if eval_records else None
    train_dataset = train_dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            preprocess_batch,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.asarray(predictions).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        mse = float(np.mean((predictions - labels) ** 2))
        return {"mse": mse}

    training_args = TrainingArguments(
        output_dir=str(settings["output_dir"]),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=eval_dataset is not None,
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        num_train_epochs=settings["num_train_epochs"],
        per_device_train_batch_size=settings["per_device_train_batch_size"],
        per_device_eval_batch_size=settings["per_device_eval_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        warmup_ratio=settings["warmup_ratio"],
        lr_scheduler_type=settings["lr_scheduler_type"],
        logging_steps=settings["logging_steps"],
        save_steps=settings["save_steps"],
        eval_steps=settings["eval_steps"],
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        save_total_limit=settings["save_total_limit"],
        bf16=settings["bf16"],
        fp16=settings["fp16"],
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(settings["output_dir"])


if __name__ == "__main__":
    main()
