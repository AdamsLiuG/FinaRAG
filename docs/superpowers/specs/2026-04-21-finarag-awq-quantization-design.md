# FinaRAG AWQ Quantization Design

**Date:** 2026-04-21

## Goal

Add a standalone, reusable quantization workflow under `training/` that takes a
base model path plus a LoRA adapter path and supports:

1. merge/export only
2. AWQ 4-bit quantization only
3. one-command pipeline that runs merge then quantization

The workflow must fit the current FinaRAG generator SFT setup centered on
`Qwen3.5-9B`, while remaining generic enough for future LoRA-trained models.

## Architecture

The feature will live in a new package:

```text
training/quantization/
├── __init__.py
├── common.py
├── merge_lora.py
├── quantize_awq.py
├── pipeline.py
└── README.md
```

The package is split by responsibility:

- `common.py` owns config loading, path resolution, subprocess helpers,
  calibration-data loading, manifest writing, and output validation.
- `merge_lora.py` builds an ephemeral LLaMA-Factory export config and runs
  `llamafactory-cli export` to merge LoRA weights into a full model directory.
- `quantize_awq.py` loads a merged model with AutoAWQ, constructs calibration
  prompts from chat-format SFT data, and writes an AWQ 4-bit model directory.
- `pipeline.py` orchestrates the first two stages without duplicating their
  implementation.

## Inputs And Outputs

### Merge stage

Required inputs:

- `--base-model-path`
- `--adapter-path`
- `--output-dir`

Optional inputs:

- `--template`
- `--trust-remote-code`
- `--export-device`
- `--export-size`
- `--llamafactory-cli`
- `--manifest-path`

Output:

- merged model directory containing model weights plus tokenizer assets
- manifest JSON recording command, inputs, outputs, and timestamp

### AWQ stage

Required inputs:

- `--model-path`
- `--output-dir`

Optional inputs:

- `--calib-path`
- `--dataset-format`
- `--max-calib-samples`
- `--max-calib-length`
- `--w-bit`
- `--q-group-size`
- `--zero-point`
- `--version`
- `--dtype`
- `--manifest-path`

Output:

- AWQ-quantized model directory
- manifest JSON with quantization config and calibration stats

### Pipeline stage

Required inputs:

- `--base-model-path`
- `--adapter-path`

Optional inputs:

- `--merged-output-dir`
- `--quantized-output-dir`
- all merge-stage overrides
- all AWQ-stage overrides

Output:

- merged model directory
- quantized model directory
- pipeline manifest referencing both stages

## Calibration Data Strategy

The quantizer should support the data already produced inside FinaRAG:

- `training/generator_sft/processed/*.chat.jsonl`
- `training/generator_sft/llamafactory_data/*.json`

The loader will support:

- `chat_jsonl`: records with `messages`
- `llamafactory_json`: records with `messages`, `instruction`/`input`/`output`,
  or `conversations`

Calibration text is generated from chat messages when possible. If the record is
instruction-style, it will be converted into a simple system-user-assistant chat
bundle before tokenization.

## Error Handling

The workflow must fail early when:

- base model path is missing
- adapter path is missing
- merged model output already exists and overwrite is not enabled
- calibration file does not exist
- zero usable calibration samples are produced
- expected artifacts are missing after merge or quantization

Errors should point to the broken path or stage directly.

## Testing

The implementation will add focused tests for:

- config and path resolution
- calibration sample extraction from both chat JSONL and LLaMA-Factory JSON
- LLaMA-Factory export config generation
- AWQ quantization config construction
- pipeline orchestration using mocked stage functions

Heavy end-to-end model execution is intentionally excluded from unit tests.

## Non-Goals

- modifying the existing generator SFT training flow
- auto-serving the quantized model with vLLM
- adding GPTQ or non-AWQ quantizers in this change
