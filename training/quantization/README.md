# 量化工作流

这个包为 FinaRAG 中经过 LoRA 微调的模型增加了一套独立、可复用的
merge + AWQ 工作流。

它被有意拆分为三个入口：

- `training.quantization.merge_lora`
- `training.quantization.quantize_awq`
- `training.quantization.pipeline`

前两个阶段彼此解耦，可以单独复用；第三个阶段只是按顺序编排它们。

## 环境准备

在执行下面的命令之前，请先激活已经安装好 LLaMA-Factory 和 AutoAWQ
的环境。

示例：

```bash
conda activate ecollm
cd /media/main/lgd/llm/FinaRAG
```

## 1. 仅合并 LoRA

```bash
python -m training.quantization.merge_lora \
  --base-model-path /media/main/lgd/llm/models/Qwen/Qwen3.5-9B \
  --adapter-path /media/main/lgd/llm/FinaRAG/training/generator_sft/saves/qwen3.5_9b_qlora_sft_2x4090 \
  --output-dir /media/main/lgd/llm/FinaRAG/training/quantization/artifacts/qwen3.5_9b_merged \
  --template default \
  --trust-remote-code \
  --overwrite-output-dir
```

## 2. 仅执行 AWQ 量化

如果省略 `--calib-path`，脚本会尝试从以下位置自动发现 FinaRAG 的
generator SFT 训练集：

- `training/generator_sft/processed/train.chat.v2.jsonl`
- `training/generator_sft/processed/train.chat.jsonl`
- `training/generator_sft/llamafactory_data/finarag_generator_v2_train.json`
- `training/generator_sft/llamafactory_data/finarag_generator_train.json`

```bash
python -m training.quantization.quantize_awq \
  --model-path /media/main/lgd/llm/FinaRAG/training/quantization/artifacts/qwen3.5_9b_merged \
  --output-dir /media/main/lgd/llm/FinaRAG/training/quantization/artifacts/qwen3.5_9b_awq_int4 \
  --calib-path /media/main/lgd/llm/FinaRAG/training/generator_sft/processed/train.chat.v2.jsonl \
  --dataset-format auto \
  --max-calib-samples 128 \
  --max-calib-length 3072 \
  --w-bit 4 \
  --q-group-size 128 \
  --zero-point \
  --version GEMM \
  --overwrite-output-dir
```

## 3. 端到端执行 Merge + AWQ

```bash
python -m training.quantization.pipeline \
  --base-model-path /media/main/lgd/llm/models/Qwen/Qwen3.5-9B \
  --adapter-path /media/main/lgd/llm/FinaRAG/training/generator_sft/saves/qwen3.5_9b_qlora_sft_2x4090 \
  --merged-output-dir /media/main/lgd/llm/FinaRAG/training/quantization/artifacts/qwen3.5_9b_merged \
  --quantized-output-dir /media/main/lgd/llm/FinaRAG/training/quantization/artifacts/qwen3.5_9b_awq_int4 \
  --calib-path /media/main/lgd/llm/FinaRAG/training/generator_sft/processed/train.chat.v2.jsonl \
  --template default \
  --trust-remote-code \
  --overwrite-merged-output-dir \
  --overwrite-quantized-output-dir
```

## 输出目录结构

每个阶段都会在其输出旁边写出一个 manifest 文件：

- merge 阶段：`merge_manifest.json`
- AWQ 阶段：`awq_manifest.json`
- pipeline 阶段：`pipeline_manifest.json`

这些 manifest 用于让重复运行和产物追踪更容易一些。

## 4. Reranker AWQ

同一套量化入口现在也支持 reranker。关键区别只有一项：

- 传 `--task-type reranker`

reranker 的 AWQ 默认使用 `llmcompressor` 后端，避免新版
Transformers 下 AutoAWQ 对 Qwen3 reranker 的兼容性问题。

这样默认 calibration 数据会优先切到：

- `training/reranker_distill/processed/qwen3_reranker_sft_train.jsonl`
- `training/reranker_distill/processed/pointwise_train.jsonl`
- `training/reranker_distill/processed/pointwise_train_raw.jsonl`

示例：

```bash
python -m training.quantization.pipeline \
  --task-type reranker \
  --base-model-path /media/main/lgd/llm/models/Qwen/Qwen3-Reranker-0.6B \
  --adapter-path /media/main/lgd/llm/FinaRAG/training/reranker_distill/saves/qwen3_reranker_0.6b_sft_lora_20260420_173247 \
  --merged-output-dir /media/main/lgd/llm/FinaRAG/training/quantization/artifacts/qwen3_reranker_0.6b_merged \
  --quantized-output-dir /media/main/lgd/llm/FinaRAG/training/quantization/artifacts/qwen3_reranker_0.6b_awq_int4 \
  --overwrite-merged-output-dir \
  --overwrite-quantized-output-dir
```

## 5. 现成 Shell 脚本

如果你更希望直接执行现成 `.sh`，可以使用：

```bash
sh training/quantization/scripts/generator_merge_qwen3_5_9b.sh
sh training/quantization/scripts/generator_awq_qwen3_5_9b.sh
sh training/quantization/scripts/reranker_merge_qwen3_0_6b.sh
sh training/quantization/scripts/reranker_awq_qwen3_0_6b.sh
```

这些脚本默认使用当前仓库里的实际路径，同时支持通过环境变量覆盖默认值。
例如：

```bash
OUTPUT_DIR=/tmp/qwen3.5_9b_merged sh training/quantization/scripts/generator_merge_qwen3_5_9b.sh
CALIB_PATH=/tmp/calib.jsonl sh training/quantization/scripts/reranker_awq_qwen3_0_6b.sh
```

对于 `Qwen3.5-9B` 这类 `model_type=qwen3_5` 的新架构模型，`generator_awq_qwen3_5_9b.sh`
会默认尝试 `BACKEND=auto`：

- 如果当前环境里的 AutoAWQ 支持该模型，就继续用 AutoAWQ
- 如果 AutoAWQ 不支持，但已经安装了 `llmcompressor`，就自动切到 `llmcompressor`
- 如果两者都不满足，会直接报出安装提示，而不是在深层堆栈里失败

当前这条 generator 量化流使用的是文本 calibration 数据。对于带视觉塔的
`qwen3_5` 检查点，脚本会默认让 `model.visual.*` 保持全精度，只量化语言模型
分支，避免因为视觉层维度不能被 `group_size=128` 整除而中断。
