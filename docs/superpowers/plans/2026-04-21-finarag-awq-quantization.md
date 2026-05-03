# FinaRAG AWQ Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `training/quantization` workflow that supports merge-only, AWQ-only, and one-command merge-plus-AWQ flows for LoRA fine-tuned Qwen-style models.

**Architecture:** Add a small quantization package under `training/` with shared helpers in `common.py`, a merge CLI that shells out to LLaMA-Factory export, an AWQ CLI that builds calibration prompts and writes a 4-bit model, and a pipeline CLI that orchestrates the two stages while keeping them decoupled.

**Tech Stack:** Python 3.12, argparse, pathlib, subprocess, JSON/JSONL, LLaMA-Factory CLI, AutoAWQ, pytest.

---

### Task 1: Create the quantization package scaffold

**Files:**
- Create: `training/quantization/__init__.py`
- Create: `training/quantization/common.py`
- Create: `training/quantization/README.md`
- Test: `tests/test_quantization_pipeline.py`

- [ ] **Step 1: Write the failing scaffold test**

```python
from pathlib import Path


def test_quantization_package_exists():
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "training/quantization/common.py").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_quantization_pipeline.py -q`
Expected: FAIL because `training/quantization/common.py` does not exist yet.

- [ ] **Step 3: Write the minimal scaffold**

```python
"""Standalone quantization helpers for FinaRAG training outputs."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_quantization_pipeline.py -q`
Expected: PASS for the scaffold existence check.

- [ ] **Step 5: Commit**

```bash
git add training/quantization/__init__.py training/quantization/common.py training/quantization/README.md tests/test_quantization_pipeline.py
git commit -m "feat: scaffold quantization workflow package"
```

### Task 2: Implement shared quantization helpers

**Files:**
- Modify: `training/quantization/common.py`
- Test: `tests/test_quantization_pipeline.py`

- [ ] **Step 1: Write failing tests for config loading and calibration extraction**

```python
def test_load_calibration_texts_from_chat_jsonl(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text('{"messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"A"}]}\n', encoding="utf-8")
    texts = load_calibration_texts(path, dataset_format="chat_jsonl", max_samples=8)
    assert len(texts) == 1
    assert "Q" in texts[0]


def test_build_awq_quant_config_defaults():
    config = build_awq_quant_config()
    assert config["w_bit"] == 4
    assert config["q_group_size"] == 128
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_quantization_pipeline.py -q`
Expected: FAIL because helper functions are not defined yet.

- [ ] **Step 3: Write minimal helper implementation**

```python
def build_awq_quant_config(w_bit=4, q_group_size=128, zero_point=True, version="GEMM"):
    return {
        "w_bit": w_bit,
        "q_group_size": q_group_size,
        "zero_point": zero_point,
        "version": version,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_quantization_pipeline.py -q`
Expected: PASS for helper behavior tests.

- [ ] **Step 5: Commit**

```bash
git add training/quantization/common.py tests/test_quantization_pipeline.py
git commit -m "feat: add shared quantization helpers"
```

### Task 3: Implement the merge CLI

**Files:**
- Create: `training/quantization/merge_lora.py`
- Modify: `training/quantization/common.py`
- Test: `tests/test_quantization_pipeline.py`

- [ ] **Step 1: Write the failing merge-config test**

```python
def test_build_merge_export_config_contains_base_and_adapter(tmp_path):
    config_path = tmp_path / "merge.yaml"
    write_merge_export_config(
        config_path=config_path,
        base_model_path="/models/Qwen3.5-9B",
        adapter_path="/adapters/qwen3.5-9b-lora",
        export_dir="/tmp/merged-model",
        template="default",
    )
    text = config_path.read_text(encoding="utf-8")
    assert "model_name_or_path: /models/Qwen3.5-9B" in text
    assert "adapter_name_or_path: /adapters/qwen3.5-9b-lora" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_quantization_pipeline.py -q`
Expected: FAIL because merge config writer does not exist yet.

- [ ] **Step 3: Write the merge CLI and config writer**

```python
def build_merge_command(llamafactory_cli, config_path):
    return [llamafactory_cli, "export", str(config_path)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_quantization_pipeline.py -q`
Expected: PASS for merge config and command assembly.

- [ ] **Step 5: Commit**

```bash
git add training/quantization/merge_lora.py training/quantization/common.py tests/test_quantization_pipeline.py
git commit -m "feat: add standalone lora merge cli"
```

### Task 4: Implement the AWQ CLI

**Files:**
- Create: `training/quantization/quantize_awq.py`
- Modify: `training/quantization/common.py`
- Test: `tests/test_quantization_pipeline.py`

- [ ] **Step 1: Write the failing AWQ args test**

```python
def test_quantize_awq_parser_defaults_to_awq_4bit():
    parser = build_quantize_arg_parser()
    args = parser.parse_args(["--model-path", "/tmp/merged", "--output-dir", "/tmp/awq"])
    assert args.w_bit == 4
    assert args.q_group_size == 128
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_quantization_pipeline.py -q`
Expected: FAIL because the AWQ CLI does not exist yet.

- [ ] **Step 3: Write the minimal AWQ implementation**

```python
quant_config = build_awq_quant_config(
    w_bit=args.w_bit,
    q_group_size=args.q_group_size,
    zero_point=args.zero_point,
    version=args.version,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_quantization_pipeline.py -q`
Expected: PASS for parser defaults and config generation.

- [ ] **Step 5: Commit**

```bash
git add training/quantization/quantize_awq.py training/quantization/common.py tests/test_quantization_pipeline.py
git commit -m "feat: add awq quantization cli"
```

### Task 5: Implement the pipeline CLI and documentation

**Files:**
- Create: `training/quantization/pipeline.py`
- Modify: `training/quantization/README.md`
- Test: `tests/test_quantization_pipeline.py`

- [ ] **Step 1: Write the failing orchestration test**

```python
def test_pipeline_runs_merge_then_quantize(monkeypatch, tmp_path):
    calls = []

    def fake_merge(**kwargs):
        calls.append(("merge", kwargs["output_dir"]))
        return {"output_dir": str(kwargs["output_dir"])}

    def fake_quantize(**kwargs):
        calls.append(("quantize", kwargs["output_dir"]))
        return {"output_dir": str(kwargs["output_dir"])}

    monkeypatch.setattr("training.quantization.pipeline.run_merge_stage", fake_merge)
    monkeypatch.setattr("training.quantization.pipeline.run_quantize_stage", fake_quantize)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_quantization_pipeline.py -q`
Expected: FAIL because the pipeline orchestration function does not exist yet.

- [ ] **Step 3: Write the pipeline CLI and README examples**

```python
merge_result = run_merge_stage(...)
quant_result = run_quantize_stage(model_path=merge_result["output_dir"], ...)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_quantization_pipeline.py -q`
Expected: PASS for pipeline orchestration and docs-backed usage assumptions.

- [ ] **Step 5: Commit**

```bash
git add training/quantization/pipeline.py training/quantization/README.md tests/test_quantization_pipeline.py
git commit -m "feat: add end-to-end quantization pipeline cli"
```

### Task 6: Run focused verification

**Files:**
- Verify: `training/quantization/*.py`
- Verify: `tests/test_quantization_pipeline.py`

- [ ] **Step 1: Run focused unit tests**

Run: `pytest tests/test_quantization_pipeline.py -q`
Expected: PASS with 0 failures.

- [ ] **Step 2: Run a repository-level smoke check for training scaffolds**

Run: `pytest tests/test_training_scaffolds.py -q`
Expected: PASS with 0 failures and no regressions in shared training helpers.

- [ ] **Step 3: Sanity-check CLI help output**

Run: `python -m training.quantization.pipeline --help`
Expected: exit 0 and usage text showing merge-plus-quantize arguments.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-04-21-finarag-awq-quantization-design.md docs/superpowers/plans/2026-04-21-finarag-awq-quantization.md
git commit -m "docs: add awq quantization design and plan"
```
