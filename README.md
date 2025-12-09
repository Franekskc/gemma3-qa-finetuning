# Gemma 3 QA Fine-Tuning (SQuAD)

Experiments comparing **Full FT**, **LoRA (PEFT)**, and **Layer Freezing** for **extractive QA** on **SQuAD 1.1** with **Gemma 3**. Designed for a single GPU (RTX 3070 Ti / GTX 1650, 8 GB).

## Project Structure

```
src/gemmaqa/
├── cli.py                # Unified CLI entry point
├── config/               # Configuration
│   ├── settings.py       # Dataclasses (QAConfig, etc.)
│   └── default.yaml      # Default configuration
├── data/                 # Data handling
│   ├── loader.py         # Dataset loading & tokenization
│   └── prepare.py        # SQuAD preparation script
├── finetuning/           # Training strategies
│   ├── base.py           # Shared model loading utilities
│   ├── lora.py           # LoRA adapter strategy
│   ├── freeze.py         # Layer freezing strategy
│   ├── full.py           # Full finetuning strategy
│   └── trainer.py        # Training orchestration
├── evaluation/           # Evaluation
│   ├── metrics.py        # QA metrics (EM, F1)
│   └── evaluation_runner.py  # Evaluation CLI
├── inference/            # Inference
│   ├── model.py          # Model loading for inference
│   └── chat.py           # Interactive chat interface
└── utils/                # Utilities
    ├── logging.py        # Structlog configuration
    ├── seed.py           # Reproducibility utilities
    └── cuda.py           # CUDA availability check
```

## Installation

This project uses `uv` for dependency management.

1. **Install `uv`** (if not already installed):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Sync Dependencies**:
   ```powershell
   uv sync
   ```

## Quick Start

### 1. Check CUDA
```powershell
uv run gemmaqa check-cuda
```

### 2. Prepare Data
```powershell
uv run gemmaqa prepare-data --output data/ --train-size 4000 --test-size 1000
```

### 3. Train Model
```powershell
# LoRA (recommended for limited VRAM)
uv run gemmaqa train --mode lora

# Full finetuning
uv run gemmaqa train --mode full

# Layer freezing
uv run gemmaqa train --mode freeze
```

### 4. Evaluate
```powershell
uv run gemmaqa eval --checkpoint outputs/lora/final --num-samples 10
```

### 5. Chat
```powershell
# Interactive mode
uv run gemmaqa chat --checkpoint outputs/lora/final

# Single question
uv run gemmaqa chat --checkpoint outputs/lora/final -q "What is the capital of France?"
```

---

## CLI Reference

### Unified CLI
```bash
gemmaqa <command> [options]
```

| Command | Description |
|---------|-------------|
| `train` | Train/finetune a model |
| `eval` | Evaluate a trained model |
| `chat` | Interactive chat with model |
| `prepare-data` | Prepare SQuAD dataset |
| `check-cuda` | Check CUDA availability |

### Train Command
```bash
gemmaqa train --mode <mode> [--config <path>] [--train-data <path>] [--val-data <path>] [--max-steps <n>]
```

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--mode`, `-m` | ✓ | - | `full`, `lora`, or `freeze` |
| `--config`, `-c` | | `config/default.yaml` | Path to config YAML |
| `--train-data` | | `data/train_subset.json` | Training data path |
| `--val-data` | | `data/val_subset.json` | Validation data path |
| `--max-steps` | | - | Max steps (for testing) |

### Eval Command
```bash
gemmaqa eval [--checkpoint <path>] [--base-model <name>] [--num-samples <n>] [--test-data <path>]
```

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--checkpoint` | | - | Path to model/adapter. If not provided, the base model will be evaluated|
| `--base-model` | | `google/gemma-3-1b-it` | Base model name |
| `--data` | | `data/test-subset.json` | Testing data path |
| `--num-samples`, `-n` | | `5` | Number of samples |
| `--temperature` | | `0.1` | Generation temperature |


### Chat Command
```bash
gemmaqa chat [--checkpoint <path>] [--question <q>] [--context <c>]
```

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--checkpoint` | | - | Path to model/adapter |
| `--base-model` | | `google/gemma-3-1b-it` | Base model name |
| `--question`, `-q` | | - | Single question (non-interactive) |
| `--context`, `-c` | | - | Context for question |
| `--temperature` | | `0.7` | Generation temperature |
| `--max-tokens` | | `50` | Max new tokens |

### Prepare-Data Command
```bash
gemmaqa prepare-data [--output <dir>] [--train-size <n>] [--test-size <n>]
```

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--output`, `-o` | | `data` | Output directory |
| `--train-size` | | `4000` | Training samples |
| `--test-size` | | `1000` | Test samples |
| `--seed` | | `42` | Random seed |

---

## Configuration

Example configuration is stored in `src/gemmaqa/config/default.yaml`:
