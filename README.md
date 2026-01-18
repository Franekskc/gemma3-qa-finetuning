# Gemma 3 QA Fine-Tuning (SQuAD)

Experiments comparing **Full FT**, **LoRA (PEFT)**, and **Layer Freezing** for **extractive QA** on **SQuAD 1.1** with *
*Gemma 3**. Designed for a single GPU (RTX 3070 Ti / GTX 1650, 8 GB).

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

### 4. Monitor Training (TensorBoard)

```powershell
# Point to the root outputs directory to compare all modes (LoRA vs Full vs Freeze)
uv run tensorboard --logdir outputs
```

### 5. Evaluate

```powershell
uv run gemmaqa eval --checkpoint outputs/lora/final --num-samples 10
```

### 6. Chat

```powershell
# Interactive mode
uv run gemmaqa chat --checkpoint outputs/lora/final

# Single question
uv run gemmaqa chat --checkpoint outputs/lora/final -q "What is the capital of France?"
```

### 7. RAG (Retrieval-Augmented Generation)

```powershell
# 1. Build Index (from corpus.json)
uv run gemmaqa rag-index --corpus data/corpus.json --output data

# 2. Evaluate RAG
# Compares retrieved context + Gemma 3 response against ground truth
uv run gemmaqa rag-eval --num-samples 10 --top-k 3

# 2b. Evaluate RAG with logs
uv run gemmaqa --log-level DEBUG rag-eval --num-samples 50 --top-k 3
```

---

## CLI Reference

### Unified CLI

```bash
gemmaqa [--log-level <level>] <command> [options]
```

| Command        | Description                 |
|----------------|-----------------------------|
| `--log-level`  | Set logging level (DEBUG)   |
| `train`        | Train/finetune a model      |
| `eval`         | Evaluate a trained model    |
| `chat`         | Interactive chat with model |
| `prepare-data` | Prepare SQuAD dataset       |
| `check-cuda`   | Check CUDA availability     |
| `rag-index`    | Build FAISS index for RAG   |
| `rag-eval`     | Evaluate RAG pipeline       |

### Train Command

```bash
gemmaqa train --mode <mode> [--config <path>] [--train-data <path>] [--val-data <path>] [--max-steps <n>]
```

| Argument         | Required | Default                  | Description                 |
|------------------|:--------:|--------------------------|-----------------------------|
| `--mode`, `-m`   |    ✓     | -                        | `full`, `lora`, or `freeze` |
| `--config`, `-c` |          | `config/default.yaml`    | Path to config YAML         |
| `--train-data`   |          | `data/train_subset.json` | Training data path          |
| `--val-data`     |          | `data/val_subset.json`   | Validation data path        |
| `--max-steps`    |          | -                        | Max steps (for testing)     |

### Eval Command

```bash
gemmaqa eval [--checkpoint <path>] [--base-model <name>] [--data <path>] [--num-samples <n>]
```

| Argument              | Required | Default                 | Description                                                              |
|-----------------------|:--------:|-------------------------|--------------------------------------------------------------------------|
| `--checkpoint`        |          | -                       | Path to model/adapter. If not provided, the base model will be evaluated |
| `--base-model`        |          | `google/gemma-3-1b-it`  | Base model name                                                          |
| `--data`              |          | `data/test-subset.json` | Testing data path                                                        |
| `--num-samples`, `-n` |          | `5`                     | Number of samples                                                        |

### Chat Command

```bash
gemmaqa chat [--checkpoint <path>] [--question <q>] [--context <c>] [--temperature <f>] [--max-tokens <n>]
```

| Argument           | Required | Default                | Description                       |
|--------------------|:--------:|------------------------|-----------------------------------|
| `--checkpoint`     |          | -                      | Path to model/adapter             |
| `--base-model`     |          | `google/gemma-3-1b-it` | Base model name                   |
| `--question`, `-q` |          | -                      | Single question (non-interactive) |
| `--context`, `-c`  |          | -                      | Context for question              |
| `--temperature`    |          | `0.7`                  | Generation temperature            |
| `--max-tokens`     |          | `50`                   | Max new tokens                    |

### Prepare-Data Command

```bash
gemmaqa prepare-data [--output <dir>] [--train-size <n>] [--val-size <n>] [--test-size <n>] [--seed <n>] [--mix-duorc <bool>]
```

| Argument         | Required | Default | Description                          |
|------------------|:--------:|---------|--------------------------------------|
| `--output`, `-o` |          | `data`  | Output directory                     |
| `--train-size`   |          | `5000`  | Training samples                     |
| `--val-size`     |          | `500`   | Validation samples                   |
| `--test-size`    |          | `5000`  | Test samples                         |
| `--seed`         |          | `42`    | Random seed                          |
| `--mix-duorc`    |          | `False` | mix DuoRC dataset into training data |

### RAG Commands

**Index**:

```bash
gemmaqa rag-index [--corpus <path>] [--output <dir>]
```

**Eval**:

```bash
gemmaqa rag-eval [--checkpoint <path>] [--index <path>] [--corpus <path>] [--num-samples <n>] [--top-k <k>] [--output <dir>]
```

---

## Configuration

Example configuration is stored in `src/gemmaqa/config/default.yaml`:
