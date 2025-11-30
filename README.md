# Gemma 3 QA Fine-Tuning (SQuAD)

Experiments comparing **Full FT**, **LoRA (PEFT)**, and **Layer Freezing** for **extractive QA** on **SQuAD 1.1** with **Gemma 3 (~4B)**. Designed for a single GPU (RTX 3070 Ti, 8 GB).

## Project Structure

The project is organized as follows:

- `src/gemmaqa/finetuning/`: Scripts for training and data processing.
- `src/gemmaqa/inference/`: Scripts for inference and evaluation.
- `src/gemmaqa/utils/`: Utility scripts (e.g., CUDA check).
- `gemma-lora-squad-final/`: Directory containing the fine-tuned LoRA adapter.

## Installation

1. Install `uv` (if not already installed):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Sync dependencies:
   ```powershell
   uv sync
   ```

## Usage

### Check CUDA Availability
Ensure your GPU is detected:
```powershell
uv run gemmaqa-check-cuda
```

### Training
To fine-tune the model:
```powershell
uv run gemmaqa-train
```

### Inference (Chat)
To run an interactive chat with the fine-tuned model:
```powershell
uv run gemmaqa-chat
```

### Evaluation
To evaluate the model on the SQuAD validation set:
```powershell
uv run gemmaqa-eval
```
