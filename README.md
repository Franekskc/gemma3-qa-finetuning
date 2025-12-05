# Gemma 3 QA Fine-Tuning (SQuAD)

Experiments comparing **Full FT**, **LoRA (PEFT)**, and **Layer Freezing** for **extractive QA** on **SQuAD 1.1** with **Gemma 3 (~4B)**. Designed for a single GPU (RTX 3070 Ti, 8 GB).

## Project Structure

- `src/gemmaqa/finetuning/`: Scripts for training and data processing.
- `src/gemmaqa/inference/`: Scripts for inference and evaluation.
- `src/gemmaqa/utils/`: Utility scripts (e.g., CUDA check).
- `gemma-lora-squad-final/`: Directory containing the fine-tuned LoRA adapter.

## Step-by-Step Tutorial

### 1. Install Dependencies
This project uses `uv` for dependency management.

1.  **Install `uv`** (if not already installed):
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Sync Dependencies**:
    ```powershell
    uv sync
    ```

### 2. Test CUDA
Ensure your GPU is detected and PyTorch is using it.
```powershell
uv run gemmaqa-check-cuda
```
*Expected Output:* `CUDA available: True`, `Device name: NVIDIA GeForce ...`

### 3. Generate Datasets
Split the SQuAD dataset into a training subset (4k examples), a test subset (1k examples), and a corpus of all unique contexts.
```powershell
uv run gemmaqa-prepare-data
```
*Output:* Creates `data/train_subset.json`, `data/test_subset.json`, and `data/corpus.json`.

### 4. Generate LoRA Matrices (Training)
Fine-tune the Gemma model using LoRA on the generated training subset.
```powershell
uv run gemmaqa-train
```
*Details:*
- Loads `google/gemma-3-1b-it` in 4-bit quantization.
- Trains on `data/train_subset.json`.
- Saves the adapter to `gemma-lora-squad-final`.

### 5. Run Inference or Evaluation

#### Option A: Interactive Chat
Chat with the fine-tuned model to test it manually.
```powershell
uv run gemmaqa-chat
```

#### Option B: SQuAD Evaluation
Evaluate on a random sample of the SQuAD validation set.
```powershell
uv run gemmaqa-eval
```

## Usage
The training script `src/gemmaqa/train.py` is the main entry point for the project. It is controlled via command-line arguments to select the training strategy and configuration file.
```bash
python src/gemmaqa/train.py --mode <mode> [--config <path/to/config.yaml>]
```

The script accepts the following arguments:

| Argument | Type | Required | Default | Description |
| :--- | :--- | :---: | :--- | :--- |
| `--mode` | `str` | **Yes** | - | Training strategy. Must be one of:<br>• `full`: Full Fine-Tuning<br>• `lora`: Low-Rank Adaptation<br>• `freeze`: layer freezing |
| `--config` | `str` | No | `src/gemmaqa/config.yaml` | Path to the YAML configuration file. |