# Gemma 3 QA Fine-Tuning (SQuAD)
Experiments comparing **Full FT**, **LoRA (PEFT)**, and **Layer Freezing** for **extractive QA** on **SQuAD 1.1** with **Gemma 3 (~4B)**. Designed for a single GPU (RTX 3070 Ti, 8 GB).


## Dependencies installation:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync
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