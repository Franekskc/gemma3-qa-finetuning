# Gemma 3 QA Fine-Tuning (SQuAD)
Experiments comparing **Full FT**, **LoRA (PEFT)**, and **Layer Freezing** for **extractive QA** on **SQuAD 1.1** with **Gemma 3 (~4B)**. Designed for a single GPU (RTX 3070 Ti, 8 GB).


## Dependencies installation:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync
```
