# Configuration Guide

This directory contains configuration files for the Gemma QA fine-tuning project.

## Parameters Affecting LoRA Execution Time

### 🔴 Highest Impact Parameters

#### `num_train_epochs`

Number of complete passes through the training dataset. Doubling epochs roughly doubles training time.

#### `max_train_samples`

Number of training samples to use. More samples = more iterations per epoch = longer training.

#### `per_device_train_batch_size` & `gradient_accumulation_steps`

Together these define the **effective batch size**. Larger batch sizes process more data per step, reducing total steps.
Limited by GPU memory.

#### `max_seq_len`

Maximum sequence length for tokenization. Longer sequences = more computation per forward/backward pass.

---

### 🟠 Moderate Impact (LoRA-Specific)

#### `r` (LoRA Rank)

The rank of low-rank matrices. Higher rank = more trainable parameters = slower training. Common values: 4-64.

#### `target_modules`

Specifies which layers receive LoRA adapters. More modules = more trainable parameters = longer training.

- Minimal: `[q_proj, v_proj]`
- Full: `[q_proj, k_proj, v_proj, o_proj]`

---

### 🟢 Lower Impact

#### `gradient_checkpointing`

When `true`, trades compute for memory by recomputing activations during backward pass. Slows training ~20% but enables
larger batches.

#### `fp16`

Mixed-precision training. Keep `true` for faster training on modern GPUs.

---

## Quick Reference

| Parameter                     | Impact           | How to Speed Up                  |
|-------------------------------|------------------|----------------------------------|
| `num_train_epochs`            | ⬆️⬆️⬆️ Very High | Reduce epochs                    |
| `max_train_samples`           | ⬆️⬆️⬆️ Very High | Use fewer samples                |
| `per_device_train_batch_size` | ⬆️⬆️ High        | Increase if GPU allows           |
| `max_seq_len`                 | ⬆️⬆️ High        | Reduce sequence length           |
| `r` (LoRA rank)               | ⬆️ Moderate      | Reduce rank                      |
| `target_modules`              | ⬆️ Moderate      | Use fewer modules                |
| `gradient_checkpointing`      | ⬆️ Moderate      | Set to `false` (needs more VRAM) |
