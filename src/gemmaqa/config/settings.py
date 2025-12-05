"""
Configuration settings for gemmaqa.
Defines dataclasses for all configuration options loaded from YAML.
"""

import yaml
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


@dataclass
class LoraConfig:
    """LoRA adapter configuration."""
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]


@dataclass
class FreezeConfig:
    """Layer freezing configuration."""
    trainable_layers: int = 4  # Number of layers from the end to keep trainable


@dataclass
class DataConfig:
    """Data loading and processing configuration."""
    max_train_samples: int | None
    val_samples: int
    max_seq_len: int
    doc_stride: int


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    output_dir: str
    num_train_epochs: int
    learning_rate: float
    per_device_train_batch_size: int
    effective_batch_size: int
    weight_decay: float
    warmup_ratio: float
    early_stopping_patience: int
    logging_steps: int
    fp16: bool
    gradient_checkpointing: bool
    gradient_accumulation_steps: int
    save_total_limit: int


@dataclass
class QAConfig:
    """Main configuration container for QA finetuning."""
    mode: Literal["full", "freeze", "lora"]
    model_name: str
    seed: int
    data: DataConfig
    training: TrainingConfig
    adapter: LoraConfig | None = None
    freeze: FreezeConfig = field(default_factory=FreezeConfig)

    @classmethod
    def load(cls, yaml_path: str | Path | None = None, selected_mode: str = "full"):
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML config file. Defaults to default.yaml in this module.
            selected_mode: Training mode ('full', 'lora', or 'freeze').
            
        Returns:
            QAConfig instance with merged common and mode-specific settings.
        """
        if yaml_path is None:
            yaml_path = DEFAULT_CONFIG_PATH
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_yaml = yaml.safe_load(f)

        common_cfg = copy.deepcopy(raw_yaml.get("common", {}))
        mode_specific_cfg = raw_yaml.get("modes", {}).get(selected_mode, {})
        
        if not mode_specific_cfg and selected_mode not in ["full", "freeze", "lora"]:
            print(f"Warning: Mode '{selected_mode}' not found in YAML modes section.")

        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
            return base_dict

        final_config_dict = deep_update(common_cfg, mode_specific_cfg)

        model_name = final_config_dict.pop("model")
        seed = int(final_config_dict.pop("seed"))

        # Parse adapter config if present
        adapter_cfg = None
        if "adapter" in final_config_dict:
            adapter_cfg = LoraConfig(**final_config_dict.pop("adapter"))
        
        # Parse freeze config if present
        freeze_cfg = FreezeConfig()
        if "freeze" in final_config_dict:
            freeze_cfg = FreezeConfig(**final_config_dict.pop("freeze"))
        
        return cls(
            mode=selected_mode,
            model_name=model_name,
            seed=seed,
            data=DataConfig(**final_config_dict["data"]),
            training=TrainingConfig(**final_config_dict["training"]),
            adapter=adapter_cfg,
            freeze=freeze_cfg,
        )
