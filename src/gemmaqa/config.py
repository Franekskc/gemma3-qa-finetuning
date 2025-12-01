import yaml
import copy
from dataclasses import dataclass
from typing import Literal

@dataclass
class LoraConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]

@dataclass
class DataConfig:
    max_train_samples: int | None
    val_samples: int
    max_seq_len: int
    doc_stride: int

@dataclass
class TrainingConfig:
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
    mode: Literal["full", "freeze", "lora"]
    model_name: str
    seed: str
    data: DataConfig
    training: TrainingConfig
    adapter: LoraConfig | None

    @classmethod
    def load(cls, yaml_path: str, selected_mode: str):
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
        seed = final_config_dict.pop("seed")

        adapter_cfg = None
        if "adapter" in final_config_dict:
            adapter_cfg = LoraConfig(**final_config_dict.pop("adapter"))
        
        return cls(
            mode=selected_mode,
            model_name=model_name,
            seed=seed,
            data=DataConfig(**final_config_dict["data"]),
            training=TrainingConfig(**final_config_dict["training"]),
            adapter=adapter_cfg
        )