"""
LoRA finetuning model loader.
Loads the base model and applies LoRA adapters for parameter-efficient finetuning.
"""

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from gemmaqa.config import QAConfig
from gemmaqa.utils import get_logger
from gemmaqa.finetuning.base import load_base_model, load_tokenizer

logger = get_logger(__name__)


def get_lora_model(cfg: QAConfig):
    """
    Loads the base model and applies LoRA configuration with 4-bit quantization.
    
    Args:
        cfg: QAConfig with model_name, adapter settings, and training config.
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Loading LoRA model", model=cfg.model_name)
    
    if cfg.adapter is None:
        raise ValueError("LoRA mode requires adapter configuration in config.yaml")
    
    # Load tokenizer
    tokenizer = load_tokenizer(cfg.model_name)
    
    # Load base model with quantization
    model = load_base_model(cfg.model_name, quantize=True)
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.training.gradient_checkpointing, )
    
    # Configure LoRA from config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.adapter.r,
        lora_alpha=cfg.adapter.lora_alpha,
        lora_dropout=cfg.adapter.lora_dropout,
        target_modules=cfg.adapter.target_modules
    )
    
    logger.info(
        "LoRA config",
        r=cfg.adapter.r,
        alpha=cfg.adapter.lora_alpha,
        dropout=cfg.adapter.lora_dropout,
        targets=cfg.adapter.target_modules
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()
    
    return model, tokenizer
