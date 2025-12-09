"""
Full finetuning model loader.
Loads the base model without any adapters for standard finetuning.
"""

import torch
from gemmaqa.config import QAConfig
from gemmaqa.utils import get_logger
from gemmaqa.finetuning.base import load_base_model, load_tokenizer, log_trainable_params

logger = get_logger(__name__)


def get_full_model(cfg: QAConfig):
    """
    Loads the base model with 4-bit quantization for memory-efficient full finetuning.
    
    Args:
        cfg: QAConfig with model_name and training settings.
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Loading full finetuning model", model=cfg.model_name)
    
    # Load tokenizer
    tokenizer = load_tokenizer(cfg.model_name)

    target_dtype = torch.bfloat16
    
    # Load base model with quantization
    model = load_base_model(cfg.model_name, quantize=False, dtype=target_dtype)
    
    # Enable gradient checkpointing for memory efficiency
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Log trainable parameters
    log_trainable_params(model)
    
    return model, tokenizer
