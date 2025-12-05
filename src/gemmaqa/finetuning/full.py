"""
Full finetuning model loader.
Loads the base model without any adapters for standard finetuning.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from gemmaqa.config import QAConfig
from gemmaqa.utils.utils import get_logger

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
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    
    # Quantization config for memory efficiency on 8GB GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Enable gradient checkpointing for memory efficiency
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model parameters",
        trainable=f"{trainable_params:,}",
        total=f"{total_params:,}",
        trainable_pct=f"{100 * trainable_params / total_params:.2f}%"
    )
    
    return model, tokenizer
