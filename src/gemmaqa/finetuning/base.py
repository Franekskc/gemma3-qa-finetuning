"""
Base model loading utilities shared across finetuning strategies.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

from gemmaqa.utils import get_logger

logger = get_logger(__name__)


def get_bnb_config() -> BitsAndBytesConfig:
    """
    Get standard 4-bit quantization config for memory-efficient loading.
    
    Returns:
        BitsAndBytesConfig for 4-bit quantization.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load tokenizer for the specified model.
    
    Args:
        model_name: HuggingFace model name or path.
        
    Returns:
        Loaded AutoTokenizer.
    """
    logger.info("Loading tokenizer", model=model_name)
    return AutoTokenizer.from_pretrained(model_name)


def load_base_model(
    model_name: str,
    quantize: bool = True,
    device_map: str = "auto",
) -> PreTrainedModel:
    """
    Load base model with optional quantization.
    
    Args:
        model_name: HuggingFace model name or path.
        quantize: Whether to use 4-bit quantization.
        device_map: Device mapping strategy.
        
    Returns:
        Loaded model.
    """
    logger.info("Loading base model", model=model_name, quantize=quantize)

    bnb_config = None
    
    if quantize:
        bnb_config = get_bnb_config()
    
    return  AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16,
    )


def log_trainable_params(model) -> dict:
    """
    Log and return trainable parameter statistics.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Dict with trainable, total, and percentage.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable_params / total_params if total_params > 0 else 0
    
    logger.info(
        "Model parameters",
        trainable=f"{trainable_params:,}",
        total=f"{total_params:,}",
        trainable_pct=f"{pct:.2f}%"
    )
    
    return {
        "trainable": trainable_params,
        "total": total_params,
        "percentage": pct,
    }
