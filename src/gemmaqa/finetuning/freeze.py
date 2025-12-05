"""
Layer freezing model loader.
Loads the base model and freezes all layers except the last N.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from gemmaqa.config import QAConfig
from gemmaqa.utils.utils import get_logger

logger = get_logger(__name__)

# Default number of layers to keep trainable (from the end)
DEFAULT_TRAINABLE_LAYERS = 4


def get_freeze_model(cfg: QAConfig, num_trainable_layers: int = DEFAULT_TRAINABLE_LAYERS):
    """
    Loads the base model with layer freezing for efficient finetuning.
    
    Freezes all layers except the last `num_trainable_layers` layers,
    which remain trainable for task-specific adaptation.
    
    Args:
        cfg: QAConfig with model_name and training settings.
        num_trainable_layers: Number of layers from the end to keep trainable.
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(
        "Loading model with layer freezing",
        model=cfg.model_name,
        trainable_layers=num_trainable_layers
    )
    
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
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last N layers
    # Gemma models have layers in model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        total_layers = len(layers)
        
        # Unfreeze the last num_trainable_layers
        for i in range(max(0, total_layers - num_trainable_layers), total_layers):
            for param in layers[i].parameters():
                param.requires_grad = True
        
        logger.info(
            "Layers configured",
            total_layers=total_layers,
            frozen_layers=total_layers - num_trainable_layers,
            trainable_layers=num_trainable_layers
        )
    else:
        logger.warning("Could not find model layers for freezing, all parameters remain frozen")
    
    # Also unfreeze the LM head for output adaptation
    if hasattr(model, 'lm_head'):
        for param in model.lm_head.parameters():
            param.requires_grad = True
        logger.info("LM head unfrozen")
    
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
