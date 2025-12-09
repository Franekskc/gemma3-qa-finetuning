"""
Layer freezing model loader.
Loads the base model and freezes all layers except the last N.
"""

import torch
from gemmaqa.config import QAConfig
from gemmaqa.utils import get_logger
from gemmaqa.finetuning.base import load_base_model, load_tokenizer, log_trainable_params

logger = get_logger(__name__)


def get_freeze_model(cfg: QAConfig):
    """
    Loads the base model with layer freezing for efficient finetuning.
    
    Freezes all layers except the last N layers (from cfg.freeze.trainable_layers),
    which remain trainable for task-specific adaptation.
    
    Args:
        cfg: QAConfig with model_name, training settings, and freeze config.
        
    Returns:
        Tuple of (model, tokenizer)
    """
    num_trainable_layers = cfg.freeze.trainable_layers
    
    logger.info(
        "Loading model with layer freezing",
        model=cfg.model_name,
        trainable_layers=num_trainable_layers
    )
    
    # Load tokenizer
    tokenizer = load_tokenizer(cfg.model_name)

    target_dtype = torch.bfloat16
    
    # Load base model with quantization
    model = load_base_model(cfg.model_name, quantize=False, dtype=target_dtype)
    
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
    
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        for param in model.model.norm.parameters():
            param.requires_grad = True
        logger.info("Final Layer Norm unfrozen")

    # Also unfreeze the LM head for output adaptation
    if hasattr(model, 'lm_head'):
        for param in model.lm_head.parameters():
            param.requires_grad = True
        logger.info("LM head unfrozen")
    
    # Enable gradient checkpointing for memory efficiency
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Log trainable parameters
    log_trainable_params(model)
    
    return model, tokenizer
