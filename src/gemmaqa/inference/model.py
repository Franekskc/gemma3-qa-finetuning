"""
Model loading utilities for inference.
"""

import torch
from transformers import AutoTokenizer
from peft import PeftModel

from gemmaqa.finetuning.base import load_base_model
from gemmaqa.utils import get_logger

logger = get_logger(__name__)


def load_model_for_inference(
    checkpoint_path: str | None = None,
    base_model_name: str = "google/gemma-3-1b-it",
    is_lora: bool = True,
):
    """
    Load a model for inference.
    
    Args:
        checkpoint_path: Path to saved model/adapter. If None, loads base model only.
        base_model_name: Base model name.
        is_lora: Whether the checkpoint is a LoRA adapter.
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Loading model for inference", base_model=base_model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = load_base_model(base_model_name, quantize=True)
    
    if checkpoint_path and is_lora:
        logger.info("Loading LoRA adapter", path=checkpoint_path)
        model = PeftModel.from_pretrained(model, checkpoint_path)
    
    model.eval()
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 0.7,
    max_new_tokens: int = 50,
    top_p: float = 0.9,
) -> str:
    """
    Generate a response for a given prompt.
    
    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        prompt: User prompt.
        temperature: Generation temperature.
        max_new_tokens: Maximum new tokens to generate.
        top_p: Top-p sampling parameter.
        
    Returns:
        Generated response text.
    """
    messages = [{"role": "user", "content": prompt}]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=terminators
        )
    
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True).strip()
