"""
Model loading and simple utilities: load base QA model, apply freezing and/or LoRA.
"""

import logging
from typing import Literal

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_base_model(model_name_or_path: str, cache_dir: str | None = None):
    """Load a pretrained QA model and tokenizer."""
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name_or_path, cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    logger.info("Loaded model %s", model_name_or_path)
    return model, tokenizer


def prepare_model(model_name_or_path: str, mode: Literal["full", "freeze", "lora"]):
    model, tokenizer = load_base_model(model_name_or_path)
    if mode == "freeze":
        pass  # TODO
    elif mode == "lora":
        pass  # TODO
    else:
        logger.info("Using full finetuning.")

    return model, tokenizer
