"""
Model loading and simple utilities
"""


from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from gemmaqa.utils import get_logger

logger = get_logger(__name__)


def load_base_model(model_name_or_path: str, cache_dir: str | None = None):
    """Load a pretrained QA model and tokenizer."""
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name_or_path, cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    return model, tokenizer


def prepare_model(model_name_or_path: str):
    model, tokenizer = load_base_model(model_name_or_path)

    return model, tokenizer
