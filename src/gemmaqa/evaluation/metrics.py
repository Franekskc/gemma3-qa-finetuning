"""
QA evaluation metrics for SQuAD-style datasets.
"""

import re
import string
from collections import Counter

from gemmaqa.utils import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Simple metrics for generative QA (text comparison)
# -----------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.
    Lowercases, removes punctuation, articles, and extra whitespace.
    """

    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punctuation(s):
        return "".join(ch for ch in s if ch not in string.punctuation)

    def lower(s):
        return s.lower()

    def white_space_fix(s):
        return " ".join(s.split())

    return white_space_fix(remove_articles(remove_punctuation(lower(text))))


def compute_exact_match(prediction: str, ground_truths: list[str]) -> float:
    """
    Compute exact match score.

    Args:
        prediction: Model's predicted answer.
        ground_truths: List of acceptable ground truth answers.

    Returns:
        1.0 if prediction exactly matches any ground truth, else 0.0
    """
    normalized_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) == normalized_pred:
            return 1.0
    return 0.0


def compute_f1(prediction: str, ground_truths: list[str]) -> float:
    """
    Compute token-level F1 score.

    Args:
        prediction: Model's predicted answer.
        ground_truths: List of acceptable ground truth answers.

    Returns:
        Best F1 score across all ground truths.
    """

    def f1_single(pred: str, gt: str) -> float:
        pred_tokens = normalize_answer(pred).split()
        gt_tokens = normalize_answer(gt).split()

        if not pred_tokens or not gt_tokens:
            return float(pred_tokens == gt_tokens)

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    return max(f1_single(prediction, gt) for gt in ground_truths)
