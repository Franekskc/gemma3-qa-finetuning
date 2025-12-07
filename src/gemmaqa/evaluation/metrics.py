"""
QA evaluation metrics for SQuAD-style datasets.
"""

import re
import string
from collections import Counter, defaultdict

import evaluate
import numpy as np

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
        return re.sub(r'\b(a|an|the)\b', ' ', s)
    
    def remove_punctuation(s):
        return ''.join(ch for ch in s if ch not in string.punctuation)
    
    def lower(s):
        return s.lower()
    
    def white_space_fix(s):
        return ' '.join(s.split())
    
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


def compute_metrics_fn(eval_pred, examples, features) -> dict[str, float]:
    """
    Robust compute_metrics for Trainer.
    
    Args:
        eval_pred: EvalPrediction from Trainer.
        examples: Original dataset examples.
        features: Tokenized features with offset mappings.
        
    Returns:
        Dict with 'em' (exact match) and 'f1' scores.
    """
    safe_ret = {"em": 0.0, "f1": 0.0}
    predictions, _ = eval_pred

    if isinstance(predictions, tuple):
        start_logits, end_logits = predictions
    else:
        start_logits = predictions[:, :, 0]
        end_logits = predictions[:, :, 1]

    try:
        # Postprocess to get SQuAD-format lists
        preds_for_eval, refs_for_eval = postprocess_qa_predictions_for_eval(
            examples=examples,
            features=features,
            raw_predictions=(start_logits, end_logits),
        )
    except Exception as e:
        logger.exception("compute_metrics_fn failed during evaluation", error=str(e))
        return safe_ret

    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=preds_for_eval, references=refs_for_eval)

    # Return metrics
    return {"em": results["exact_match"], "f1": results["f1"]}


def postprocess_qa_predictions_for_eval(
    examples, features, raw_predictions, n_best_size=20, max_answer_length=30
) -> tuple:
    """
    Convert raw model logits to SQuAD-formatted prediction & reference lists.
    
    Args:
        examples: Original dataset examples.
        features: Tokenized features with offset mappings.
        raw_predictions: Tuple of (start_logits, end_logits).
        n_best_size: Number of best predictions to consider.
        max_answer_length: Maximum answer length in characters.

    Returns:
        A tuple with two elements:
        - list of {"id": id, "prediction_text": text}
        - list of {"id": id, "answers": {"text": [...], "answer_start": [...]}}
    """
    all_start_logits, all_end_logits = raw_predictions

    features_per_example = defaultdict(list)
    for i, f in enumerate(features):
        features_per_example[f["example_id"]].append(i)

    preds_for_eval = []
    refs_for_eval = []

    for example in examples:
        example_id = example["id"]
        context = example.get("context", "")
        feature_indices = features_per_example.get(example_id, [])
        best_answers = []

        for fi in feature_indices:
            start_logits = all_start_logits[fi]
            end_logits = all_end_logits[fi]
            offsets = features[fi]["offset_mapping"]

            start_idxs = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_idxs = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for si in start_idxs:
                for ei in end_idxs:
                    if si >= len(offsets) or ei >= len(offsets):
                        continue
                    if offsets[si] == (0, 0) or offsets[ei] == (0, 0):
                        continue
                    if ei < si:
                        continue
                    length = offsets[ei][1] - offsets[si][0]
                    if length > max_answer_length:
                        continue

                    score = float(start_logits[si] + end_logits[ei])
                    start_char = offsets[si][0]
                    end_char = offsets[ei][1]
                    text = context[start_char:end_char]
                    best_answers.append({"text": text, "score": score})

        if best_answers:
            best = max(best_answers, key=lambda x: x["score"])
            pred_text = best["text"]
        else:
            pred_text = ""  # no-answer

        preds_for_eval.append({"id": example_id, "prediction_text": pred_text})

        # build refs
        answers = example.get("answers", {})
        answer_texts = (
            answers.get("text", [])
            if isinstance(answers.get("text", []), (list, tuple))
            else [str(answers.get("text", ""))]
        )
        answer_starts = (
            answers.get("answer_start", [])
            if isinstance(answers.get("answer_start", []), (list, tuple))
            else [int(answers.get("answer_start", 0))]
        )
        refs_for_eval.append(
            {
                "id": example_id,
                "answers": {"text": answer_texts, "answer_start": answer_starts},
            }
        )

    return preds_for_eval, refs_for_eval
