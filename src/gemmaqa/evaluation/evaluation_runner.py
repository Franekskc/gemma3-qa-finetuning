"""
Evaluation runner for trained models.
Runs inference on test data and computes metrics.
"""

import random
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer

from gemmaqa.evaluation.metrics import compute_exact_match, compute_f1
from gemmaqa.finetuning.base import load_base_model
from gemmaqa.utils import get_logger

logger = get_logger(__name__)


def load_model_for_eval(
    checkpoint_path: str,
    base_model_name: str = "google/gemma-3-1b-it",
):
    """
    Load a trained model for evaluation.
    Automatically detects if the checkpoint is a LoRA adapter or a full model.

    Args:
        checkpoint_path: Path to saved model/adapter.
        base_model_name: Base model name (used only if loading LoRA).

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Inspecting checkpoint", path=checkpoint_path)

    if checkpoint_path:
        is_lora = (Path(checkpoint_path) / "adapter_config.json").exists()

        if is_lora:
            logger.info("Detected LoRA adapter structure.")

            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model = load_base_model(base_model_name, quantize=True)

            logger.info("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, checkpoint_path)
        else:
            logger.info("Detected Full/Freeze model structure.")

            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            model = load_base_model(checkpoint_path, quantize=False)
    else:
        logger.info(
            "No custom model configuration. Loading base model.",
            base_model=base_model_name,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = load_base_model(base_model_name, quantize=True)

    # Padding for Gemma
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_evaluation(
    model,
    tokenizer,
    num_samples: int = 5,
    data_path: str = "data/test_subset.json",
    temperature: float = 0.1,
    max_new_tokens: int = 50,
):
    """
    Run evaluation on random samples.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        num_samples: Number of samples to evaluate.
        data_path: Optional path to test data JSON.
        temperature: Generation temperature.
        max_new_tokens: Maximum new tokens to generate.
    """
    # Load dataset
    if data_path:
        dataset = load_dataset("json", data_files=data_path, split="train")
    else:
        dataset = load_dataset("squad", split="validation")

    # Select random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    examples: Dataset = dataset.select(indices)

    print("=" * 60 + "\n")

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>"),
    ]

    # Track scores
    all_em_scores = []
    all_f1_scores = []

    for i, example in enumerate(
        tqdm(examples, desc=f"Evaluating on {len(examples)} samples...", unit="sample")
    ):
        context = example["context"]
        question = example["question"]
        ground_truth_answers = example["answers"]["text"]

        # Format prompt using chat template
        messages = [
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                eos_token_id=terminators,
            )

        response = outputs[0][input_ids.shape[-1] :]
        model_answer = tokenizer.decode(response, skip_special_tokens=True).strip()

        # Calculate metrics for this sample
        em_score = compute_exact_match(model_answer, ground_truth_answers)
        f1_score = compute_f1(model_answer, ground_truth_answers)
        all_em_scores.append(em_score)
        all_f1_scores.append(f1_score)

        # print(f"[{i+1}/{len(examples)}] Question: {question}")
        # print(f"    Context: {context[:80]}...")
        # print(f"    Ground Truth: {ground_truth_answers}")
        # print(f"    Model Answer: {model_answer}")
        # print(f"    EM: {em_score:.0f}  |  F1: {f1_score:.2f}")
        # print("-" * 60)

    # Print aggregate scores
    avg_em = sum(all_em_scores) / len(all_em_scores) * 100
    avg_f1 = sum(all_f1_scores) / len(all_f1_scores) * 100

    print("\n" + "=" * 60)
    print(f"AGGREGATE SCORES ({len(examples)} samples)")
    print("=" * 60)
    print(f"  Exact Match:  {avg_em:.1f}%")
    print(f"  F1 Score:     {avg_f1:.1f}%")
    print("=" * 60)
