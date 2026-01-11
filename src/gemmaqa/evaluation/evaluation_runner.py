"""
Evaluation runner for trained models.
Runs inference on test data and computes metrics.
"""

import random
import json
from datetime import datetime
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
    checkpoint_path: str | None = None,
    num_samples: int = 5,
    data_path: str = "data/test_subset.json",
    max_new_tokens: int = 50,
    retriever = None,
    k: int = 3,
    output_dir: str | Path | None = None,
):
    """
    Run evaluation on random samples.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        checkpoint_path: Path to checkpoint (for saving results).
        num_samples: Number of samples to evaluate.
        data_path: Optional path to test data JSON.
        max_new_tokens: Maximum new tokens to generate.
        retriever: Optional RAG retriever instance. If provided, runs RAG evaluation.
        k: Number of contexts to retrieve for RAG.
        output_dir: Custom directory to save results to. If None, tries to use checkpoint_path parent.
    """
    # Load dataset
    if data_path:
        dataset = load_dataset("json", data_files=data_path, split="train")
    else:
        dataset = load_dataset("squad", split="validation")

    # Select random samples
    real_num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), real_num_samples)
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
        tqdm(examples, desc=f"Evaluating on {real_num_samples} samples...", unit="sample")
    ):
        context = example["context"]
        question = example["question"]
        ground_truth_answers = example["answers"]["text"]

        logger.debug("Context: {}".format(context))
        logger.debug("Question: {}".format(question))
        logger.debug("Ground truth answers: {}".format(ground_truth_answers))

        if retriever:
            # RAG Generation
            from gemmaqa.rag.generator import generate_rag_response
            
            # We ignore the 'context' from the dataset and use 'retriever'
            model_answer, retrieved_ctxs = generate_rag_response(
                model=model,
                tokenizer=tokenizer,
                question=question,
                retriever=retriever,
                k=k,
                max_new_tokens=max_new_tokens
            )


            # Optional: Print retrieved context titles for debugging
            logger.debug(f"Retrieved: {[c['title'] for c in retrieved_ctxs]}")
            logger.debug(f"Model answer: {model_answer}")

        else:
            # Standard Generation
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
                    do_sample=False,
                    eos_token_id=terminators,
                )

            response = outputs[0][input_ids.shape[-1] :]
            model_answer = tokenizer.decode(response, skip_special_tokens=True).strip()

        # Calculate metrics for this sample
        em_score = compute_exact_match(model_answer, ground_truth_answers)
        f1_score = compute_f1(model_answer, ground_truth_answers)
        all_em_scores.append(em_score)
        all_f1_scores.append(f1_score)

    # Print aggregate scores
    avg_em = sum(all_em_scores) / len(all_em_scores) * 100
    avg_f1 = sum(all_f1_scores) / len(all_f1_scores) * 100

    print("\n" + "=" * 60)
    print(f"AGGREGATE SCORES ({real_num_samples} samples)")
    print("=" * 60)
    print(f"  Exact Match:  {avg_em:.1f}%")
    print(f"  F1 Score:     {avg_f1:.1f}%")
    print("=" * 60)

    # Save the results
    # Save the results
    final_output_dir = None
    
    if output_dir:
        final_output_dir = Path(output_dir)
    elif checkpoint_path:
        final_output_dir = Path(checkpoint_path).parent

    if final_output_dir:
        final_output_dir.mkdir(parents=True, exist_ok=True)
        results_file = final_output_dir / "eval_results.json"
        
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": real_num_samples,
            "exact_match": round(avg_em, 4),
            "f1_score": round(avg_f1, 4),
            "data_source": data_path,
            "mode": "rag" if retriever else "standard",
            "retriever_k": k if retriever else None
        }

        try:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=4)
            
            logger.info(f"Results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    else:
        logger.warning("No output directory or checkpoint path provided. Results not saved to file.")
