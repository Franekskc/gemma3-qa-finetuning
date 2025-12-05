"""
Evaluation runner for trained models.
Runs inference on test data and computes metrics.
"""

import argparse
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import PeftModel

from gemmaqa.finetuning.base import load_base_model
from gemmaqa.utils import configure_logging, get_logger

logger = get_logger(__name__)


def load_model_for_eval(
    checkpoint_path: str,
    base_model_name: str = "google/gemma-3-1b-it",
    is_lora: bool = True,
):
    """
    Load a trained model for evaluation.
    
    Args:
        checkpoint_path: Path to saved model/adapter.
        base_model_name: Base model name for LoRA adapters.
        is_lora: Whether the checkpoint is a LoRA adapter.
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Loading model for evaluation", checkpoint=checkpoint_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = load_base_model(base_model_name, quantize=True)
    
    if is_lora:
        logger.info("Loading LoRA adapter", path=checkpoint_path)
        model = PeftModel.from_pretrained(model, checkpoint_path)
    
    return model, tokenizer


def run_evaluation(
    model,
    tokenizer,
    num_samples: int = 5,
    data_path: str | None = None,
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
    examples = dataset.select(indices)

    print(f"\nEvaluating on {len(examples)} samples...\n")
    print("-" * 50)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]

    for example in examples:
        context = example['context']
        question = example['question']
        ground_truth_answers = example['answers']['text']

        # Format prompt using chat template
        messages = [
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                eos_token_id=terminators
            )

        response = outputs[0][input_ids.shape[-1]:]
        model_answer = tokenizer.decode(response, skip_special_tokens=True).strip()

        print(f"Question: {question}")
        print(f"Context (truncated): {context[:100]}...")
        print(f"Ground Truth: {ground_truth_answers}")
        print(f"Model Answer: {model_answer}")
        print("-" * 50)


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained Gemma QA model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint or LoRA adapter"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-3-1b-it",
        help="Base model name (default: google/gemma-3-1b-it)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to test data JSON (default: SQuAD validation)"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=5,
        help="Number of samples to evaluate (default: 5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature (default: 0.1)"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Checkpoint is a full model, not a LoRA adapter"
    )
    
    args = parser.parse_args()
    
    configure_logging()
    
    model, tokenizer = load_model_for_eval(
        checkpoint_path=args.checkpoint,
        base_model_name=args.base_model,
        is_lora=not args.no_lora,
    )
    
    run_evaluation(
        model=model,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        data_path=args.data,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
