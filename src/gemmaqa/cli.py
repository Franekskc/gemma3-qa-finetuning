"""
Unified CLI for gemmaqa.
Provides subcommands for training, evaluation, chat, and data preparation.
"""

import argparse
import sys

from gemmaqa.config.settings import DEFAULT_CONFIG_PATH


def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        prog="gemmaqa",
        description="Gemma QA Finetuning Toolkit - Train, evaluate, and chat with Gemma models",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -------------------------------------------------------------------------
    # Train subcommand
    # -------------------------------------------------------------------------
    train_parser = subparsers.add_parser("train", help="Train/finetune a model")
    train_parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["full", "lora", "freeze"],
        help="Training mode: full (full finetuning), lora (LoRA adapters), freeze (layer freezing)",
    )
    train_parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG_PATH})",
    )
    train_parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to training data JSON (default: data/train_subset.json)",
    )
    train_parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data JSON (default: data/val_subset.json)",
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max training steps (for testing, overrides epochs)",
    )

    # -------------------------------------------------------------------------
    # Eval subcommand
    # -------------------------------------------------------------------------
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint or LoRA adapter. If not provided, the base model will be evaluated",
    )
    eval_parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-3-1b-it",
        help="Base model name (default: google/gemma-3-1b-it)",
    )
    eval_parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to test data JSON (default: data/test-subset.json)",
    )
    eval_parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=5,
        help="Number of samples to evaluate (default: 5)",
    )
    eval_parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature (default: 0.1)",
    )

    # -------------------------------------------------------------------------
    # Chat subcommand
    # -------------------------------------------------------------------------
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint"
    )
    chat_parser.add_argument(
        "--base-model", type=str, default="google/gemma-3-1b-it", help="Base model name"
    )
    chat_parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="Single question (non-interactive)",
    )
    chat_parser.add_argument(
        "--context", "-c", type=str, default=None, help="Context for question"
    )

    # -------------------------------------------------------------------------
    # Data subcommand
    # -------------------------------------------------------------------------
    data_parser = subparsers.add_parser("prepare-data", help="Prepare SQuAD dataset")
    data_parser.add_argument(
        "--output", "-o", type=str, default="data", help="Output directory"
    )
    data_parser.add_argument(
        "--train-size", type=int, default=4000, help="Training samples"
    )
    data_parser.add_argument("--test-size", type=int, default=1000, help="Test samples")

    # -------------------------------------------------------------------------
    # Check CUDA
    # -------------------------------------------------------------------------
    subparsers.add_parser("check-cuda", help="Check CUDA availability")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # --- LOGIC ROUTING ---
    if args.command == "train":
        from gemmaqa.config import QAConfig
        from gemmaqa.finetuning.trainer import run_training
        from gemmaqa.utils import configure_logging

        configure_logging()
        config_path = args.config
        cfg = QAConfig.load(config_path, args.mode)

        run_training(
            cfg,
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            max_steps=args.max_steps,
        )

    elif args.command == "eval":
        from gemmaqa.evaluation.evaluation_runner import (
            load_model_for_eval,
            run_evaluation,
        )
        from gemmaqa.utils import configure_logging

        configure_logging()
        model, tokenizer = load_model_for_eval(
            checkpoint_path=args.checkpoint,
            base_model_name=args.base_model,
        )
        run_evaluation(
            model,
            tokenizer,
            num_samples=args.num_samples,
            data_path=args.data,
            temperature=args.temperature,
        )

    elif args.command == "chat":
        from gemmaqa.inference.chat import run_chat, run_single_query
        from gemmaqa.inference.model import load_model_for_inference
        from gemmaqa.utils import configure_logging

        configure_logging()
        model, tokenizer = load_model_for_inference(
            checkpoint_path=args.checkpoint,
            base_model_name=args.base_model,
        )

        if args.question:
            answer = run_single_query(model, tokenizer, args.question, args.context)
            print(f"Answer: {answer}")
        else:
            run_chat(model, tokenizer)

    elif args.command == "prepare-data":
        from gemmaqa.data.prepare import prepare_dataset

        prepare_dataset(
            output_dir=args.output,
            train_size=args.train_size,
            test_size=args.test_size,
        )

    elif args.command == "check-cuda":
        from gemmaqa.utils.cuda import main as cuda_main

        cuda_main()
        