"""
Unified CLI for gemmaqa.
Provides subcommands for training, evaluation, chat, and data preparation.
"""

import argparse
import sys


def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        prog="gemmaqa",
        description="Gemma QA Finetuning Toolkit - Train, evaluate, and chat with Gemma models",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train/finetune a model")
    train_parser.add_argument(
        "--mode", "-m",
        type=str,
        required=True,
        choices=["full", "lora", "freeze"],
        help="Training mode"
    )
    train_parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config YAML"
    )
    train_parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data JSON"
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max training steps (for testing)"
    )
    
    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-3-1b-it",
        help="Base model name"
    )
    eval_parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=5,
        help="Number of samples"
    )
    eval_parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Checkpoint is full model"
    )
    
    # Chat subcommand
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    chat_parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-3-1b-it",
        help="Base model name"
    )
    chat_parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Single question (non-interactive)"
    )
    chat_parser.add_argument(
        "--context", "-c",
        type=str,
        default=None,
        help="Context for question"
    )
    
    # Prepare data subcommand
    data_parser = subparsers.add_parser("prepare-data", help="Prepare SQuAD dataset")
    data_parser.add_argument(
        "--output", "-o",
        type=str,
        default="data",
        help="Output directory"
    )
    data_parser.add_argument(
        "--train-size",
        type=int,
        default=4000,
        help="Training samples"
    )
    data_parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Test samples"
    )
    
    # Check CUDA subcommand
    subparsers.add_parser("check-cuda", help="Check CUDA availability")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate module
    if args.command == "train":
        from gemmaqa.config import QAConfig
        from gemmaqa.config.settings import DEFAULT_CONFIG_PATH
        from gemmaqa.utils import configure_logging
        
        configure_logging()
        config_path = args.config or str(DEFAULT_CONFIG_PATH)
        cfg = QAConfig.load(config_path, args.mode)
        
        from gemmaqa.finetuning.trainer import run_training
        run_training(cfg, train_data_path=args.data, max_steps=args.max_steps)
        
    elif args.command == "eval":
        from gemmaqa.evaluation.evaluation_runner import load_model_for_eval, run_evaluation
        from gemmaqa.utils import configure_logging
        
        configure_logging()
        model, tokenizer = load_model_for_eval(
            checkpoint_path=args.checkpoint,
            base_model_name=args.base_model,
            is_lora=not args.no_lora,
        )
        run_evaluation(model, tokenizer, num_samples=args.num_samples)
        
    elif args.command == "chat":
        from gemmaqa.inference.model import load_model_for_inference
        from gemmaqa.inference.chat import run_chat, run_single_query
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


if __name__ == "__main__":
    main()
