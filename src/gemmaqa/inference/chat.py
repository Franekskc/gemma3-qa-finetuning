"""
Interactive chat interface for trained models.
"""

import argparse

from gemmaqa.inference.model import load_model_for_inference, generate_response
from gemmaqa.utils import configure_logging, get_logger

logger = get_logger(__name__)


def run_chat(
    model,
    tokenizer,
    temperature: float = 0.7,
    max_new_tokens: int = 50,
):
    """
    Run an interactive chat session.
    
    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        temperature: Generation temperature.
        max_new_tokens: Maximum new tokens to generate.
    """
    print("\n" + "=" * 50)
    print("Gemma QA Chat Interface")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 50 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            
            print(f"Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def run_single_query(
    model,
    tokenizer,
    question: str,
    context: str | None = None,
    temperature: float = 0.7,
    max_new_tokens: int = 50,
):
    """
    Run a single QA query.
    
    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        question: Question to ask.
        context: Optional context for the question.
        temperature: Generation temperature.
        max_new_tokens: Maximum new tokens to generate.
        
    Returns:
        Generated answer.
    """
    if context:
        prompt = f"Context: {context}\n\nQuestion: {question}"
    else:
        prompt = question
    
    return generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def main():
    """CLI entry point for chat interface."""
    parser = argparse.ArgumentParser(description="Interactive chat with trained Gemma QA model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint or LoRA adapter (default: base model only)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-3-1b-it",
        help="Base model name (default: google/gemma-3-1b-it)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum new tokens (default: 50)"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Checkpoint is a full model, not a LoRA adapter"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Single question to ask (non-interactive mode)"
    )
    parser.add_argument(
        "--context", "-c",
        type=str,
        default=None,
        help="Context for the question (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    configure_logging()
    
    model, tokenizer = load_model_for_inference(
        checkpoint_path=args.checkpoint,
        base_model_name=args.base_model,
        is_lora=not args.no_lora,
    )
    
    if args.question:
        # Single query mode
        answer = run_single_query(
            model=model,
            tokenizer=tokenizer,
            question=args.question,
            context=args.context,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
        )
        print(f"Answer: {answer}")
    else:
        # Interactive mode
        run_chat(
            model=model,
            tokenizer=tokenizer,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()