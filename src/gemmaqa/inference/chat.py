"""
Interactive chat interface for trained models.
"""


from gemmaqa.inference.model import generate_response
from gemmaqa.utils import get_logger

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
