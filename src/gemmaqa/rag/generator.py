"""
RAG Generator module.
Handles prompt construction and response generation.
"""

from gemmaqa.inference.model import generate_response
from gemmaqa.rag.retriever import GemmaQARetriever


def format_rag_prompt(question: str, target_context: str, contexts: list[dict]) -> str:
    """
    Format the RAG prompt with retrieved contexts as few-shot examples.
    
    Args:
        question: User question.
        target_context: The context to extract the answer from.
        contexts: List of retrieved context dicts (from retriever) to serve as examples.
        
    Returns:
        Formatted prompt string.
    """
    examples_str = ""
    for i, ctx in enumerate(contexts):
        examples_str += f"### Example {i+1}\n{ctx['text']}\n\n"
    
    prompt = (
        "You are a helpful assistant. Your task is to extract the answer to the QUESTION based on the provided CONTEXT.\n"
        "Follow the examples below in terms of how to extract the knowledge from the context and correctly format the responses.\n"
        "If the answer is not in the context, say 'I don't know'.\n\n"
        "--- EXAMPLES START ---\n\n"
        f"{examples_str}"
        "--- EXAMPLES END ---\n\n"
        "Now, please perform the task for the following:\n\n"
        f"Context: {target_context}\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return prompt


def generate_rag_response(
    model,
    tokenizer,
    question: str,
    target_context: str,
    retriever: GemmaQARetriever,
    k: int = 3,
    temperature: float = 0.5,
    max_new_tokens: int = 100,
) -> tuple[str, list[dict]]:
    """
    Generate a response using RAG.
    
    Args:
        model: Loaded LLM.
        tokenizer: Loaded tokenizer.
        question: User question.
        target_context: The context to extract answer from.
        retriever: Initialized and loaded GemmaQARetriever.
        k: Number of contexts to retrieve.
        temperature: Generation temperature.
        max_new_tokens: Max tokens to generate.
        
    Returns:
        Tuple of (generated_answer, retrieved_contexts)
    """
    contexts = retriever.retrieve(question, k=k)
    
    prompt = format_rag_prompt(question, target_context, contexts)
    
    answer = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    
    return answer, contexts
