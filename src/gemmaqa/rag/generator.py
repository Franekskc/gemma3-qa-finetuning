"""
RAG Generator module.
Handles prompt construction and response generation.
"""

from gemmaqa.inference.model import generate_response
from gemmaqa.rag.retriever import GemmaQARetriever


def format_rag_prompt(question: str, contexts: list[dict]) -> str:
    """
    Format the RAG prompt with retrieved contexts.
    
    Args:
        question: User question.
        contexts: List of retrieved context dicts (from retriever).
        
    Returns:
        Formatted prompt string.
    """
    context_str = ""
    for i, ctx in enumerate(contexts):
        context_str += f"Context {i+1}: {ctx['text']}\n\n"
    
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided context. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        f"{context_str}"
        f"Question: {question}"
    )
    return prompt


def generate_rag_response(
    model,
    tokenizer,
    question: str,
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
        retriever: Initialized and loaded GemmaQARetriever.
        k: Number of contexts to retrieve.
        temperature: Generation temperature.
        max_new_tokens: Max tokens to generate.
        
    Returns:
        Tuple of (generated_answer, retrieved_contexts)
    """
    contexts = retriever.retrieve(question, k=k)
    prompt = format_rag_prompt(question, contexts)
    
    answer = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    
    return answer, contexts
