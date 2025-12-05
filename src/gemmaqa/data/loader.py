"""
Data loading and processing for QA finetuning.
"""

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer


def format_squad_example(example: dict) -> dict:
    """
    Formats a SQuAD example into a prompt for Causal LM.
    
    Args:
        example: SQuAD example dict with 'context', 'question', 'answers' keys.
        
    Returns:
        Dict with 'text' key containing the formatted prompt.
    """
    context = example['context']
    question = example['question']
    answer = example['answers']['text'][0] if example['answers']['text'] else ""
    
    # Simple prompt format
    text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}"
    return {"text": text}


def load_and_process_data(
    tokenizer: PreTrainedTokenizer,
    data_path: str | None = None,
    num_samples: int | None = None,
    max_length: int = 512,
) -> Dataset:
    """
    Loads SQuAD dataset (from HuggingFace or local JSON), selects a subset, and formats it for training.
    
    Args:
        tokenizer: Tokenizer to use for tokenization.
        data_path: Optional path to local JSON file. If None, loads from HuggingFace.
        num_samples: Optional limit on number of samples to use.
        max_length: Maximum sequence length for tokenization.
        
    Returns:
        Tokenized dataset ready for training.
    """
    # Load dataset
    if data_path:
        dataset = load_dataset("json", data_files=data_path, split="train")
    else:
        dataset = load_dataset("squad", split="train")
    
    # Select subset if num_samples is specified
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Format data
    dataset = dataset.map(format_squad_example)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)
    
    return tokenized_datasets
