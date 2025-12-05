from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def format_squad_example(example):
    """Formats a SQuAD example into a prompt for Causal LM."""
    context = example['context']
    question = example['question']
    answer = example['answers']['text'][0] if example['answers']['text'] else ""
    
    # Simple prompt format
    text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}"
    return {"text": text}

def load_and_process_data(tokenizer, data_path=None, num_samples=None):
    """
    Loads SQuAD dataset (from HF or local JSON), selects a subset, and formats it for training.
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
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)
    
    return tokenized_datasets
