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

def load_and_process_data(tokenizer, num_samples=100):
    """
    Loads SQuAD dataset, selects a subset, and formats it for training.
    """
    # Load dataset
    dataset = load_dataset("squad", split="train")
    
    # Select subset
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Format data
    dataset = dataset.map(format_squad_example)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # We need 'labels' for Trainer to compute loss. 
    # For Causal LM, labels are usually the same as input_ids.
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)
    
    return tokenized_datasets
