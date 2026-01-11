"""
Data loading and processing for QA finetuning.
"""

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer


def load_raw_dataset(
    data_path: str | None = None, num_samples: int | None = None, split: str = "train"
) -> Dataset:
    """
    Helper to load dataset without tokenization.
    """
    # 1. Load
    if data_path:
        dataset = load_dataset("json", data_files=data_path, split="train")
    else:
        dataset = load_dataset("squad", split=split)

    # 2. Slice (optional)
    if num_samples is not None:
        # Zabezpieczenie przed wzięciem więcej próbek niż istnieje
        real_limit = min(num_samples, len(dataset))
        dataset = dataset.select(range(real_limit))

    return dataset

def preprocess_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizer, max_length: int = 512) -> Dataset:
    """
    Formats and tokenizes the dataset for Trainer.
    """
    
    def tokenize_function(examples):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for i in range(len(examples['context'])):
            context = examples['context'][i]
            question = examples['question'][i]
            
            if isinstance(examples['answers'][i], dict) and examples['answers'][i]['text']:
                answer = examples['answers'][i]['text'][0]
            else:
                answer = ""

            messages_user = [
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
            ]
            user_tokens = tokenizer.apply_chat_template(messages_user, tokenize=True, add_generation_prompt=True)
            
            messages_full = messages_user + [{"role": "model", "content": answer}]
            full_tokens = tokenizer.apply_chat_template(messages_full, tokenize=True)

            labels = full_tokens[:]
            for j in range(len(user_tokens)):
                labels[j] = -100
            
            if len(full_tokens) > max_length:
                continue
            
            model_inputs["input_ids"].append(full_tokens)
            model_inputs["labels"].append(labels)
            model_inputs["attention_mask"].append([1] * len(full_tokens))

        return model_inputs
        

    processed_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return processed_dataset


def load_train_and_eval_data(
    tokenizer: PreTrainedTokenizer,
    train_data_path: str | None = None,
    val_data_path: str | None = None,
    train_samples: int | None = None,
    val_samples: int | None = None,
    max_length: int = 512,
) -> dict[str, Dataset]:
    """
    Loads raw datasets and tokenizes them immediately.
    """

    # Load Train
    raw_train = load_raw_dataset(
        data_path=train_data_path, num_samples=train_samples, split="train"
    )

    # Load Validation
    if val_data_path:
        raw_eval = load_raw_dataset(
            data_path=val_data_path, num_samples=val_samples, split="train"
        )
    else:
        raw_eval = load_raw_dataset(
            data_path=None, num_samples=val_samples, split="validation"
        )

    train_dataset = preprocess_dataset(raw_train, tokenizer, max_length)
    eval_dataset = preprocess_dataset(raw_eval, tokenizer, max_length)

    return {
        "train": train_dataset,
        "eval": eval_dataset,
    }

