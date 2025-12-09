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
        texts = []
        
        for i in range(len(examples['context'])):
            context = examples['context'][i]
            question = examples['question'][i]
            
            answers = examples['answers'][i]
            if isinstance(answers, dict):
                ans_list = answers.get('text', [])
                answer = ans_list[0] if len(ans_list) > 0 else ""
            else:
                answer = ""

            messages = [
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
                {"role": "model", "content": answer}
            ]
            
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            texts.append(full_text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized

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

