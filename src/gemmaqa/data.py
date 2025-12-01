"""
Utilities to load SQuAD and prepare tokenized features for QA.
"""

from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from gemmaqa.utils import get_logger

logger = get_logger(__name__)


def load_squad(train_samples, val_samples, seed) -> DatasetDict:
    """Load SQuAD dataset and prepare a custom split.

    The splitting scheme is as follows:
    - train: Randomly selected `train_samples` from the original training set.
    - validation: Randomly selected `val_samples` from the original training set
        (disjoint from the new train set).
    - test: The full official SQuAD validation set.
    """
    logger.info("Loading SQuAD dataset from Hugging Face Hub")
    raw_ds = load_dataset("squad")
    
    total_samples_needed = train_samples + val_samples
    
    max_available = len(raw_ds["train"])
    if total_samples_needed > max_available:
        logger.warning(
            "Requested more samples than available",
            requested=total_samples_needed,
            available=max_available,
            action="Using all available samples"
        )
        total_samples_needed = max_available

    shuffled_subset = raw_ds["train"].shuffle(seed=seed).select(range(total_samples_needed))

    split_ds = shuffled_subset.train_test_split(test_size=val_samples, seed=seed)

    final_ds = DatasetDict({
        "train": split_ds["train"],
        "validation": split_ds["test"],
        "test": raw_ds["validation"]
    })

    logger.info(
        "Data split created successfully",
        train_size=len(final_ds['train']),
        internal_val_size=len(final_ds['validation']),
        official_test_size=len(final_ds['test'])
    )
    
    return final_ds


def prepare_train_features(
    examples: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    doc_stride: int
) -> dict[str, Any]:
    """Tokenize examples for training.

    Produces input_ids, attention_mask, offset_mapping and start_positions/end_positions.
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # If no answers, set to CLS index
        if len(answers["answer_start"]) == 0:
            start_positions.append(
                tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
            )
            end_positions.append(
                tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
            )
            continue

        answer_start = answers["answer_start"][0]
        answer_text = answers["text"][0]
        answer_end = answer_start + len(answer_text)

        # Find token start and end indices in the tokenized span
        token_start_index = 0
        while token_start_index < len(offsets) and sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(offsets) - 1
        while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # If the answer is not fully inside the span, label (cls, cls)
        if not (
            offsets[token_start_index][0] <= answer_start
            and offsets[token_end_index][1] >= answer_end
        ):
            start_positions.append(
                tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
            )
            end_positions.append(
                tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
            )
        else:
            # Otherwise, find the exact token indices
            while (
                token_start_index < len(offsets)
                and offsets[token_start_index][0] <= answer_start
            ):
                token_start_index += 1
            start_positions.append(token_start_index - 1)

            while offsets[token_end_index][1] >= answer_end:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples


def prepare_validation_features(
    examples: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    doc_stride: int
) -> dict[str, Any]:
    """Tokenize examples for validation/prediction.
    
    Keep offset_mapping for postprocessing.
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # Save example id to map predictions back to original examples
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set offsets to None for question tokens, keep for context tokens
        offsets = tokenized_examples["offset_mapping"][i]
        tokenized_examples["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else (0, 0) for k, o in enumerate(offsets)
        ]

    return tokenized_examples
