"""
Dataset preparation script.
Downloads SQuAD dataset and creates train/test subsets and corpus.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, load_dataset

from gemmaqa.utils import get_logger

logger = get_logger(__name__)


def prepare_dataset(
    output_dir: str | Path = "data",
    train_size: int = 4000,
    val_size_ratio: float = 0.1,
    test_size: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Prepare SQuAD dataset subsets for training and evaluation.

    Args:
        output_dir: Directory to save output files.
        train_size: Number of training samples to select.
        test_size: Number of test samples to select.
        seed: Random seed for reproducibility.

    Returns:
        Dict with paths to created files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    logger.info("Loading SQuAD dataset...")
    squad_train: Dataset = load_dataset("squad", split="train")
    squad_val = load_dataset("squad", split="validation")

    logger.info(f"Original Train size: {len(squad_train)}")
    logger.info(f"Original Val size: {len(squad_val)}")

    # 1. Train Subset
    full_train_indices = list(range(len(squad_train)))
    random.shuffle(full_train_indices)
    current_pool = full_train_indices[:train_size]

    # 2. Split Train into Train and Validation
    split_idx = int(len(current_pool) * (1 - val_size_ratio))

    my_train_indices = current_pool[:split_idx]
    my_val_indices = current_pool[split_idx:]

    train_subset = squad_train.select(my_train_indices)
    val_subset = squad_train.select(my_val_indices)

    # 3. Test Subset (z oficjalnego validation - model nigdy tego nie widział w treningu)
    val_indices = list(range(len(squad_val)))
    random.shuffle(val_indices)
    test_subset = squad_val.select(val_indices[:test_size])

    # 4. Corpus (unique contexts for RAG)
    logger.info("Building Corpus from Training set...")
    unique_contexts = set()
    corpus = []

    for example in squad_train:
        ctx = example["context"]
        if ctx not in unique_contexts:
            unique_contexts.add(ctx)
            corpus.append({"id": example["id"], "title": example["title"], "text": ctx})

    logger.info(f"Final Train Subset size: {len(train_subset)}")
    logger.info(f"Final Validation Subset size: {len(val_subset)}")
    logger.info(f"Final Test Subset size: {len(test_subset)}")
    logger.info(f"Corpus size (unique contexts): {len(corpus)}")

    # Save to disk
    logger.info(f"Saving to {output_dir}...")

    train_path = output_dir / "train_subset.json"
    val_path = output_dir / "val_subset.json"
    test_path = output_dir / "test_subset.json"
    corpus_path = output_dir / "corpus.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump([ex for ex in train_subset], f, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump([ex for ex in val_subset], f, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump([ex for ex in test_subset], f, indent=2)

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    logger.info("Done!")

    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "corpus": str(corpus_path),
    }


def main():
    """CLI entry point for data preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare SQuAD dataset subsets for training and evaluation"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data",
        help="Output directory for dataset files (default: data)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=4000,
        help="Number of training samples (default: 4000)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Number of test samples (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    prepare_dataset(
        output_dir=args.output,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
