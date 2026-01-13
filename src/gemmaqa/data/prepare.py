"""
Dataset preparation script.
Downloads SQuAD dataset and creates train/test subsets and corpus.
"""

import json
import random
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

from gemmaqa.utils import get_logger

logger = get_logger(__name__)


def prepare_dataset(
    output_dir: str | Path = "data",
    train_size: int = 4000,
    val_size: int = 500,
    test_size: int = 1000,
    seed: int = 42,
    mix_duorc: bool = False,
) -> dict:
    """
    Prepare dataset subsets for training and evaluation.
    Can optionally mix SQuAD with DuoRC for Data Augmentation.

    Args:
        output_dir: Directory to save output files.
        train_size: Number of training samples to select.
        val_size: Number of validation samples to select.
        test_size: Number of test samples to select.
        seed: Random seed for reproducibility.
        mix_duorc: If True, mixes DuoRC dataset into training data.

    Returns:
        Dict with paths to created files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    # 1. Load SQuAD (Base)
    logger.info("Loading SQuAD dataset...")
    squad_train: Dataset = load_dataset("squad", split="train")
    squad_val = load_dataset("squad", split="validation")

    logger.info(f"Original SQuAD Train size: {len(squad_train)}")
    logger.info(f"Original SQuAD Val size: {len(squad_val)}")

    # 2. Data Augmentation (Optional)
    raw_train_dataset = squad_train

    if mix_duorc:
        logger.info("Mixing DuoRC into Training Data...")
        try:
            duorc = load_dataset("duorc", "ParaphraseRC", split="train")
            logger.info(f"Original DuoRC Train size: {len(duorc)}")

            # --- Normalization DuoRC into SQuAD format ---
            duorc = duorc.rename_column("plot", "context")
            duorc = duorc.rename_column("question_id", "id")

            cols_to_keep = ["context", "question", "answers", "id"]
            squad_train = squad_train.select_columns(cols_to_keep)
            duorc = duorc.select_columns(cols_to_keep)

            # fix response format (List -> Dict)
            # DuoRC response: ["odp1", "odp2"]
            # SQuAD response: {'text': ["odp1"], 'answer_start': [123]}
            def fix_duorc_structure(example):
                return {"answers": {"text": example["answers"], "answer_start": []}}

            duorc = duorc.map(fix_duorc_structure, desc="Formatting DuoRC answers")
            duorc = duorc.cast(squad_train.features)

            # Connecting
            raw_train_dataset = concatenate_datasets([squad_train, duorc])
            logger.info(f"Combined Size: {len(raw_train_dataset)} (SQuAD + DuoRC)")

        except Exception as e:
            logger.error(f"Failed to load DuoRC: {e}")
            logger.info("Falling back to pure SQuAD.")

    # 3. Shuffle & Slice Training Data
    full_indices = list(range(len(raw_train_dataset)))
    random.shuffle(full_indices)

    # 4. Prepare Train and Validation SubSets
    total_needed = train_size + val_size
    real_total = min(total_needed, len(full_indices))
    mixed_pool = raw_train_dataset.select(full_indices[:real_total])
    split_point = min(train_size, len(mixed_pool))

    train_subset = mixed_pool.select(range(0, split_point))
    val_subset = mixed_pool.select(range(split_point, len(mixed_pool)))


    # 5. Prepare Test Subset (PURE SQuAD)
    logger.info("Preparing Test set (Pure SQuAD)...")
    val_indices = list(range(len(squad_val)))
    random.shuffle(val_indices)
    test_subset = squad_val.select(val_indices[:test_size])

    logger.info(f"Final Train Subset size: {len(train_subset)}")
    logger.info(f"Final Validation Subset size: {len(val_subset)}")
    logger.info(f"Final Test Subset size: {len(test_subset)}")

    # Save to disk
    logger.info(f"Saving to {output_dir}...")

    train_path = output_dir / "train_subset.json"
    val_path = output_dir / "val_subset.json"
    test_path = output_dir / "test_subset.json"

    def save_json(data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([ex for ex in data], f, indent=2)

    save_json(train_subset, train_path)
    save_json(val_subset, val_path)
    save_json(test_subset, test_path)

    logger.info("Done!")

    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
    }
