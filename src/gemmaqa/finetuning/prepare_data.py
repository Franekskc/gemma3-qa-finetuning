import json
import os
import random
from datasets import load_dataset, Dataset


def main():
    # Configuration
    output_dir = "data"
    train_size = 4000
    test_size = 1000
    seed = 42

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    print("Loading SQuAD dataset...")
    # Load full training set to split into our Train/Test
    
    squad_train: Dataset = load_dataset("squad", split="train")
    squad_val = load_dataset("squad", split="validation")

    print(f"Original Train size: {len(squad_train)}")
    print(f"Original Val size: {len(squad_val)}")

    # 1. Train Subset (4000 from Train)
    # We shuffle to get a random subset
    train_indices = list(range(len(squad_train)))
    random.shuffle(train_indices)
    train_subset_indices = train_indices[:train_size]
    train_subset: Dataset = squad_train.select(train_subset_indices)
    
    # 2. Test Subset (1000 from Validation)
    val_indices = list(range(len(squad_val)))
    random.shuffle(val_indices)
    test_subset_indices = val_indices[:test_size]
    test_subset = squad_val.select(test_subset_indices)

    # 3. Corpus (Knowledge Base)
    # extract unique contexts.
    print("Building Corpus from Training set...")
    unique_contexts = set()
    corpus = []
    
    # iterate over the whole training set
    for example in squad_train:
        ctx = example['context']
        if ctx not in unique_contexts:
            unique_contexts.add(ctx)
            corpus.append({
                "id": example['id'],
                "title": example['title'],
                "text": ctx
            })
            
    print(f"Train Subset size: {len(train_subset)}")
    print(f"Test Subset size: {len(test_subset)}")
    print(f"Corpus size (unique contexts): {len(corpus)}")

    # Save to disk
    print(f"Saving to {output_dir}...")
    
    # Save datasets as JSONL for easy loading or just use HF Dataset save_to_disk
    # JSONL is more portable for inspection
    
    with open(os.path.join(output_dir, "train_subset.json"), "w", encoding="utf-8") as f:
        json.dump([ex for ex in train_subset], f, indent=2)
        
    with open(os.path.join(output_dir, "test_subset.json"), "w", encoding="utf-8") as f:
        json.dump([ex for ex in test_subset], f, indent=2)
        
    with open(os.path.join(output_dir, "corpus.json"), "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
