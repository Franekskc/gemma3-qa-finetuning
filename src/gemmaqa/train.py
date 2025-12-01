import logging
import os
import sys
from functools import partial
from typing import Literal

from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from gemmaqa.data import load_squad, prepare_train_features, prepare_validation_features
from gemmaqa.eval import compute_metrics_fn
from gemmaqa.modeling import prepare_model
from gemmaqa.utils import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def train(
    model_name_or_path: str,
    mode: Literal["full", "freeze", "lora"],
    learning_rate: float = 0.0,
    output_dir: str = "outputs",
    max_train_samples: int = 1000,
    per_device_train_batch_size: int = 2,
    effective_batch_size: int = 8,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 5,
    early_stopping_patience: int = 1,
):
    """Train the QA model.

    mode:
      - `full`: full fine-tuning (all params trainable)
      - `freeze`: TODO - layer freezing (placeholder)
      - `lora`: TODO - LoRA (placeholder)
    """
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = prepare_model(model_name_or_path, mode=mode)
    logger.info("Loaded model %s", model_name_or_path)

    # enable gradient checkpointing if available
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing on model.")
        except Exception as e:
            logger.warning("Could not enable gradient checkpointing: %s", e)

    # Learning rate defaults based on mode
    if learning_rate == 0.0:
        learning_rate = 1e-4 if mode.startswith("lora") else 2e-5

    ds = load_squad(max_train_samples, 500)

    logger.info("Tokenizing training data...")
    train_features = ds["train"].map(
        partial(prepare_train_features, tokenizer=tokenizer),
        batched=True,
        remove_columns=ds["train"].column_names,
    )
    val_features = ds["validation"].map(
        partial(prepare_validation_features, tokenizer=tokenizer),
        batched=True,
        remove_columns=ds["validation"].column_names,
    )

    compute_metrics = partial(
        compute_metrics_fn, examples=ds["validation"], features=val_features
    )

    if effective_batch_size > 0:
        calc_accumulation_steps = effective_batch_size // per_device_train_batch_size
        gradient_accumulation_steps = max(1, calc_accumulation_steps)
        logger.info(
            f"Effective batch size set to {effective_batch_size}. "
            f"Calculated gradient accumulation steps: {gradient_accumulation_steps} "
            f"(per_device={per_device_train_batch_size})"
        )

    train_batch_size = per_device_train_batch_size
    
    total_train_steps = int(
        len(train_features)
        / train_batch_size
        / gradient_accumulation_steps
        * num_train_epochs
    )
    logger.info("Total training steps (approx): %d", total_train_steps)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=per_device_train_batch_size,
        optim="stable_adamw",
        gradient_checkpointing=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=True,
        logging_steps=20,
        save_total_limit=1,
        remove_unused_columns=True,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",
    )

    # Early stopping callback (monitor eval_F1 via compute_metrics_fn)
    early_stop_cb = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience
    )

    def qa_data_collator(features):
        features_for_model = []
        for feature in features:
            clean_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }

            if "start_positions" in feature and "end_positions" in feature:
                clean_feature["start_positions"] = feature["start_positions"]
                clean_feature["end_positions"] = feature["end_positions"]
            else:
                clean_feature["start_positions"] = 0
                clean_feature["end_positions"] = 0

            features_for_model.append(clean_feature)

        return default_data_collator(features_for_model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=val_features,
        data_collator=qa_data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stop_cb],
    )

    trainer.train()
    logger.info("Training finished. Running evaluation (predict + postprocess)...")

    trainer.save_model(output_dir)

    test_features = ds["test"].map(
        partial(prepare_validation_features, tokenizer=tokenizer),
        batched=True,
        remove_columns=ds["test"].column_names,
    )
    trainer.eval_dataset = test_features
    trainer.compute_metrics = partial(
        compute_metrics_fn, examples=ds["test"], features=test_features
    )
    metrics = trainer.evaluate()
    logger.info(f"Official Test Metrics: {metrics}")

    return metrics


if __name__ == "__main__":
    metrics = train(
        "distilbert-base-uncased",
        output_dir="out_smoke",
        mode="full",
        max_train_samples=10000,
        num_train_epochs=3,
        learning_rate=2e-5,
    )
    print("Smoke metrics:", metrics)
