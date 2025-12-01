import argparse
import logging
import os
import sys
from functools import partial

from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from gemmaqa.config import QAConfig
from gemmaqa.data import load_squad, prepare_train_features, prepare_validation_features
from gemmaqa.eval import compute_metrics_fn
from gemmaqa.modeling import prepare_model
from gemmaqa.utils import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Activate QA training in chosen mode.")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["full", "freeze", "lora"],
        help="Training mode: full (full fine-tuning), freeze (layer-freezing), lora (LoRA).",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=r"src\gemmaqa\config.yaml",
        help="Config file path.",
    )

    return parser.parse_args()


def train(cfg: QAConfig):
    """Train the QA model.

    mode:
      - `full`: full fine-tuning (all params trainable)
      - `freeze`: TODO - layer freezing (placeholder)
      - `lora`: TODO - LoRA (placeholder)
    """
    set_seed(42)

    t_cfg = cfg.training
    d_cfg = cfg.data

    os.makedirs(t_cfg.output_dir, exist_ok=True)

    model, tokenizer = prepare_model(cfg.model_name, mode=cfg.mode)
    logger.info("Loaded model %s", cfg.model_name)

    if t_cfg.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing on model.")
        except Exception as e:
            logger.warning("Could not enable gradient checkpointing: %s", e)

    ds = load_squad(d_cfg.max_train_samples, d_cfg.val_samples)

    logger.info("Tokenizing training data...")
    train_features = ds["train"].map(
        partial(
            prepare_train_features,
            tokenizer=tokenizer,
            max_length=d_cfg.max_seq_len,
            doc_stride=d_cfg.doc_stride,
        ),
        batched=True,
        remove_columns=ds["train"].column_names,
    )
    val_features = ds["validation"].map(
        partial(
            prepare_validation_features,
            tokenizer=tokenizer,
            max_length=d_cfg.max_seq_len,
            doc_stride=d_cfg.doc_stride,
        ),
        batched=True,
        remove_columns=ds["validation"].column_names,
    )

    compute_metrics = partial(
        compute_metrics_fn, examples=ds["validation"], features=val_features
    )

    if t_cfg.effective_batch_size > 0:
        calc_accumulation_steps = (
            t_cfg.effective_batch_size // t_cfg.per_device_train_batch_size
        )
        gradient_accumulation_steps = max(1, calc_accumulation_steps)
        logger.info(
            f"Effective batch size set to {t_cfg.effective_batch_size}. "
            f"Calculated gradient accumulation steps: {gradient_accumulation_steps} "
            f"(per_device={t_cfg.per_device_train_batch_size})"
        )

    train_batch_size = t_cfg.per_device_train_batch_size

    total_train_steps = int(
        len(train_features)
        / train_batch_size
        / gradient_accumulation_steps
        * t_cfg.num_train_epochs
    )
    logger.info("Total training steps (approx): %d", total_train_steps)

    training_args = TrainingArguments(
        output_dir=t_cfg.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=t_cfg.per_device_train_batch_size,
        optim="stable_adamw",
        gradient_checkpointing=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=t_cfg.num_train_epochs,
        learning_rate=t_cfg.learning_rate,
        weight_decay=t_cfg.weight_decay,
        fp16=True,
        logging_steps=t_cfg.logging_steps,
        save_total_limit=t_cfg.save_total_limit,
        remove_unused_columns=True,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        warmup_ratio=t_cfg.warmup_ratio,
        lr_scheduler_type="linear",
    )

    # Early stopping callback (monitor eval_F1 via compute_metrics_fn)
    early_stop_cb = EarlyStoppingCallback(
        early_stopping_patience=t_cfg.early_stopping_patience
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

    trainer.save_model(t_cfg.output_dir)

    test_features = ds["test"].map(
        partial(
            prepare_validation_features,
            tokenizer=tokenizer,
            max_length=d_cfg.max_seq_len,
            doc_stride=d_cfg.doc_stride,
        ),
        batched=True,
        remove_columns=ds["test"].column_names,
    )
    trainer.eval_dataset = test_features
    trainer.compute_metrics = partial(
        compute_metrics_fn, examples=ds["test"], features=test_features
    )
    metrics = trainer.evaluate()

    return metrics


if __name__ == "__main__":
    args = parse_arguments()

    # Check if file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found {args.config}")
        exit(1)

    print(f"--- Starting training in: {args.mode.upper()} mode ---")

    # load configuration
    config = QAConfig.load(
        yaml_path=args.config,
        selected_mode=args.mode,
    )

    metrics = train(config)

    print("Metrics:", metrics)
