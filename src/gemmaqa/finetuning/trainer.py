"""
Training orchestration for all finetuning modes.
"""

import argparse
from pathlib import Path

from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from gemmaqa.config import QAConfig
from gemmaqa.config.settings import DEFAULT_CONFIG_PATH
from gemmaqa.data import load_train_and_eval_data
from gemmaqa.finetuning.freeze import get_freeze_model
from gemmaqa.finetuning.full import get_full_model
from gemmaqa.finetuning.lora import get_lora_model
from gemmaqa.utils import configure_logging, get_logger, set_seed

logger = get_logger(__name__)


def get_model_and_tokenizer(cfg: QAConfig):
    """
    Route to the correct model loader based on mode.

    Args:
        cfg: QAConfig with mode and model settings.

    Returns:
        Tuple of (model, tokenizer)
    """
    if cfg.mode == "lora":
        return get_lora_model(cfg)
    elif cfg.mode == "full":
        return get_full_model(cfg)
    elif cfg.mode == "freeze":
        return get_freeze_model(cfg)
    else:
        raise ValueError(
            f"Unknown mode: {cfg.mode}. Must be one of: full, lora, freeze"
        )


def build_training_args(cfg: QAConfig) -> TrainingArguments:
    """
    Build TrainingArguments from config.

    Args:
        cfg: QAConfig with training settings.

    Returns:
        TrainingArguments instance.
    """
    grad_accum = (
        cfg.training.effective_batch_size // cfg.training.per_device_train_batch_size
    )

    return TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_steps=cfg.training.logging_steps,
        save_strategy="epoch",
        save_total_limit=cfg.training.save_total_limit,
        bf16=cfg.training.bf16,
        report_to="none",
        optim="stable_adamw",
        # Evaluation settings
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )


def run_training(
    cfg: QAConfig,
    train_data_path: str = "data/train_subset.json",
    val_data_path: str = "data/val_subset.json",
    max_steps: int | None = None,
):
    """
    Run the training pipeline.

    Args:
        cfg: QAConfig with all settings.
        train_data_path: Optional path to training data JSON.
        val_data_path: Optional path to validation data JSON.
        max_steps: Optional max steps override (for testing).
    """
    # Set seed for reproducibility
    set_seed(cfg.seed)

    # Load model and tokenizer
    logger.info("Loading model", mode=cfg.mode)
    model, tokenizer = get_model_and_tokenizer(cfg)

    tokenizer.padding_side = "right"

    # Load and process data (train + eval)
    logger.info("Loading data")
    datasets = load_train_and_eval_data(
        tokenizer=tokenizer,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        train_samples=cfg.data.max_train_samples,
        val_samples=cfg.data.val_samples,
        max_length=cfg.data.max_seq_len,
    )
    train_dataset = datasets["train"]
    eval_dataset = datasets["eval"]

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    # Training arguments
    training_args = build_training_args(cfg)

    # Override max_steps if specified (for testing)
    if max_steps is not None:
        training_args.max_steps = max_steps

    logger.info(
        "Training args",
        output_dir=training_args.output_dir,
        lr=training_args.learning_rate,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training", mode=cfg.mode)
    trainer.train()

    # Save model
    output_path = Path(cfg.training.output_dir) / "final"
    logger.info("Saving model", path=str(output_path))

    if cfg.mode == "lora":
        # Save only the adapter for LoRA
        model.save_pretrained(str(output_path))
    else:
        # Save full model for full/freeze modes
        trainer.save_model(str(output_path))

    logger.info("Training complete!")


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train Gemma QA model")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["full", "lora", "freeze"],
        help="Training mode: full (full finetuning), lora (LoRA adapters), freeze (layer freezing)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data JSON (default: data/train_subset.json)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max training steps (for testing, overrides epochs)",
    )
    args = parser.parse_args()

    # Setup logging
    configure_logging()

    # Load config
    logger.info("Loading config", path=args.config, mode=args.mode)
    cfg = QAConfig.load(args.config, args.mode)

    # Run training
    run_training(cfg, train_data_path=args.data, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
