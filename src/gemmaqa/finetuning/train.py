import transformers
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from .lora import get_lora_model
from .data import load_and_process_data

def main():
    model_name = "google/gemma-3-1b-it"
    
    # Load model and tokenizer
    print(f"Loading model {model_name}...")
    model, tokenizer = get_lora_model(model_name)
    
    # Load and process data
    print("Loading data...")
    # Use the prepared training subset
    tokenized_datasets = load_and_process_data(tokenizer, data_path="data/train_subset.json")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./gemma-lora-squad",
        per_device_train_batch_size=1, # Small batch size for 4GB GPU
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=50, # Short run for verification
        save_strategy="no",
        fp16=True, # Use mixed precision
        report_to="none"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    model.save_pretrained("./gemma-lora-squad-final")
    print("Done!")

if __name__ == "__main__":
    main()
