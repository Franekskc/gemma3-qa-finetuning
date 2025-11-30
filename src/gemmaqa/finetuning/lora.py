import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def get_lora_model(model_name="google/gemma-3-1b-it"):
    """
    Loads the base model and applies LoRA configuration with 4-bit quantization.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"] 
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()
    
    return model, tokenizer
