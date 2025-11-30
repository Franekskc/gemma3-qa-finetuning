import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def main():
    # Model and adapter paths
    base_model_name = "google/gemma-3-1b-it"
    adapter_path = "./gemma-lora-squad-final"

    print(f"Loading base model: {base_model_name}")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    # 1. Define the message structure
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]

    # 2. Apply the template (Adds <start_of_turn>, <end_of_turn>, etc. automatically)
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")

    print(f"\nPrompt (Formatted): {tokenizer.decode(input_ids[0])}")
    print("\nGenerating...")

    # 3. Add explicit EOS token handling
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=terminators  # Stop exactly when the model finishes the turn
    )

    # 4. Decode ONLY the new response (remove the input prompt)
    response = outputs[0][input_ids.shape[-1]:]
    print(f"\nResponse:\n{tokenizer.decode(response, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()