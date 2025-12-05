import torch
import random
from datasets import load_dataset
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

    # Load SQuAD validation set
    print("Loading SQuAD validation set...")
    dataset = load_dataset("squad", split="validation")
    
    # Select 5 random examples
    indices = random.sample(range(len(dataset)), 5)
    examples = dataset.select(indices)

    print("\nStarting evaluation on 5 random examples...\n")
    print("-" * 50)

    for example in examples:
        context = example['context']
        question = example['question']
        ground_truth_answers = example['answers']['text']

        # Format prompt using chat template
        messages = [
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda")

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<end_of_turn>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.1, # Lower temperature for more deterministic answers
            top_p=0.9,
            eos_token_id=terminators
        )

        response = outputs[0][input_ids.shape[-1]:]
        model_answer = tokenizer.decode(response, skip_special_tokens=True).strip()

        print(f"Question: {question}")
        print(f"Context (truncated): {context[:100]}...")
        print(f"Ground Truth: {ground_truth_answers}")
        print(f"Model Answer: {model_answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()
