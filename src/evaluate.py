import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "models/qlora_adapter"

    print("Tokenizer load ho raha hai...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Base Model load ho raha hai...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    # Test Prompt
    test_instruction = "Audit this Python code for vulnerabilities."
    test_input = "import os\nos.system('ping ' + user_input)"
    prompt = f"### Instruction:\n{test_instruction}\n\n### Input:\n{test_input}\n\n### Response:\n"

    print("\n--- BASE MODEL RESPONSE ---")
    base_output = generate_response(base_model, tokenizer, prompt)
    print(base_output)

    print("\nFine-tuned LoRA Adapter load ho raha hai...")
    fine_tuned_model = PeftModel.from_pretrained(base_model, adapter_path)

    print("\n--- FINE-TUNED MODEL RESPONSE ---")
    fine_tuned_output = generate_response(fine_tuned_model, tokenizer, prompt)
    print(fine_tuned_output)

if __name__ == "__main__":
    main()