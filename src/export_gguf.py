import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def export_merged_model():
    base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "models/qlora_adapter"
    merged_output_dir = "models/merged_model"

    print("Base model aur adapter load ho raha hai...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    # Load adapter onto base model
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("LoRA Adapter ko base model ke sath merge kiya ja raha hai...")
    merged_model = model.merge_and_unload()

    print(f"Merged model ko '{merged_output_dir}' mein save kar rahe hain...")
    os.makedirs(merged_output_dir, exist_ok=True)
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
    
    print("Merged FP32 model ready hai! Ab isse `llama.cpp` se GGUF/Quantized format mein convert kiya ja sakta hai.")

if __name__ == "__main__":
    export_merged_model()