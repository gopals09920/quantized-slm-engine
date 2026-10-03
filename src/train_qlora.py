import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def train():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_path = "data/processed_dataset"
    output_dir = "models/qlora_adapter"

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    print(f"Loading base model on CPU: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA Config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # SFTConfig explicitly set for CPU
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=10,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        dataset_text_field="text",
        use_cpu=True,
        fp16=False,
        bf16=False
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    print("Starting Fast CPU Fine-Tuning...")
    trainer.train()

    print(f"Saving LoRA Adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning complete!")

if __name__ == "__main__":
    train()