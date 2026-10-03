import json
import os
from datasets import Dataset, DatasetDict

def format_prompt(example):
    """
    Instruction + Input ko model prompt format mein structure karta hai.
    """
    instruction = example.get("instruction", "")
    user_input = example.get("input", "")
    output = example.get("output", "")

    if user_input:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    return {"text": prompt}

def prepare_dataset(input_file="data/domain_data.jsonl", output_dir="data/processed_dataset"):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} nahi mila!")

    print(f"Loading dataset from {input_file}...")
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_list(data)
    
    # Apply Prompt Formatting
    formatted_dataset = hf_dataset.map(format_prompt)

    # Train / Validation Split (80% Train, 20% Val)
    split_dataset = formatted_dataset.train_test_split(test_size=0.2, seed=42)
    dataset_dict = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    print(f"Dataset successfully saved to '{output_dir}'!")
    print(f"Train samples: {len(dataset_dict['train'])}, Validation samples: {len(dataset_dict['validation'])}")

if __name__ == "__main__":
    prepare_dataset()