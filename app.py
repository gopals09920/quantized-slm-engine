import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

st.set_page_config(page_title="Quantized SLM Domain Engine", layout="wide")

st.title("🛡️ Quantized SLM Domain Fine-Tuning Engine")
st.caption("Comparing Base SLM vs QLoRA Fine-Tuned Model Performance")

@st.cache_resource
def load_models():
    base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "models/qlora_adapter"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    fine_tuned_model = PeftModel.from_pretrained(base_model, adapter_path)
    return base_model, fine_tuned_model, tokenizer

st.info("Loading models into memory... Please wait.")
base_model, fine_tuned_model, tokenizer = load_models()
st.success("Models loaded successfully!")

# Input Section
instruction = st.text_input("Instruction", "Audit this Python code for vulnerabilities.")
user_code = st.text_area("Python Code Input", "import os\nos.system('ping ' + user_input)")

if st.button("Run Audit & Compare"):
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{user_code}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 Base SLM (Un-tuned)")
        with st.spinner("Generating Base output..."):
            with torch.no_grad():
                outputs_base = base_model.generate(
                    **inputs, max_new_tokens=100, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id
                )
            base_res = tokenizer.decode(outputs_base[0], skip_special_tokens=True)
            st.code(base_res)

    with col2:
        st.subheader("🟢 Fine-Tuned SLM (Domain Engine)")
        with st.spinner("Generating Fine-tuned output..."):
            with torch.no_grad():
                outputs_ft = fine_tuned_model.generate(
                    **inputs, max_new_tokens=100, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id
                )
            ft_res = tokenizer.decode(outputs_ft[0], skip_special_tokens=True)
            st.code(ft_res)