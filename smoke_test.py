import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "meta-llama/Meta-Llama-3.1-8B-Instruct"   # same as MODEL_ID you trained from
ADAPTER = os.environ.get("ADAPTER_DIR", "/home/fxp23/llama_ft/outputs/llama3-8b-qlora")            # your OUTPUT_DIR from training

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Attach the LoRA adapter
from peft import PeftModel
model = PeftModel.from_pretrained(base, ADAPTER)

# Build a test chat
messages = [
    {"role":"system","content":"You are a careful, supportive health assistant."},
    {"role":"user","content":"Patient context\n- Age: 48\n- Sex: Female\n\nQuestion\nI keep coughing at night. Any safe steps to feel better?"}
]

prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok([prompt], return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,         # greedy for deterministic smoke test
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.05
    )

print(tok.decode(out[0], skip_special_tokens=True))
