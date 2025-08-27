#!/usr/bin/env python3
import os, json, torch, sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ---------------- CONFIG ----------------
BASE_MODEL  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_DIR = "/home/fxp23/llama_ft/outputs/llama3-8b-qlora"
DATA_FILE   = "/storage/mimic_data/output/mimic_iv_subjects_sample10_with_dx_px.jsonl"

SYSTEM_MSG = (
    "You are a careful, supportive health assistant. You are not a substitute for a clinician. "
    "Explain in plain language, be concise and empathetic, and highlight red-flags when appropriate."
)

# ---------------- CHECK ADAPTER ----------------
if not (os.path.isdir(ADAPTER_DIR) and os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_config.json"))):
    sys.exit(f"❌ Fine-tuned adapter not found at {ADAPTER_DIR}. Aborting.")

# ---------------- LOAD DATA ----------------
def load_subject_index(path):
    idx = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                sid = str(obj.get("subject_id"))
                if sid:
                    idx[sid] = obj
            except Exception:
                continue
    return idx

subject_index = load_subject_index(DATA_FILE)
if not subject_index:
    sys.exit(f"❌ No subjects found in {DATA_FILE}")

# ---------------- LOAD MODEL ----------------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model + adapter...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_cfg,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()
print("✅ Fine-tuned model loaded.\n")

# ---------------- GENERATE ----------------
def generate(messages, max_new_tokens=256):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split("Assistant:")[-1].strip() if "Assistant:" in text else text.strip()

# ---------------- MAIN LOOP ----------------
while True:
    sid = input("Enter subject_id (or 'exit'): ").strip()
    if sid.lower() == "exit":
        print("Bye!")
        break
    subj = subject_index.get(sid)
    if not subj:
        print(f"❌ Subject {sid} not found.\n")
        continue

    question = input("Enter your health question: ").strip()
    if not question:
        continue

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": f"Patient record:\n{subj}\n\nQuestion:\n{question}"},
    ]

    print("\nThinking...\n")
    answer = generate(messages)
    print("Assistant:\n" + answer + "\n")
