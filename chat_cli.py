#!/usr/bin/env python3
import os, sys, torch, readline  # readline = nicer CLI history
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = os.environ.get("MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
ADAPTER = os.environ.get("ADAPTER_DIR", "/home/fxp23/llama_ft/outputs/llama3-8b-qlora")

SYSTEM_MSG = (
    "You are a careful, supportive health assistant. You are not a substitute for a clinician."
)

# ---- load tokenizer/model + adapter (4-bit) ----
assert os.path.isdir(ADAPTER), f"Adapter dir not found: {ADAPTER}"
assert os.path.isfile(os.path.join(ADAPTER, "adapter_config.json")), "adapter_config.json missing"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb_cfg, device_map="auto", torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

# ---- chat state ----
messages = [{"role": "system", "content": SYSTEM_MSG}]

def generate_reply(messages, max_new_tokens=256, temperature=0.2, top_p=0.9):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    # Strip the preceding turns to isolate the last assistant message
    # Simple heuristic: everything after the last "<assistant>" if present
    # (Llama 3.1 template uses special tokens; skip_special_tokens handles most)
    return text.split("Assistant:")[-1].strip() if "Assistant:" in text else text.strip()

def main():
    print("Loaded. Type your message. Commands: /reset  /exit  (Ctrl+C also exits)")
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user:
            continue
        if user == "/exit":
            print("Bye!")
            break
        if user == "/reset":
            del messages[1:]
            print("Context cleared.")
            continue

        messages.append({"role": "user", "content": user})
        reply = generate_reply(messages)
        messages.append({"role": "assistant", "content": reply})
        print("\nAssistant:", reply)

if __name__ == "__main__":
    main()
