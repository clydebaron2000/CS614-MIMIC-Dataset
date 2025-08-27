#!/usr/bin/env python3
import os, json, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER = "/home/fxp23/llama_ft/outputs/llama3-8b-qlora"  # adjust if different
DEVICE = 0  # GPU id

def load_model():
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE},
        attn_implementation="sdpa",  # uses scaled dot-product attention if available
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    return model

def load_tok():
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def clamp_tokens(tok, messages, max_input_tokens=700):
    # Build chat text, then truncate from the left (oldest context) if needed
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if ids.size(0) > max_input_tokens:
        ids = ids[-max_input_tokens:]
    return ids.unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-input-tokens", type=int, default=700)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    # Optional: slightly reduce fragmentation
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Loading tokenizer/model (4-bit + LoRA adapter)...")
    tok = load_tok()
    model = load_model()

    print("Ready. Type your question. Ctrl+C to exit.")
    while True:
        try:
            user = input("\nUSER> ").strip()
            if not user:
                continue

            messages = [
                {"role":"system","content":"You are a careful, supportive health assistant."},
                {"role":"user","content":user},
            ]
            input_ids = clamp_tokens(tok, messages, max_input_tokens=args.max_input_tokens).to(model.device)

            with torch.inference_mode():
                out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=False,           # deterministic; lower memory variance
                    use_cache=True,            # KV cache is efficient for short outputs
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            # Decode only the newly generated part
            gen = out[0, input_ids.size(1):]
            print("\nASSISTANT>", tok.decode(gen, skip_special_tokens=True).strip())

            torch.cuda.empty_cache()  # keep VRAM tidy between turns
        except KeyboardInterrupt:
            print("\nBye!")
            break

if __name__ == "__main__":
    # Strongly recommended before launching:
    # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    main()
