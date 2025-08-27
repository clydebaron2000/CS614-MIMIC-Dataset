#!/usr/bin/env python3
import os
import json
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ---------------- Defaults (can be overridden by env or CLI) ----------------
DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER_DIR = "/home/fxp23/llama_ft/outputs/llama3-8b-qlora"  # your trained LoRA
DEFAULT_SUBJECTS_FILE = "/storage/mimic_data/output/mimic_iv_subjects_sample10_with_dx_px.jsonl"
DEFAULT_SYSTEM = "You are a careful, supportive health assistant"
# ---------------- Utilities ----------------
def require_adapter(path: str):
    cfg = os.path.join(path, "adapter_config.json")
    wts = os.path.join(path, "adapter_model.safetensors")
    if not (os.path.isfile(cfg) and os.path.isfile(wts)):
        print(f"[ERROR] Fine-tuned LoRA adapter not found at: {path}")
        print("Expected files:")
        print(f"  - {cfg}")
        print(f"  - {wts}")
        sys.exit(1)

def load_tokenizer(base_model_id: str):
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_model(base_model_id: str, adapter_dir: str | None):
    # 4-bit quant + SDPA attention to keep VRAM low
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",  # uses scaled-dot product attention if available
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model

# ---------------- Subject lookup ----------------
def find_subject_record(path: str, subj_id: str):
    """
    Scans JSONL for a record whose subject id matches.
    Accepts keys like 'subject_id', 'SUBJECT_ID'.
    Returns the parsed JSON object or None if not found.
    """
    want = str(subj_id).strip()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # Try common id fields
            candidates = [
                obj.get("subject_id"),
                obj.get("SUBJECT_ID"),
                obj.get("id"),
                obj.get("subjectId"),
            ]
            # Also check nested 'subject' block
            if obj.get("subject") and isinstance(obj["subject"], dict):
                candidates.append(obj["subject"].get("subject_id"))
                candidates.append(obj["subject"].get("SUBJECT_ID"))

            candidates = [str(x) for x in candidates if x is not None]
            if want in candidates:
                return obj
    return None

def extract_context_text(record: dict, fallback_id: str):
    """
    We avoid rebuilding prompts/med-classes; we just reuse whatever context
    is present in the record. We try, in order:
      - 'patient_context'
      - 'context'
      - a 'messages' style user block that starts with 'Patient context'
      - otherwise a minimal fallback string
    """
    # Direct fields commonly used
    for key in ("patient_context", "Patient context", "context"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Try messages-style (as in many chat datasets)
    msgs = record.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                cont = m.get("content") or ""
                if isinstance(cont, str) and cont.strip():
                    # If it already contains "Patient context", use as-is.
                    if cont.lower().startswith("patient context"):
                        # strip any trailing 'Question...' lines; the QA script will add its own question
                        return cont.split("\n\nQuestion")[0].strip()
                    # Otherwise, still useful as user context
                    return cont.strip()

    # Minimal fallback: don’t derive new structures—keep it simple
    sid = record.get("subject_id") or record.get("SUBJECT_ID") or fallback_id
    return f"Patient context\n- Subject ID: {sid}\n- (No structured context block found in source file.)"

# ---------------- Chat & generation ----------------

def clamp_for_vram(tok, messages, max_input_tokens=700):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    if ids.size(0) > max_input_tokens:
        enc["input_ids"] = ids[-max_input_tokens:].unsqueeze(0)
        if "attention_mask" in enc:
            enc["attention_mask"] = enc["attention_mask"][:, -max_input_tokens:]
    return enc

def answer(model, tok, context_text: str, question: str,
           max_input_tokens=700, max_new_tokens=192,
           temperature=0.2, top_p=0.9, system_prompt: str = DEFAULT_SYSTEM):
    messages = [
        # Keep system minimal; your LoRA has the real behavior.
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context_text}\n\nQuestion\n{question}"},
    ]

    enc = clamp_for_vram(tok, messages, max_input_tokens=max_input_tokens)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,                 # deterministic & lighter on memory
            use_cache=True,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    gen = out[0, input_ids.size(1):]
    return tok.decode(gen, skip_special_tokens=True).strip()

def get_context_for_subject(subjects_file: str, subject_id: str):
    rec = find_subject_record(subjects_file, subject_id)
    if rec is None:
        return None, None
    context_text = extract_context_text(rec, subject_id)
    preview = context_text if context_text and len(context_text) < 800 else (context_text[:800] + " [...]" if context_text else "")
    return context_text, preview

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Ask health questions by subject_id using (base or LoRA) Llama 3.1.")
    parser.add_argument("--subjects-file", default=os.environ.get("SUBJECTS_FILE", DEFAULT_SUBJECTS_FILE))
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", DEFAULT_BASE_MODEL),
                        help="Base model repo or local snapshot path (env: MODEL_ID)")
    parser.add_argument("--adapter-dir", default=os.environ.get("ADAPTER_DIR", DEFAULT_ADAPTER_DIR),
                        help="LoRA adapter directory (env: ADAPTER_DIR). Leave empty for base model.")
    parser.add_argument("--max-input-tokens", type=int, default=700)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--system-prompt", default=os.environ.get("SYSTEM_PROMPT", ""),
                    help="Override the system prompt text. If empty, uses DEFAULT_SYSTEM.")
    parser.add_argument("--system-prompt-file",
                    help="Path to a text file containing the system prompt (overrides --system-prompt).")
    # One-shot mode (non-interactive; useful for batch eval)
    parser.add_argument("--once", action="store_true", help="Run one subject+question and exit")
    parser.add_argument("--subject-id", type=str, help="Subject ID for --once")
    parser.add_argument("--question", type=str, help="Question for --once")
    args = parser.parse_args()

    sys_prompt = DEFAULT_SYSTEM
    if args.system_prompt:
        sys_prompt = args.system_prompt
    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            sys_prompt = f.read().strip()

    print(f"[INFO] Using system prompt ({len(sys_prompt)} chars).")

    # Optional fragmentation help:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True

    # Validate adapter only if provided (base-only runs should work)
    adapter_dir = (args.adapter_dir or "").strip()
    base_model_id = args.model_id

    if args.once:
        if not args.subject_id or not args.question:
            print("--once requires --subject-id and --question", file=sys.stderr)
            sys.exit(2)

    # Load tokenizer/model once
    print(f"[INFO] Loading tokenizer/model (base: {base_model_id})...")
    if adapter_dir:
        print(f"[INFO] Applying LoRA adapter: {adapter_dir}")
        require_adapter(adapter_dir)
    tok = load_tokenizer(base_model_id)
    model = load_model(base_model_id, adapter_dir if adapter_dir else None)
    print("[INFO] Ready.")

    # One-shot branch
    if args.once:
        ctx, _ = get_context_for_subject(args.subjects_file, args.subject_id)
        if ctx is None:
            print(f"[WARN] subject_id '{args.subject_id}' not found in {args.subjects_file}")
            sys.exit(3)
        resp = answer(
            model, tok, ctx, args.question,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        print("ASSISTANT>", resp)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        sys.exit(0)

    # Interactive loop (original behavior)
    while True:
        try:
            subj = input("\nEnter subject_id (or 'q' to quit): ").strip()
            if not subj:
                continue
            if subj.lower() in ("q", "quit", "exit"):
                print("Bye.")
                break

            ctx, preview = get_context_for_subject(args.subjects_file, subj)
            if ctx is None:
                print(f"[WARN] subject_id '{subj}' not found in {args.subjects_file}")
                continue

            print("\n--- Context Preview ---")
            if preview:
                print(preview)
            else:
                print("(no context available)")
            print("-----------------------")

            q = input("\nNow enter the health question: ").strip()
            if not q:
                print("[WARN] Empty question; try again.")
                continue

            print("\n[Generating answer ...]")
            resp = answer(
                model, tok, ctx, q,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                system_prompt=sys_prompt,
            )
            print("\nASSISTANT>", resp)

            # Keep VRAM tidy across turns
            torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("\nBye.")
            break

if __name__ == "__main__":
    main()

