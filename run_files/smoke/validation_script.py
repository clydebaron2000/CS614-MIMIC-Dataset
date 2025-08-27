#!/usr/bin/env python3
import os, sys, json, argparse, time, re, csv, tarfile, datetime, unicodedata
from typing import Optional, Tuple, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from peft import PeftModel

# ---------- utils ----------
def norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0"," ").replace("\u200B","")
    s = re.sub(r"[ \t]+"," ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def clean_csv(s: str) -> str:
    s = norm(s).replace("\r"," ").replace("\n"," ").strip()
    if s[:1] in ("=","+","-","@"):
        s = "'" + s
    return s

def words_len(s: str) -> int:
    return len(re.findall(r"\S+", s))

def jaccard(a: str, b: str) -> float:
    A = set(re.findall(r"[a-z0-9]+", a.lower()))
    B = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    return len(A & B) / len(A | B)

# ---------- subject/context ----------
def find_subject_record(path: str, subj_id: str):
    sid = str(subj_id).strip()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try: obj = json.loads(line)
            except Exception: continue
            cands = []
            for k in ("subject_id","SUBJECT_ID","id","subjectId"):
                v = obj.get(k); 
                if v is not None: cands.append(str(v))
            if isinstance(obj.get("subject"), dict):
                for k in ("subject_id","SUBJECT_ID"):
                    v = obj["subject"].get(k)
                    if v is not None: cands.append(str(v))
            if sid in cands:
                return obj
    return None

def extract_context_text(record: dict, fallback_id: str):
    for key in ("patient_context","Patient context","context"):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    msgs = record.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                cont = m.get("content") or ""
                if isinstance(cont, str) and cont.strip():
                    return cont.split("\n\nQuestion")[0].strip()
    return f"Patient context\n- Subject ID: {fallback_id}\n- (No structured context block found in source file.)"

# ---------- base/qlora loaders ----------
def load_tokenizer(base_id: str):
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok

def load_base(base_id: str):
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_id, quantization_config=bnb, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="sdpa", local_files_only=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="sdpa", local_files_only=True
        )
    model.eval()
    return model

# ---------- generation ----------
def build_user(tok, context_text: str, question: str, max_input_tokens: int):
    # Keep a cushion for system + formatting
    reserve = 256
    max_ctx = max(64, max_input_tokens - reserve)
    ids = tok(context_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if ids.size(0) > max_ctx:
        context_text = tok.decode(ids[-max_ctx:], skip_special_tokens=True)
    return f"""{context_text}

Question
{question}"""

def gen_answer(model, tok, sys_prompt: str, context_text: str, question: str,
               max_input_tokens=896, max_new_tokens=320) -> Tuple[str,int]:
    user = build_user(tok, context_text, question, max_input_tokens)
    messages = [
        {"role": "system", "content": sys_prompt.strip() if sys_prompt else
         "You are a careful, supportive health assistant."},
        {"role": "user", "content": user},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    if ids.size(0) > max_input_tokens:
        enc["input_ids"] = ids[-max_input_tokens:].unsqueeze(0)
        if "attention_mask" in enc:
            enc["attention_mask"] = enc["attention_mask"][:, -max_input_tokens:]
    enc = {k: v.to(model.device) for k, v in enc.items()}

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    dt = int((time.time()-t0)*1000)
    gen = out[0, enc["input_ids"].shape[1]:]
    txt = tok.decode(gen, skip_special_tokens=True).strip()
    return txt, dt

# ---------- lightweight relevance (embeds if available; else Jaccard) ----------
class Embedder:
    def __init__(self):
        self.tok = None
        self.mdl = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Try to load MiniLM locally; fall back to None
        try:
            rid = "sentence-transformers/all-MiniLM-L6-v2"
            self.tok = AutoTokenizer.from_pretrained(rid, local_files_only=True)
            self.mdl = AutoModel.from_pretrained(rid, local_files_only=True).to(self.device).eval()
        except Exception:
            self.tok = None
            self.mdl = None

    def encode(self, texts: List[str]) -> Optional[torch.Tensor]:
        if self.mdl is None or self.tok is None:
            return None
        with torch.inference_mode():
            enc = self.tok(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            enc = {k: v.to(self.device) for k,v in enc.items()}
            out = self.mdl(**enc)  # [last_hidden_state]
            # mean pool
            attn = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
            masked = out.last_hidden_state * attn
            emb = masked.sum(dim=1) / attn.sum(dim=1).clamp(min=1e-6)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb.detach().cpu()

def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b.T).squeeze().clamp(-1,1).item())

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Validate one subject/question; save artifacts; print scp command")
    ap.add_argument("--subjects-file", required=True)
    ap.add_argument("--subject-id", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--system-prompt-file", default=None)
    ap.add_argument("--model-id", default=os.environ.get("MODEL_ID","meta-llama/Meta-Llama-3.1-8B-Instruct"))
    ap.add_argument("--adapter-dir", default="", help="If set, apply QLoRA adapter")
    ap.add_argument("--max-input-tokens", type=int, default=896)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--out-root", default=f"/tmp/{os.environ.get('USER','user')}")
    ap.add_argument("--ideal-file", default=None, help="Optional JSONL mapping {question, ideal_answer}")
    args = ap.parse_args()

    # Out dir
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, f"validation_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # System prompt
    sys_prompt = ""
    if args.system_prompt_file and os.path.isfile(args.system_prompt_file):
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            sys_prompt = f.read()

    # Subject context
    rec = find_subject_record(args.subjects_file, args.subject_id)
    if rec is None:
        print(f"[ERROR] subject_id {args.subject_id} not found in {args.subjects_file}")
        sys.exit(2)
    ctx = extract_context_text(rec, args.subject_id)

    # Tokenizer & base
    tok = load_tokenizer(args.model_id)

    # BASE
    base = load_base(args.model_id)
    base_ans, base_ms = gen_answer(base, tok, sys_prompt, ctx, args.question,
                                   args.max_input_tokens, args.max_new_tokens)
    base_ans = norm(base_ans)
    base_len = words_len(base_ans)
    del base; torch.cuda.empty_cache()

    # QLORA
    qlora = load_base(args.model_id)
    if args.adapter_dir:
        cfg = os.path.join(args.adapter_dir, "adapter_config.json")
        wts = os.path.join(args.adapter_dir, "adapter_model.safetensors")
        if not (os.path.isfile(cfg) and os.path.isfile(wts)):
            print(f"[ERROR] adapter files not found in {args.adapter_dir}")
            sys.exit(3)
        qlora = PeftModel.from_pretrained(qlora, args.adapter_dir)
    qlora.eval()
    q_ans, q_ms = gen_answer(qlora, tok, sys_prompt, ctx, args.question,
                             args.max_input_tokens, args.max_new_tokens)
    q_ans = norm(q_ans)
    q_len = words_len(q_ans)

    # Relevance metrics
    emb = Embedder()
    if emb.mdl is not None:
        vq = emb.encode([args.question])[0]
        vb = emb.encode([base_ans])[0]
        vqlo = emb.encode([q_ans])[0]
        rel_q_base = cos(vb, vq)
        rel_q_qlora = cos(vqlo, vq)
    else:
        rel_q_base = jaccard(base_ans, args.question)
        rel_q_qlora = jaccard(q_ans, args.question)

    # Ideal answer (optional)
    ideal_txt = ""
    if args.ideal_file and os.path.isfile(args.ideal_file):
        with open(args.ideal_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try: obj = json.loads(line)
                except Exception: continue
                if str(obj.get("question","")).strip().lower() == args.question.strip().lower():
                    ideal_txt = norm(obj.get("ideal_answer",""))
                    break
    if ideal_txt:
        if emb.mdl is not None:
            vi = emb.encode([ideal_txt])[0]
            vb = emb.encode([base_ans])[0]
            vqlo = emb.encode([q_ans])[0]
            rel_i_base = cos(vb, vi)
            rel_i_qlora = cos(vqlo, vi)
        else:
            rel_i_base = jaccard(base_ans, ideal_txt)
            rel_i_qlora = jaccard(q_ans, ideal_txt)
    else:
        rel_i_base = rel_i_qlora = ""

    # Write markdown with both answers
    md_path = os.path.join(out_dir, "answers.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Validation run\n\n")
        f.write(f"**subject_id:** {args.subject_id}\n\n")
        f.write(f"**question:** {args.question}\n\n")
        f.write(f"## BASE ( {base_ms} ms, {base_len} words )\n\n{base_ans}\n\n")
        f.write(f"## QLORA ( {q_ms} ms, {q_len} words )\n\n{q_ans}\n")

    # Write CSV (append)
    csv_path = os.path.join(out_dir, "metrics.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow([
                "subject_id","question",
                "lat_ms_base","lat_ms_qlora",
                "len_base","len_qlora",
                "rel_to_question_base","rel_to_question_qlora",
                "rel_to_ideal_base","rel_to_ideal_qlora",
                "ans_base","ans_qlora"
            ])
        w.writerow([
            args.subject_id, args.question,
            base_ms, q_ms,
            base_len, q_len,
            f"{rel_q_base:.4f}" if isinstance(rel_q_base,float) else "",
            f"{rel_q_qlora:.4f}" if isinstance(rel_q_qlora,float) else "",
            f"{rel_i_base:.4f}" if isinstance(rel_i_base,float) else "",
            f"{rel_i_qlora:.4f}" if isinstance(rel_i_qlora,float) else "",
            clean_csv(base_ans), clean_csv(q_ans)
        ])

    # Tarball + SCP helper
    tgz = f"{out_dir}.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(out_dir, arcname=os.path.basename(out_dir))

    # Print console summary + scp line you can paste in PowerShell
    print("\n===== BASE ANSWER =====\n" + base_ans + f"\n\n----- ({base_ms:,} ms) -----\n")
    print("===== QLORA ANSWER =====\n" + q_ans + f"\n\n----- ({q_ms:,} ms) -----\n")
    print(f"[INFO] Saved:\n  {md_path}\n  {csv_path}\n  {tgz}")

    user = os.environ.get("USER","YOURID")
    host = "cci-p127.cci.drexel.edu"
    win = r'C:/Users/dmeve/OneDrive/Drexel/CS614/'
    print("\n# From your Windows PC (PowerShell):")
    print(f'scp -J {user}@tux.cs.drexel.edu {user}@{host}:"{tgz}" "{win}"')

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
