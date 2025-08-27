#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time, csv, argparse, random, tarfile, math
import re, unicodedata, difflib
from typing import Dict, Tuple, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from peft import PeftModel

# -------------- Small utils --------------

def words(s: str) -> int:
    return len((s or "").strip().split())

def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    # assume a,b are already L2-normalized 2D (1,d)
    return float(torch.mm(a, b.t()).item())

def _norm_q(s: str) -> str:
    # Unicode-normalize, lowercase, unify spaces and dash-like chars
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"[–—−-]+", "-", s)   # hyphenize all dash variants
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_file_text(path: Optional[str]) -> Optional[str]:
    if not path: return None
    if not os.path.isfile(path): return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# -------------- Embeddings --------------

class HFEmbedder:
    """
    Minimal sentence embedding with mean-pool over last_hidden_state.
    Default model: sentence-transformers/all-MiniLM-L6-v2 (works offline if cached).
    """
    def __init__(self, model_id="sentence-transformers/all-MiniLM-L6-v2",
                 cache_dir=None, local_only=False, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {}
        if cache_dir: kw["cache_dir"] = cache_dir
        if local_only or os.environ.get("HF_HUB_OFFLINE") == "1":
            kw["local_files_only"] = True
        self.tok = AutoTokenizer.from_pretrained(model_id, **kw)
        self.mod = AutoModel.from_pretrained(model_id, **kw).to(self.device)
        self.mod.eval()

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: (B,T,D); attention_mask: (B,T)
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B,T,1)
        summed = (last_hidden_state * mask).sum(dim=1)                  # (B,D)
        counts = mask.sum(dim=1).clamp(min=1e-6)                        # (B,1)
        sent = summed / counts
        # L2 normalize
        sent = torch.nn.functional.normalize(sent, p=2, dim=1)
        return sent

    def encode(self, texts: List[str]) -> torch.Tensor:
        enc = self.tok(texts, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.mod(**enc)
        emb = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
        return emb.detach().cpu()

# -------------- Subject/context helpers --------------

def find_subject_record(path: str, subj_id: str):
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
            candidates = [
                obj.get("subject_id"),
                obj.get("SUBJECT_ID"),
                obj.get("id"),
                obj.get("subjectId"),
            ]
            if obj.get("subject") and isinstance(obj["subject"], dict):
                candidates.append(obj["subject"].get("subject_id"))
                candidates.append(obj["subject"].get("SUBJECT_ID"))
            candidates = [str(x) for x in candidates if x is not None]
            if want in candidates:
                return obj
    return None

def extract_context_text(record: dict, fallback_id: str) -> str:
    for key in ("patient_context", "Patient context", "context"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    msgs = record.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                cont = m.get("content") or ""
                if isinstance(cont, str) and cont.strip():
                    if cont.lower().startswith("patient context"):
                        return cont.split("\n\nQuestion")[0].strip()
                    return cont.strip()
    sid = record.get("subject_id") or record.get("SUBJECT_ID") or fallback_id
    return f"Patient context\n- Subject ID: {sid}\n- (No structured context block found in source file.)"

# -------------- Model loading & generation --------------

def load_tokenizer(base_model_id: str):
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def _bnb_config():
    try:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    except Exception:
        return None

def load_base_model(base_model_id: str):
    bnb = _bnb_config()
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    if bnb is not None:
        kw["quantization_config"] = bnb
    model = AutoModelForCausalLM.from_pretrained(base_model_id, **kw)
    model.eval()
    return model

def load_qlora_model(base_model_id: str, adapter_dir: str):
    base = load_base_model(base_model_id)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model

def clamp_for_vram(tok, messages, max_input_tokens=700) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize chat -> left-truncate -> ALWAYS return 2D Long tensors:
    (1, seq_len) input_ids and (1, seq_len) attention_mask.
    We build the mask ourselves to avoid tokenizer quirks.
    """
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids_1d = enc["input_ids"][0].to(dtype=torch.long)
    if ids_1d.size(0) > max_input_tokens:
        ids_1d = ids_1d[-max_input_tokens:]
    mask_1d = torch.ones_like(ids_1d, dtype=torch.long)
    input_ids = ids_1d.unsqueeze(0).contiguous()
    attention = mask_1d.unsqueeze(0).contiguous()
    # safety:
    assert input_ids.dim() == 2 and attention.dim() == 2
    assert input_ids.size() == attention.size()
    return input_ids, attention

def generate(model, tok, system_prompt: str, context_text: str, question: str,
             max_input_tokens=700, max_new_tokens=192) -> str:
    messages = [
        {"role": "system", "content": (system_prompt or "You are a helpful assistant.").strip()},
        {"role": "user",   "content": f"{context_text}\n\nQuestion\n{question}"},
    ]
    input_ids, attention = clamp_for_vram(tok, messages, max_input_tokens)
    input_ids = input_ids.to(model.device, non_blocking=True)
    attention = attention.to(model.device, non_blocking=True)
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention,      # guaranteed 2D
            max_new_tokens=max_new_tokens,
            do_sample=False,               # deterministic for eval
            use_cache=True,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen = out[0, input_ids.size(1):]
    return tok.decode(gen, skip_special_tokens=True).strip()

# -------------- Main runner --------------

def main():
    print("[PATCH] validation_batch.py loaded from:", __file__)

    ap = argparse.ArgumentParser(description="Batch validation with single-load base & QLoRA models.")
    ap.add_argument("--subjects-file", required=True)
    ap.add_argument("--subject-ids-file", default=os.path.expanduser(f"/tmp/{os.environ.get('USER','user')}/subject_ids.txt"))
    ap.add_argument("--questions-file", default=os.path.expanduser("~/run_files/questions_guidance.txt"))
    ap.add_argument("--system-prompt-file", default=os.path.expanduser("~/run_files/triage_system.txt"))
    ap.add_argument("--ideal-file", default=os.path.expanduser("~/run_files/ideal_guidance.jsonl"))
    ap.add_argument("--adapter-dir", default=os.environ.get("ADIR", ""))  # "" for base-only
    ap.add_argument("--model-id", default=os.environ.get("MODEL_ID", "")) # local snapshot path preferred
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out-root", default=os.path.expanduser(f"/tmp/{os.environ.get('USER','user')}"))
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--echo", type=int, default=1)
    ap.add_argument("--pick", choices=["roundrobin", "random"], default="roundrobin")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True

    # Resolve base model id
    base_model_id = args.model_id.strip()
    if not base_model_id:
        # auto-detect cached Llama-3.1-8B-Instruct snapshot
        glb = os.path.expanduser("~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots")
        snaps = []
        if os.path.isdir(glb):
            for name in os.listdir(glb):
                path = os.path.join(glb, name)
                if os.path.isdir(path):
                    snaps.append(path)
        base_model_id = sorted(snaps)[-1] if snaps else ""
    if not base_model_id:
        print("[ERR] Could not determine base model path. Pass --model-id.")
        sys.exit(1)

    # Load tokenizer + models once
    print(f"[INFO] Loading tokenizer/model once (base: {base_model_id})...")
    tok = load_tokenizer(base_model_id)
    base_model = load_base_model(base_model_id)

    qlora_model = None
    if args.adapter_dir:
        print(f"[INFO] Loading QLoRA adapter once: {args.adapter_dir}")
        qlora_model = load_qlora_model(base_model_id, args.adapter_dir)

    # Load subjects
    if not os.path.isfile(args.subjects_file):
        print(f"[ERR] subjects file not found: {args.subjects_file}")
        sys.exit(1)
    with open(args.subjects_file, "r", encoding="utf-8") as f:
        # we’ll scan file on demand; no need to preload all
        pass

    # Subject IDs to evaluate
    if not os.path.isfile(args.subject_ids_file):
        print(f"[ERR] subject-ids-file not found: {args.subject_ids_file}")
        sys.exit(1)
    with open(args.subject_ids_file, "r", encoding="utf-8") as f:
        all_sids = [ln.strip() for ln in f if ln.strip()]
    if not all_sids:
        print("[ERR] subject-ids-file is empty.")
        sys.exit(1)

    # Questions (canonical)
    if not os.path.isfile(args.questions_file):
        print(f"[ERR] questions file not found: {args.questions_file}")
        sys.exit(1)
    with open(args.questions_file, "r", encoding="utf-8") as f:
        questions = [ln.strip() for ln in f if ln.strip()]
    if not questions:
        print("[ERR] questions file is empty.")
        sys.exit(1)

    # Ideal map: exact + normalized for robustness
    ideal_map: Dict[str, str] = {}
    ideal_norm_map: Dict[str, str] = {}
    if args.ideal_file and os.path.isfile(args.ideal_file):
        with open(args.ideal_file, "r", encoding="utf-8") as f:
            for i, ln in enumerate(f, 1):
                try:
                    obj = json.loads(ln)
                except Exception:
                    print(f"[WARN] invalid JSON at line {i} in {args.ideal_file}; skipping")
                    continue
                q = (obj.get("q") or "").strip()
                ideal = (obj.get("ideal") or "").strip()
                if not q or not ideal:
                    print(f"[WARN] missing q/ideal at line {i} in {args.ideal_file}; skipping")
                    continue
                ideal_map[q] = ideal
                ideal_norm_map[_norm_q(q)] = ideal
        print(f"[INFO] Loaded {len(ideal_map)} ideal rubric entries.")
    else:
        print("[INFO] No ideal file provided; rel_to_ideal_* will be NaN.")

    # Warn about questions missing exact ideal
    missing_exact = [q for q in questions if q not in ideal_map]
    if missing_exact:
        print("[WARN] The following questions have no EXACT ideal entry:")
        for q in missing_exact:
            hint = " (has normalized match)" if _norm_q(q) in ideal_norm_map else ""
            print(f"   - {q}{hint}")

    # Output dir
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, f"batch_once_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "combined_metrics.csv")

    # Write CSV header
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "subject_id", "question",
            "lat_ms_base", "lat_ms_qlora",
            "len_base", "len_qlora",
            "rel_cos_prompt_base", "rel_cos_prompt_qlora",
            "rel_to_ideal_base", "rel_to_ideal_qlora",
            "base_answer", "qlora_answer"
        ])

    # Embedder (prompt/ideal metrics)
    cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.path.expanduser("~/.cache/huggingface/transformers")
    local_only = os.environ.get("HF_HUB_OFFLINE") == "1"
    have_embed = True
    try:
        encoder = HFEmbedder(cache_dir=cache_dir, local_only=local_only)
    except Exception as e:
        print(f"[WARN] Embedding model unavailable ({e}); relevance metrics will be NaN.")
        have_embed = False

    # System prompt
    sys_prompt = load_file_text(args.system_prompt_file) or "You are a careful, supportive health assistant."

    # Pick SIDs & iterate
    rng = random.Random(42)
    if args.n > len(all_sids):
        sample_sids = all_sids[:]
    else:
        sample_sids = rng.sample(all_sids, args.n)

    # Round-robin or random over questions
    def pick_q(i: int) -> str:
        if args.pick == "roundrobin":
            return questions[i % len(questions)]
        return rng.choice(questions)

    # Trials
    for i, sid in enumerate(sample_sids, 1):
        qtext = pick_q(i-1)
        print(f"[{i}/{len(sample_sids)}] subject={sid}")
        print(f"Q: {qtext}")

        rec = find_subject_record(args.subjects_file, sid)
        if rec is None:
            print(f"[WARN] subject_id '{sid}' not found in {args.subjects_file}")
            continue
        ctx = extract_context_text(rec, sid)

        # BASE
        t0 = time.time()
        base_ans = generate(base_model, tok, sys_prompt, ctx, qtext, max_new_tokens=args.max_new_tokens)
        base_ms = int((time.time() - t0) * 1000)
        print(f"-- BASE ({base_ms} ms, {words(base_ans)}w) --")
        if args.echo:
            print(base_ans or "<empty>")
            print()

        # QLoRA
        q_ans, q_ms = "", 0
        if qlora_model is not None:
            t1 = time.time()
            q_ans = generate(qlora_model, tok, sys_prompt, ctx, qtext, max_new_tokens=args.max_new_tokens)
            q_ms = int((time.time() - t1) * 1000)
            print(f"-- QLORA ({q_ms} ms, {words(q_ans)}w) --")
            if args.echo:
                print(q_ans or "<empty>")
                print()

        # Metrics
        base_len = words(base_ans)
        q_len = words(q_ans)

        rel_base = float("nan")
        rel_q = float("nan")
        rel_ib = float("nan")
        rel_iq = float("nan")

        if have_embed:
            q_emb = encoder.encode([qtext])
            if base_ans:
                b_emb = encoder.encode([base_ans]); rel_base = cos(q_emb, b_emb)
            if q_ans:
                qa_emb = encoder.encode([q_ans]);   rel_q = cos(q_emb, qa_emb)

            # Ideal exact-first, then normalized fallback (warn if fallback used)
            ideal_text = ideal_map.get(qtext)
            used_norm = False
            if ideal_text is None:
                ideal_text = ideal_norm_map.get(_norm_q(qtext))
                used_norm = ideal_text is not None
                if used_norm:
                    print(f"[WARN] Ideal matched only after normalization for question: {qtext}")
                else:
                    cm = difflib.get_close_matches(qtext, list(ideal_map.keys()), n=1, cutoff=0.6)
                    if cm:
                        print(f"[WARN] No ideal for: {qtext} ; closest in file: {cm[0]}")

            if ideal_text:
                i_emb = encoder.encode([ideal_text])
                if base_ans:
                    rel_ib = cos(i_emb, b_emb)
                if q_ans:
                    rel_iq = cos(i_emb, qa_emb)

        # Append row
        with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                sid, qtext,
                base_ms, q_ms,
                base_len, q_len,
                f"{rel_base:.4f}" if not math.isnan(rel_base) else "",
                f"{rel_q:.4f}"    if not math.isnan(rel_q)    else "",
                f"{rel_ib:.4f}"   if not math.isnan(rel_ib)   else "",
                f"{rel_iq:.4f}"   if not math.isnan(rel_iq)   else "",
                base_ans, q_ans
            ])
        sys.stdout.flush()

    # Tarball & SCP hint
    tgz = os.path.join(args.out_root, f"{os.path.basename(out_dir)}.tgz")
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(out_dir, arcname=os.path.basename(out_dir))

    host = os.uname().nodename if hasattr(os, "uname") else "cci-p127.cci.drexel.edu"
    print(f"\nTarball: {tgz}\n")
    # Windows PowerShell scp line with ProxyJump via tux
    print("# From your Windows PC (PowerShell):")
    print(f'scp -J {os.environ.get("USER","user")}@tux.cs.drexel.edu {os.environ.get("USER","user")}@{host}:"{tgz}" "C:/Users/dmeve/OneDrive/Drexel/CS614/"')
    print(f"\n[INFO] Done. Results under: {args.out_root} ({os.path.basename(out_dir)}/ and the .tgz tarball).")

if __name__ == "__main__":
    main()
