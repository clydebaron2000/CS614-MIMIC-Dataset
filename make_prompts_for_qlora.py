#!/usr/bin/env python3
import os, json, argparse, math, re
from collections import Counter

# ---------------- CONFIG DEFAULTS ----------------
DEFAULT_INPUT  = "/storage/mimic_data/output/mimic_iv_subjects_sample10_ft_min.jsonl"
DEFAULT_OUTPUT = "/storage/mimic_data/output/mimic_iv_subjects_sample10_prompts.jsonl"

# How much context to include to keep tokens small
MAX_TABLES_IN_PROMPT     = 10          # top-K tables by row count
MAX_PROCS_IN_PROMPT      = 10          # top-K procedure titles per subject
MAX_DIAGNOSES_IN_TARGET  = 8           # top-K diagnoses in the gold answer
MIN_DIAGNOSES_REQUIRED   = 1           # skip subjects with no diagnoses (no target)

# Optional: include diagnosis names in prompt? (default False to avoid leakage)
INCLUDE_DIAGNOSES_IN_PROMPT = False

SYSTEM_MSG = (
    "You are a careful, supportive health assistant. You cannot provide a diagnosis. "
    "Use the provided context to flag potential health issues in plain language and suggest "
    "general next steps a patient might discuss with their clinician. Keep it concise and clear."
)

PROMPT_TEMPLATE = """Patient snapshot
- Age: {age}
- Sex: {sex}
- Tables present (row counts): {table_counts}
{maybe_procs}\
Task
Based on this snapshot, list the key health issues this patient may face and 1–2 plain‑language tips for each. \
Avoid medical jargon. Do not claim certainty or provide treatment. If uncertain, say so briefly.
"""

# ---------------- HELPERS ----------------
def load_subjects(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def norm_sex(x):
    if not isinstance(x, str): return "Unknown"
    s = x.strip().upper()
    if s in ("M","MALE"): return "Male"
    if s in ("F","FEMALE"): return "Female"
    return "Unknown"

def safe_age(obj):
    # many MIMIC exports store anchor_age; fall back to "Unknown"
    for k in ("anchor_age","age"):
        if k in obj:
            try:
                v = int(float(obj[k]))
                if 0 <= v <= 130:
                    return v
            except Exception:
                pass
    return "Unknown"

def table_row_counts(obj):
    rows = []
    for k, v in obj.items():
        if isinstance(v, list):
            rows.append((k, len(v)))
    # biggest first; keep top-K
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows

def proc_titles(obj, k=10):
    px = obj.get("procedures_icd_labeled", []) or []
    # compact-schema keys: t=long_title
    titles = [p.get("t") for p in px if p.get("t")]
    # most frequent first
    c = Counter(titles)
    return [t for t,_ in c.most_common(k)]

def diagnosis_titles(obj, k=8):
    dx = obj.get("diagnoses_icd_labeled", []) or []
    titles = [d.get("long_title") for d in dx if d.get("long_title")]
    c = Counter(titles)
    return [t for t,_ in c.most_common(k)]

def make_user_prompt(obj, max_tables=10, max_procs=10, include_dx=False):
    age = safe_age(obj)
    sex = norm_sex(obj.get("sex", ""))

    # tables string
    rows = table_row_counts(obj)[:max_tables]
    table_str = ", ".join(f"{k}={n}" for k,n in rows) if rows else "none"

    # procedures (optional)
    procs = proc_titles(obj, k=max_procs)
    procs_str = ""
    if procs:
        procs_str = "- Notable procedures: " + "; ".join(procs) + "\n"

    extra_dx = ""
    if include_dx:
        dx_titles = diagnosis_titles(obj, k=MAX_DIAGNOSES_IN_TARGET)
        if dx_titles:
            extra_dx = "- Known conditions (from chart): " + "; ".join(dx_titles) + "\n"

    prompt = PROMPT_TEMPLATE.format(
        age=age,
        sex=sex,
        table_counts=table_str,
        maybe_procs=procs_str + extra_dx
    )
    return prompt.strip()

def make_assistant_answer(obj, max_dx=8):
    """
    Gold answer: patient-facing restatement of diagnoses (titles only) with brief tips.
    """
    items = diagnosis_titles(obj, k=max_dx)
    if not items:
        return None

    out_lines = []
    for title in items:
        tip = "Consider discussing screening, risk factors, and lifestyle changes with your clinician."
        # a tiny bit of tailored language for a few common patterns
        low = title.lower()
        if "hypertension" in low or "blood pressure" in low:
            tip = "Ask about blood pressure goals, home monitoring, and salt/activity guidance."
        elif "diabetes" in low:
            tip = "Review A1c goals, nutrition, and foot/eye care; ask about monitoring."
        elif "hyperlipidemia" in low or "cholesterol" in low:
            tip = "Discuss lipid targets, diet, and whether medication is appropriate."
        elif "depress" in low or "anxiety" in low:
            tip = "Discuss mood, sleep, stress supports, and options for therapy."
        elif "atrial fibrillation" in low:
            tip = "Ask about stroke prevention, heart‑rate control, and symptoms to watch."
        elif "reflux" in low or "gastro‑esophageal" in low or "esophageal reflux" in low:
            tip = "Talk about diet triggers, timing of meals, and when to consider meds."
        elif "kidney" in low:
            tip = "Ask about kidney function, hydration, and medication adjustments."
        elif "coronary" in low or "atherosclerotic" in low:
            tip = "Discuss heart risk reduction (BP, cholesterol, smoking), and warning signs."
        out_lines.append(f"- **{title}** — {tip}")

    disclaimer = (
        "These are not diagnoses. Use this list to start a conversation with your clinician. "
        "If you have urgent symptoms (chest pain, trouble breathing, stroke signs), call emergency services."
    )
    return "\n".join(out_lines + ["", disclaimer]).strip()

def to_messages(prompt, answer):
    return {
        "messages": [
            {"role":"system","content": SYSTEM_MSG},
            {"role":"user","content": prompt},
            {"role":"assistant","content": answer}
        ]
    }

# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser(description="Convert subject JSONL to chat prompt/response pairs for QLoRA.")
    ap.add_argument("--input",  default=DEFAULT_INPUT,  help="Input subject JSONL")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL of chat messages")
    ap.add_argument("--max_tables", type=int, default=MAX_TABLES_IN_PROMPT)
    ap.add_argument("--max_procs",  type=int, default=MAX_PROCS_IN_PROMPT)
    ap.add_argument("--max_dx",     type=int, default=MAX_DIAGNOSES_IN_TARGET)
    ap.add_argument("--include_dx_in_prompt", action="store_true", default=INCLUDE_DIAGNOSES_IN_PROMPT,
                    help="If set, includes known diagnoses in the prompt (not recommended; leaks labels).")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    n_in = n_out = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue

            answer = make_assistant_answer(obj, max_dx=args.max_dx)
            if not answer:
                continue  # skip subjects without diagnoses (no label)

            prompt = make_user_prompt(
                obj,
                max_tables=args.max_tables,
                max_procs=args.max_procs,
                include_dx=args.include_dx_in_prompt
            )

            rec = to_messages(prompt, answer)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[DONE] {n_in} subjects scanned -> {n_out} chat pairs written to {args.output}")

if __name__ == "__main__":
    main()
