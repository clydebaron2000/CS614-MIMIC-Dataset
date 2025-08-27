#!/usr/bin/env python3
import os, json, argparse, random, re
from collections import Counter

# ---------------- I/O DEFAULTS ----------------
DEFAULT_INPUT  = "/storage/mimic_data/output/mimic_iv_subjects_sample10_ft_min.jsonl"
DEFAULT_OUTPUT = "/storage/mimic_data/output/mimic_iv_subjects_sample10_symptom_prompts_ctx_dxpx.jsonl"

SEED = 42
PAIRS_PER_SUBJECT = 3
MAX_TOTAL_PAIRS   = None   # set an int to cap dataset size

# ---------------- GLOBAL STYLE ----------------
SYSTEM_MSG = (
    "You are a careful, supportive health assistant. You are not a substitute for a clinician. "
    "You may not create new diagnoses; you may provide information about diagnoses that are already established in the provided context. "
    "Explain in plain language; keep it concise and empathetic. "
    "Think deeply about the question and consider potential red‑flags given the context. "
    "If a life‑threatening, red‑flag diagnosis is suggested by the question, recommend going to the Emergency Department for evaluation and briefly explain why. "
    "Briefly cite the specific items from the provided context that informed your guidance in parentheses at the end (e.g., ‘(context: diagnoses, procedures, meds)’). "
    "If you recommend over‑the‑counter medications or treatments, do not include substances listed in the subject’s allergies."
)

# ---------------- UTILITIES ----------------
def load_subjects(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def norm_sex(x):
    if not isinstance(x, str): return "Unknown"
    s = x.strip().lower()
    if s in ("m","male"): return "Male"
    if s in ("f","female"): return "Female"
    return "Unknown"

def safe_age(obj):
    for k in ("anchor_age","age"):
        if k in obj:
            try:
                v = int(float(obj[k])); 
                if 0 <= v <= 130: return v
            except Exception:
                pass
    return "Unknown"

def list_table(obj, key):
    v = obj.get(key, [])
    return v if isinstance(v, list) else []

def short_list(items, k):
    return items[:k] if len(items) > k else items

# ---------------- MED MAP (quick heuristic) ----------------
MED_CLASS_PATTERNS = {
    "statin":      r"(atorvastatin|rosuvastatin|simvastatin|pravastatin|lovastatin|pitavastatin)",
    "ace_inhib":   r"(lisinopril|enalapril|benazepril|ramipril|captopril)",
    "arb":         r"(losartan|valsartan|olmesartan|irbesartan|candesartan|telmisartan)",
    "beta_block":  r"(metoprolol|atenolol|carvedilol|propranolol|bisoprolol|nebivolol)",
    "ccb":         r"(amlodipine|diltiazem|verapamil|nifedipine)",
    "diuretic":    r"(hydrochlorothiazide|chlorthalidone|furosemide|torsemide|bumetanide|spironolactone|eplerenone)",
    "anticoag":    r"(warfarin|apixaban|rivaroxaban|edoxaban|dabigatran)",
    "antiplatelet":r"(aspirin|clopidogrel|prasugrel|ticagrelor)",
    "insulin":     r"\binsulin\b",
    "metformin":   r"\bmetformin\b",
    "sulfonylurea":r"(glipizide|glyburide|glimepiride)",
    "sglt2":       r"(empagliflozin|dapagliflozin|canagliflozin|ertugliflozin)",
    "glp1":        r"(semaglutide|liraglutide|dulaglutide|exenatide|tirzepatide)",
    "ppi":         r"(omeprazole|esomeprazole|pantoprazole|lansoprazole|rabeprazole|dexlansoprazole)",
    "h2_blocker":  r"(famotidine|ranitidine)",
    "ssri":        r"(sertraline|fluoxetine|paroxetine|citalopram|escitalopram)",
    "snri":        r"(venlafaxine|desvenlafaxine|duloxetine)",
    "bzd":         r"(lorazepam|alprazolam|diazepam|clonazepam)",
    "opioid":      r"(oxycodone|hydrocodone|morphine|fentanyl|hydromorphone|tramadol|codeine)",
    "thyroid":     r"(levothyroxine|liothyronine)",
}
MED_CLASS_RX = {k: re.compile(v, re.I) for k,v in MED_CLASS_PATTERNS.items()}

def extract_med_strings(obj, max_items=30):
    meds = []
    for key in ("pharmacy","prescriptions"):
        for row in list_table(obj, key):
            if isinstance(row, dict):
                for field in ("drug", "drug_name", "formulary_drug_cd", "generic_name", "prod_strength"):
                    val = row.get(field)
                    if isinstance(val, str) and 2 <= len(val) <= 80:
                        meds.append(val)
    meds = [re.sub(r"\s+", " ", m).strip() for m in meds]
    uniq, seen = [], set()
    for m in meds:
        lm = m.lower()
        if lm not in seen:
            uniq.append(m); seen.add(lm)
    return uniq[:max_items]

def meds_to_classes(meds):
    classes = set()
    for m in meds:
        for cls, rx in MED_CLASS_RX.items():
            if rx.search(m):
                classes.add(cls)
    return sorted(classes)

# ---------------- RISKS / DIAG / PROC ----------------
TOBACCO_CODES = {"Z87891","V1582"}  # tobacco history

def subject_dx_codes_titles(obj, k_titles=8):
    dx = list_table(obj, "diagnoses_icd_labeled")
    codes = []
    titles = []
    for d in dx:
        if isinstance(d, dict):
            c = str(d.get("icd_code","")).strip()
            t = d.get("long_title") or d.get("t")
            if c: codes.append(c)
            if t: titles.append(t)
    # collapse duplicates by title frequency
    top_titles = [t for t,_ in Counter(titles).most_common(k_titles)]
    return set(codes), top_titles

def subject_proc_titles(obj, k=5):
    px = list_table(obj, "procedures_icd_labeled")
    titles = [p.get("t") for p in px if isinstance(p, dict) and p.get("t")]
    return [t for t,_ in Counter(titles).most_common(k)]

def subject_risks(obj):
    risks = set()
    dx_codes, _ = subject_dx_codes_titles(obj)
    if dx_codes & TOBACCO_CODES:
        risks.add("tobacco_history")
    return sorted(risks)

def prior_utilization(obj):
    n_adm = len(list_table(obj, "admissions"))
    n_icu = len(list_table(obj, "icustays"))
    return n_adm, n_icu

# ---------------- PROCEDURE TRIGGERS → prompt themes ----------------
PROC_THEME_PATTERNS = {
    "cardiac": r"(angiography|angioplasty|stent|cabg|bypass|pci|catheterization)",
    "ablation": r"(ablation)",
    "endoscopy": r"(endoscopy|colonoscopy|egd|gastroscopy)",
    "dialysis": r"(dialysis|hemodialysis)",
}
PROC_THEME_RX = {k: re.compile(v, re.I) for k,v in PROC_THEME_PATTERNS.items()}

def proc_themes_from_titles(titles):
    themes = set()
    joined = " ; ".join(titles)
    for k, rx in PROC_THEME_RX.items():
        if rx.search(joined):
            themes.add(k)
    return themes

# ---------------- PROMPT BUNDLES (selected by meds, risks, dx/proc themes) ----------------
BUNDLES = {
    "reflux": {
        "when": lambda ctx: ("ppi" in ctx["med_classes"] or "h2_blocker" in ctx["med_classes"] or "reflux_dx" in ctx["flags"]),
        "user": [
            "I’ve had burning in my chest after meals and a cough at night. What can I try?",
            "My acid reflux is bad lately. Should I be taking anything over the counter?",
            "I get a sour taste and stomach discomfort after eating. Any advice?"
        ],
        "tips": [
            "Smaller meals, avoid late eating, elevate head of bed, and identify triggers (spicy, acidic, caffeine).",
            "Weight management and avoiding tobacco can help reflux symptoms.",
            "If symptoms persist, discuss short trials of acid‑reducing meds and when to investigate further."
        ],
        "red": [
            "Trouble swallowing, unintentional weight loss, vomiting blood, black stools, or severe chest pain."
        ]
    },
    "bleeding": {
        "when": lambda ctx: "anticoag" in ctx["med_classes"],
        "user": [
            "I’m on a blood thinner and noticed bruising. What should I watch for?",
            "I take a blood thinner and had back pain after a minor fall. Is that urgent?"
        ],
        "tips": [
            "Know bleeding precautions and who to call for advice if you miss a dose.",
            "Tell clinicians and dentists you’re on a blood thinner.",
            "Be cautious with contact sports and activities with fall risk."
        ],
        "red": [
            "Head injury, black/tarry stools, vomiting blood, severe unexplained bruising or weakness."
        ]
    },
    "glycemia": {
        "when": lambda ctx: any(c in ctx["med_classes"] for c in ["insulin","metformin","sglt2","glp1","sulfonylurea"]) or "diabetes_dx" in ctx["flags"],
        "user": [
            "My blood sugar’s been higher and I feel thirsty and tired. What should I do?",
            "I’m new to diabetes and not sure where to start with monitoring and diet."
        ],
        "tips": [
            "Learn how to monitor glucose and understand your A1c goal.",
            "Focus on balanced meals and regular activity; plan foot and eye checks.",
            "Discuss medication options and hypoglycemia prevention with your clinician."
        ],
        "red": [
            "Very high sugars with vomiting, confusion, or breathing changes."
        ]
    },
    "bp": {
        "when": lambda ctx: any(c in ctx["med_classes"] for c in ["ace_inhib","arb","beta_block","ccb","diuretic"]) or "htn_dx" in ctx["flags"],
        "user": [
            "My home blood pressure has been high this week. What steps should I take?",
            "I’m getting headaches and my blood pressure is often elevated. Is this dangerous?"
        ],
        "tips": [
            "Recheck BP with a proper cuff (seated, back supported, arm at heart level); keep a log.",
            "Limit salt and alcohol; consider gentle activity as tolerated.",
            "Ask your clinician about targets and whether medication changes are needed."
        ],
        "red": [
            "Severe headache, chest pain, shortness of breath, vision loss, confusion, or BP ≥180/120 with symptoms."
        ]
    },
    "cad": {
        "when": lambda ctx: ("cardiac" in ctx["proc_themes"] or "cad_dx" in ctx["flags"]),
        "user": [
            "I get chest pressure when I walk uphill that goes away with rest. What should I do?",
            "I had a stent placed before—how do I lower my risk now?"
        ],
        "tips": [
            "Discuss symptom pattern and risk control (blood pressure, cholesterol, diabetes, tobacco).",
            "Carry an action plan for chest pain; know when to call emergency services.",
            "Ask about exercise, diet, and medication adherence."
        ],
        "red": [
            "Chest pain at rest, spreading pain, sweating, shortness of breath—call emergency services."
        ]
    },
    "afib": {
        "when": lambda ctx: ("ablation" in ctx["proc_themes"] or "afib_dx" in ctx["flags"]),
        "user": [
            "My heart feels like it’s racing and irregular. Could that be atrial fibrillation?",
            "I was told I have AFib—how do I lower my stroke risk?"
        ],
        "tips": [
            "Ask about rate/rhythm control and stroke prevention.",
            "Know your anticoagulation plan and bleeding precautions.",
            "Limit excess alcohol and stimulants; track triggers."
        ],
        "red": [
            "Fainting, severe chest pain, or stroke symptoms—seek emergency care."
        ]
    },
    "mood": {
        "when": lambda ctx: any(c in ctx["med_classes"] for c in ["ssri","snri","bzd"]) or "mood_dx" in ctx["flags"],
        "user": [
            "I’ve been feeling down and anxious for weeks. What can I do?",
            "I’m having trouble sleeping and worry a lot. Any suggestions?"
        ],
        "tips": [
            "Talk with a clinician about therapy options; build a support plan.",
            "Regular sleep schedule, activity, and limiting alcohol can help.",
            "If medications are discussed, review benefits, timing, and side effects together."
        ],
        "red": [
            "Thoughts of self‑harm, inability to care for yourself, or severe worsening—seek urgent help."
        ]
    },
    "tobacco": {
        "when": lambda ctx: "tobacco_history" in ctx["risks"],
        "user": [
            "I have a chronic cough and a history of smoking. How can I quit?",
            "I used to smoke and still cough at night. What should I do?"
        ],
        "tips": [
            "Ask about medications and counseling to help quitting.",
            "Avoid triggers; plan support and follow‑up.",
            "Discuss screening (e.g., lung health) if you qualify."
        ],
        "red": [
            "Coughing blood, weight loss, or severe shortness of breath."
        ]
    },
    # generic fallbacks
    "msk": {
        "when": lambda ctx: True,
        "user": [
            "My lower back has been sore after playing sports. What can help it improve?",
            "I pulled a muscle and it’s still stiff a week later. Any safe steps?"
        ],
        "tips": [
            "Relative rest, gentle movement, heat/ice, and over‑the‑counter options if safe for you.",
            "Gradually return to activity; focus on form and core strength.",
            "If pain persists, discuss targeted physical therapy."
        ],
        "red": [
            "Severe weakness, numbness in the groin, loss of bladder/bowel control, or fever with back pain."
        ]
    },
    "cough": {
        "when": lambda ctx: True,
        "user": [
            "I’ve had a cough for a week and my stomach hurts from coughing. What should I do?",
            "I keep coughing at night. Any safe steps to feel better?"
        ],
        "tips": [
            "Hydration, rest, humidified air, and honey/lozenges may help if appropriate.",
            "Avoid smoke and known irritants; consider allergy triggers.",
            "If symptoms persist or worsen, arrange a clinician visit."
        ],
        "red": [
            "Trouble breathing, chest pain, high fever, coughing blood, or confusion."
        ]
    }
}

# Map ICD codes to flags to inform bundle selection
def dx_flags_from_codes(codes):
    flags = set()
    # Hypertension (ICD9 4019, ICD10 I10)
    if "4019" in codes or "I10" in codes:
        flags.add("htn_dx")
    # Hyperlipidemia (E785 or 2724)
    if "E785" in codes or "2724" in codes:
        flags.add("lipid_dx")
    # Diabetes (25000)
    if "25000" in codes:
        flags.add("diabetes_dx")
    # GERD (K219 or 53081)
    if "K219" in codes or "53081" in codes:
        flags.add("reflux_dx")
    # AFib (42731)
    if "42731" in codes:
        flags.add("afib_dx")
    # CAD (41401 or I2510)
    if "41401" in codes or "I2510" in codes:
        flags.add("cad_dx")
    # Mood/anxiety (311, F329, F419)
    if {"311","F329","F419"} & codes:
        flags.add("mood_dx")
    return flags

# ---------------- MESSAGE BUILDERS ----------------
def make_context_block(age, sex, med_classes, meds_tiny, n_adm, n_icu, proc_titles):
    parts = [f"Age: {age}", f"Sex: {sex}"]
    if n_adm or n_icu:
        parts.append(f"Prior stays — admissions: {n_adm}, ICU: {n_icu}")
    if med_classes:
        parts.append("Key meds/classes: " + ", ".join(sorted(med_classes)))
    elif meds_tiny:
        parts.append("Current meds (sample): " + ", ".join(meds_tiny))
    if proc_titles:
        parts.append("Past procedures: " + "; ".join(short_list(proc_titles, 4)))
    return "Patient context\n- " + "\n- ".join(parts)

def mk_user_text(bundle_key, ctx_block):
    template = random.choice(BUNDLES[bundle_key]["user"])
    return f"{ctx_block}\n\nQuestion\n{template}"

def mk_assistant_text(bundle_key, ctx, dx_titles_topN):
    # Possible diagnoses section (from chart titles)
    dx_list = dx_titles_topN
    poss = ""
    if dx_list:
        poss = "Possible diagnoses (to discuss, not definitive):\n" + "\n".join(f"- {t}" for t in dx_list) + "\n\n"

    # Tips + small personalizations
    tips = "\n".join(f"- {t}" for t in BUNDLES[bundle_key]["tips"])
       
    
    if "anticoag" in ctx["med_classes"]:
        tips += "\n- Since you take a blood thinner, watch for unusual bleeding and review interactions with your clinician."
    if any(c in ctx["med_classes"] for c in ["insulin","metformin","sglt2","glp1","sulfonylurea"]):
        tips += "\n- Because of diabetes medicines, review glucose targets and low‑sugar safety (hypoglycemia signs)."

    reds = "\n".join(f"- {r}" for r in BUNDLES[bundle_key]["red"])

    msg = (
        poss +
        "Here are general pointers that may help:\n" + tips + "\n\n" +
        "Urgent warning signs (seek emergency care if these occur):\n" + reds + "\n\n" +
        "This is not a diagnosis or a prescription. Use this to plan next steps with your clinician."
    )
    return msg.strip()

def to_record(user_text, assistant_text):
    return {
        "messages": [
            {"role":"system","content": SYSTEM_MSG},
            {"role":"user","content": user_text},
            {"role":"assistant","content": assistant_text}
        ]
    }

# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser(description="Generate contextual, dx/procedure‑grounded symptom chat pairs for QLoRA.")
    ap.add_argument("--input",  default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--pairs-per-subject", type=int, default=PAIRS_PER_SUBJECT)
    ap.add_argument("--max-total", type=int, default=MAX_TOTAL_PAIRS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-dx", type=int, default=6, help="max diagnoses to list in assistant output")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    random.seed(args.seed)

    n_in = n_out = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for subj in load_subjects(args.input):
            n_in += 1

            age = safe_age(subj)
            sex = norm_sex(subj.get("sex", ""))
            n_adm, n_icu = prior_utilization(subj)

            meds = extract_med_strings(subj, max_items=30)
            med_classes = meds_to_classes(meds)
            meds_tiny = short_list(meds, 5)

            proc_titles = subject_proc_titles(subj, k=6)
            proc_themes = proc_themes_from_titles(proc_titles)

            dx_codes, dx_titles = subject_dx_codes_titles(subj, k_titles=args.max_dx)
            dxflags = dx_flags_from_codes(dx_codes)

            risks = subject_risks(subj)

            ctx = {
                "age": age, "sex": sex,
                "med_classes": set(med_classes),
                "risks": set(risks),
                "n_adm": n_adm, "n_icu": n_icu,
                "proc_themes": set(proc_themes),
                "flags": set(dxflags),
            }

            # choose bundles that match context; ensure nonempty by allowing fallbacks
            candidates = []
            for key, b in BUNDLES.items():
                try:
                    if b["when"](ctx):
                        candidates.append(key)
                except Exception:
                    pass
            # Dedup, shuffle, limit
            seen, chosen = set(), []
            random.shuffle(candidates)
            for k in candidates:
                if k not in seen:
                    chosen.append(k); seen.add(k)
                if len(chosen) >= args.pairs_per_subject:
                    break
            if not chosen:
                chosen = ["msk"]  # ultimate fallback

            ctx_block = make_context_block(age, sex, med_classes, meds_tiny, n_adm, n_icu, proc_titles)

            for bkey in chosen:
                user_text = mk_user_text(bkey, ctx_block)
                assistant_text = mk_assistant_text(bkey, ctx, dx_titles)
                fout.write(json.dumps(to_record(user_text, assistant_text), ensure_ascii=False) + "\n")
                n_out += 1
                if args.max_total and n_out >= args.max_total:
                    break
            if args.max_total and n_out >= args.max_total:
                break

    print(f"[DONE] subjects scanned: {n_in}  -> chat pairs written: {n_out} -> {args.output}")

if __name__ == "__main__":
    main()
