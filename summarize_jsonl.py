import json
import pandas as pd
from collections import Counter
from tqdm import tqdm

# ---------- CONFIG ----------
INPUT_FILE = "/storage/mimic_data/output/mimic_iv_subjects_sample10_with_dx_px.jsonl"
MAX_LINES = None   # set to e.g. 5000 for a quick test

# ---------- LOAD ----------
print(f"[LOAD] Reading {INPUT_FILE} ...")
subjects = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if MAX_LINES and i >= MAX_LINES:
            break
        subjects.append(json.loads(line))

print(f"[INFO] Loaded {len(subjects)} subject records")

# ---------- BASIC STATS ----------
n_tables = Counter()
n_diag_per_subject = []
diag_counter = Counter()

for subj in tqdm(subjects, desc="Scanning subjects"):
    # Count how many tables appear
    for k in subj.keys():
        if isinstance(subj[k], list):
            n_tables[k] += len(subj[k])

    # Diagnoses
    diags = subj.get("diagnoses_icd_labeled", [])
    n_diag_per_subject.append(len(diags))
    for d in diags:
        diag_counter[(d["icd_code"], d["icd_version"], d["long_title"])] += 1

# ---------- SUMMARY ----------
print("\n=== SUBJECT-LEVEL ===")
print(f"Total subjects: {len(subjects)}")
print(f"Average #diagnoses per subject: {pd.Series(n_diag_per_subject).mean():.2f}")
print(f"Median #diagnoses per subject: {pd.Series(n_diag_per_subject).median():.0f}")
print(f"Max #diagnoses per subject: {max(n_diag_per_subject)}")

print("\n=== TOP DIAGNOSES ===")
for (code, version, title), count in diag_counter.most_common(20):
    print(f"{code} (v{version}) - {title} : {count}")

print("\n=== TABLE ROW COUNTS (across all subjects) ===")
for table, total in n_tables.most_common(20):
    print(f"{table}: {total}")
