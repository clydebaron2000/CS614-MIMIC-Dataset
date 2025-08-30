
import os, json, random, math
import pandas as pd

# ---------- CONFIG ----------
HOSP_DIR    = r"\storage\mimic_data\hosp_csvs" #update to your path
ICU_DIR     = r"\storage\mimic_data\icu_csvs" #update to your path
OUTPUT_DIR  = r"\storage\mimic_data\output" #update to your path
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "mimic_iv_subjects_sample10.jsonl")

RANDOM_SEED   = 42          # set for reproducibility
CHUNKSIZE     = 200_000     # tune based on your machine
SAMPLE_PCT    = 0.10        # 10%

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- HELPERS ----------
def list_csv_like(folder):
    """Return absolute paths for *.csv and *.csv.gz in a folder."""
    files = []
    for fn in os.listdir(folder):
        low = fn.lower()
        if low.endswith(".csv") or low.endswith(".csv.gz"):
            files.append(os.path.join(folder, fn))
    return sorted(files)

def table_name_from_path(path):
    """Return table name like 'admissions' from '.../admissions.csv(.gz)'."""
    base = os.path.basename(path)
    for ext in (".csv.gz", ".csv"):
        if base.endswith(ext):
            return base[:-len(ext)]
    return os.path.splitext(base)[0]

def read_patients_subjects(patients_path):
    """Read all subject_ids from patients.csv(.gz)."""
    print(f"[LOAD] {patients_path} (for subject_id universe)")
    # patients is small-ish; OK to load fully
    df = pd.read_csv(patients_path, compression="infer", usecols=["subject_id"], low_memory=False)
    subs = df["subject_id"].dropna().astype(int).unique().tolist()
    return subs

def sample_subjects(all_subjects, pct=SAMPLE_PCT, seed=RANDOM_SEED):
    random.seed(seed)
    n_total = len(all_subjects)
    n_sample = max(1, int(math.floor(n_total * pct)))
    return set(random.sample(list(all_subjects), n_sample))

def scan_and_collect(files, sampled_subjects):
    """
    Walk all CSV/CSV.GZs; for any table with a subject_id column, collect rows for sampled subjects.
    Returns a dict: { subject_id -> { 'subject_id': int, '<table>': [rows,...], ... } }
    """
    store = {sid: {"subject_id": int(sid)} for sid in sampled_subjects}

    for path in files:
        tname = table_name_from_path(path)
        print(f"[SCAN] {tname} ...")

        # First peek header to see if 'subject_id' exists
        try:
            head = pd.read_csv(path, nrows=0, compression="infer", low_memory=False)
        except Exception as e:
            print(f"  [WARN] Cannot read header for {path}: {e}")
            continue

        if "subject_id" not in head.columns:
            print(f"  [SKIP] {tname} has no subject_id column; skipped for this subject-level build.")
            continue

        # Stream rows in chunks and append to per-subject buckets
        for chunk in pd.read_csv(path, compression="infer", low_memory=False, chunksize=CHUNKSIZE):
            if "subject_id" not in chunk.columns:
                continue
            # Keep only sampled subjects in this chunk
            filt = chunk["subject_id"].isin(sampled_subjects)
            if not filt.any():
                continue

            sub = chunk.loc[filt]
            # Ensure plain Python types for downstream JSON (leave final sanitize to json.dumps)
            sub = sub.where(pd.notna(sub), None)

            # Group rows by subject and append
            for sid, group in sub.groupby("subject_id"):
                sid = int(sid)
                rows = group.to_dict(orient="records")
                # initialize list for this table
                if tname not in store[sid]:
                    store[sid][tname] = []
                store[sid][tname].extend(rows)

        print(f"  [OK] collected rows for sampled subjects from {tname}")
    return store

# ---------- DISCOVER FILES ----------
hosp_files = list_csv_like(HOSP_DIR)
icu_files  = list_csv_like(ICU_DIR)
all_files  = hosp_files + icu_files

# Find patients.csv(.gz)
patients_path = None
for p in hosp_files:
    if table_name_from_path(p).lower() == "patients":
        patients_path = p
        break
if patients_path is None:
    raise FileNotFoundError("Could not find patients.csv or patients.csv.gz in the HOSP folder.")

# ---------- SAMPLE SUBJECTS ----------
all_subjects = read_patients_subjects(patients_path)
print(f"[INFO] Total unique subject_id: {len(all_subjects)}")
sampled_subjects = sample_subjects(all_subjects, pct=SAMPLE_PCT, seed=RANDOM_SEED)
print(f"[INFO] Sampled ~{int(SAMPLE_PCT*100)}%: {len(sampled_subjects)} subjects")

# ---------- COLLECT ROWS PER SUBJECT ----------
subjects_data = scan_and_collect(all_files, sampled_subjects)

# ---------- WRITE JSONL (one subject per line) ----------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for sid in sorted(subjects_data.keys()):
        line = json.dumps(subjects_data[sid], ensure_ascii=False)
        f.write(line + "\n")

print(f"[DONE] Wrote {len(subjects_data)} subject records -> {OUTPUT_FILE}")
