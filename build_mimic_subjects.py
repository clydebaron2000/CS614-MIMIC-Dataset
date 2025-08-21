import os, json, random, math
import pandas as pd

# ---------- CONFIG ----------
HOSP_DIR    = "/storage/mimic_data/hosp_csvs"
ICU_DIR     = "/storage/mimic_data/icu_csvs"   # set to "" if ICU not ready
OUTPUT_DIR  = "/storage/mimic_data/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "mimic_iv_subjects_sample10.jsonl")

RANDOM_SEED   = 42
CHUNKSIZE     = 200_000
SAMPLE_PCT    = 0.10

# Columns to ignore everywhere
IGNORE_COLS = {
    "admittime",
    "anchor_year",
    "anchor_year_group",
    "edouttime",
    "edregtime",
    "insurance",
    "language",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def list_csv_like(folder):
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if fn.lower().endswith(".csv") or fn.lower().endswith(".csv.gz")
    )

def table_name_from_path(path):
    base = os.path.basename(path)
    for ext in (".csv.gz", ".csv"):
        if base.lower().endswith(ext):
            return base[:-len(ext)]
    return os.path.splitext(base)[0]

def find_table(files, wanted_name):
    wanted = wanted_name.lower()
    for p in files:
        if table_name_from_path(p).lower() == wanted:
            return p
    return None

def read_patients_subjects(patients_path):
    print(f"[LOAD] universe from {patients_path}", flush=True)
    df = pd.read_csv(
        patients_path,
        compression="infer",
        usecols=["subject_id"],
        dtype={"subject_id": "Int64"},
        low_memory=False,
    )
    return df["subject_id"].dropna().astype("int64").unique().tolist()

def sample_subjects(all_subjects, pct=SAMPLE_PCT, seed=RANDOM_SEED):
    random.seed(seed)
    n_total = len(all_subjects)
    n_sample = max(1, int(math.floor(n_total * pct)))
    return set(random.sample(list(all_subjects), n_sample))

def build_hadm_to_subject_map(admissions_path, sampled_subjects):
    """
    Build hadm_id -> subject_id map, restricted to sampled subjects.
    Used when a table lacks subject_id but has hadm_id.
    """
    if not admissions_path:
        print("[INFO] admissions table not found; cannot map hadm_id -> subject_id", flush=True)
        return {}
    print(f"[MAP] building hadm_id->subject_id from {admissions_path}", flush=True)
    m = {}
    usecols = ["subject_id", "hadm_id"]
    for chunk in pd.read_csv(
        admissions_path, compression="infer", usecols=usecols,
        dtype={"subject_id": "Int64", "hadm_id": "Int64"},
        chunksize=CHUNKSIZE, low_memory=False
    ):
        chunk = chunk.dropna(subset=["subject_id", "hadm_id"])
        sub = chunk[chunk["subject_id"].astype("int64").isin(sampled_subjects)]
        if sub.empty:
            continue
        for sid, hadm in zip(sub["subject_id"].astype("int64"), sub["hadm_id"].astype("int64")):
            m[int(hadm)] = int(sid)
    print(f"[MAP] mapped {len(m):,} admissions", flush=True)
    return m

def stream_tables(files, sampled_subjects, hadm_to_subject):
    """
    Collect rows per sampled subject across tables.
    - If 'subject_id' exists: use it.
    - Else if 'hadm_id' exists: map via hadm_to_subject; unmapped rows skipped.
    - Else: skip table.
    Drop IGNORE_COLS when present.
    """
    store = {sid: {"subject_id": int(sid)} for sid in sampled_subjects}
    for path in files:
        tname = table_name_from_path(path)
        try:
            head = pd.read_csv(path, nrows=0, compression="infer", low_memory=False)
        except Exception as e:
            print(f"[WARN] header read failed for {tname}: {e}", flush=True)
            continue

        has_sid = "subject_id" in head.columns
        has_hadm = "hadm_id" in head.columns
        if not has_sid and not has_hadm:
            print(f"[SKIP] {tname}: neither subject_id nor hadm_id present", flush=True)
            continue

        print(f"[SCAN] {tname} [{'subject_id' if has_sid else 'hadm_id->subject_id'}]", flush=True)
        try:
            for chunk in pd.read_csv(path, compression="infer", low_memory=False, chunksize=CHUNKSIZE):
                if has_sid:
                    chunk["subject_id"] = pd.to_numeric(chunk["subject_id"], errors="coerce").astype("Int64")
                    chunk = chunk[chunk["subject_id"].isin(sampled_subjects)]
                    if chunk.empty: 
                        continue
                else:
                    if "hadm_id" not in chunk.columns:
                        continue
                    chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
                    sid_mapped = chunk["hadm_id"].map(lambda x: int(hadm_to_subject.get(int(x)) if pd.notna(x) else None))
                    chunk = chunk.assign(subject_id=sid_mapped).dropna(subset=["subject_id"])
                    chunk["subject_id"] = chunk["subject_id"].astype("int64")
                    chunk = chunk[chunk["subject_id"].isin(sampled_subjects)]
                    if chunk.empty:
                        continue

                # drop ignored columns
                to_drop = [c for c in IGNORE_COLS if c in chunk.columns]
                if to_drop:
                    chunk = chunk.drop(columns=to_drop)

                # NaNs -> None for JSON
                chunk = chunk.where(pd.notna(chunk), None)

                for sid, group in chunk.groupby("subject_id"):
                    sid = int(sid)
                    rows = group.to_dict(orient="records")
                    store.setdefault(sid, {}).setdefault(tname, []).extend(rows)
        except Exception as e:
            print(f"[WARN] streaming failed for {tname}: {e}", flush=True)
        else:
            print(f"[OK]  {tname}", flush=True)
    return store

def main():
    hosp_files = list_csv_like(HOSP_DIR)
    icu_files  = list_csv_like(ICU_DIR)
    all_files  = hosp_files + icu_files
    if not all_files:
        raise SystemExit(f"No CSV/CSV.GZ files found in {HOSP_DIR!r} or {ICU_DIR!r}")

    patients_path   = find_table(hosp_files, "patients")
    admissions_path = find_table(hosp_files, "admissions")
    if patients_path is None:
        raise FileNotFoundError(f"patients.csv(.gz) not found in {HOSP_DIR}")

    all_subjects = read_patients_subjects(patients_path)
    print(f"[INFO] Total unique subject_id: {len(all_subjects):,}", flush=True)
    sampled_subjects = sample_subjects(all_subjects)
    print(f"[INFO] Sampled {len(sampled_subjects):,} subjects (~{int(SAMPLE_PCT*100)}%)", flush=True)

    hadm_to_subject = build_hadm_to_subject_map(admissions_path, sampled_subjects)

    subjects_data = stream_tables(all_files, sampled_subjects, hadm_to_subject)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sid in sorted(subjects_data.keys()):
            f.write(json.dumps(subjects_data[sid], ensure_ascii=False) + "\n")
            n += 1
    print(f"[DONE] wrote {n} subject records -> {OUTPUT_FILE}", flush=True)

if __name__ == "__main__":
    main()
