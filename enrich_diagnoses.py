import os, json
import pandas as pd

# ---------- CONFIG ----------
HOSP_DIR       = "/storage/mimic_data/hosp_csvs"
INPUT_JSONL    = "/storage/mimic_data/output/mimic_iv_subjects_sample10.jsonl"
OUTPUT_JSONL   = "/storage/mimic_data/output/mimic_iv_subjects_sample10_with_dx.jsonl"

# Filenames (csv or csv.gz OK)
DX_FILE        = os.path.join(HOSP_DIR, "diagnoses_icd.csv")
DX_FILE_GZ     = os.path.join(HOSP_DIR, "diagnoses_icd.csv.gz")
DICT_FILE      = os.path.join(HOSP_DIR, "d_icd_diagnoses.csv")
DICT_FILE_GZ   = os.path.join(HOSP_DIR, "d_icd_diagnoses.csv.gz")

CHUNKSIZE      = 500_000   # adjust to your RAM

def pick_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of these exist: " + ", ".join(paths))

def build_icd_dict(dict_path):
    # Read as strings to keep leading zeros in icd_code; icd_version often 9/10 -> keep as string as well.
    usecols = ["icd_code", "icd_version", "long_title"]
    df = pd.read_csv(dict_path, compression="infer", usecols=usecols, dtype=str, low_memory=False)
    # Normalize code and version as strings without whitespace
    df["icd_code"] = df["icd_code"].astype(str).str.strip()
    df["icd_version"] = df["icd_version"].astype(str).str.strip()
    # Build dict keyed by (code, version)
    return {(c, v): t for c, v, t in zip(df["icd_code"], df["icd_version"], df["long_title"])}

def collect_dx_per_subject(dx_path, icd_map):
    """
    Stream diagnoses_icd; attach long_title via (icd_code, icd_version).
    Return dict: subject_id -> [ {subject_id, hadm_id, seq_num, icd_code, icd_version, long_title}, ... ]
    """
    out = {}
    usecols = ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
    for chunk in pd.read_csv(dx_path, compression="infer", usecols=usecols, dtype=str,
                             chunksize=CHUNKSIZE, low_memory=False):
        # Clean/normalize
        for col in ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(str).str.strip()
        # Join titles
        chunk["long_title"] = [
            icd_map.get((c or "", v or ""), None)
            for c, v in zip(chunk.get("icd_code", []), chunk.get("icd_version", []))
        ]
        # Group and collect
        for sid, group in chunk.groupby("subject_id", dropna=True):
            if sid is None or sid == "" or sid == "nan":
                continue
            rows = group.to_dict(orient="records")
            out.setdefault(int(sid), []).extend(rows)
    return out

def main():
    dx_path   = pick_existing(DX_FILE, DX_FILE_GZ)
    dict_path = pick_existing(DICT_FILE, DICT_FILE_GZ)
    print(f"[INFO] Using diagnoses file:      {dx_path}")
    print(f"[INFO] Using ICD dictionary file: {dict_path}")

    print("[STEP] Build ICD map...")
    icd_map = build_icd_dict(dict_path)
    print(f"[OK] ICD entries: {len(icd_map):,}")

    print("[STEP] Collect diagnoses per subject (streaming)...")
    per_subject_dx = collect_dx_per_subject(dx_path, icd_map)
    print(f"[OK] Subjects with diagnoses: {len(per_subject_dx):,}")

    print("[STEP] Write augmented JSONL...")
    n_in, n_out = 0, 0
    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            obj = json.loads(line)
            sid = int(obj.get("subject_id"))
            # attach labeled diagnoses (may be absent for some subjects)
            if sid in per_subject_dx:
                obj["diagnoses_icd_labeled"] = per_subject_dx[sid]
            else:
                obj["diagnoses_icd_labeled"] = []
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[DONE] {n_in} -> {n_out} subjects written to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
