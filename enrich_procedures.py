import os, json
import pandas as pd

# ---------- CONFIG ----------
HOSP_DIR       = "/storage/mimic_data/hosp_csvs"
INPUT_JSONL    = "/storage/mimic_data/output/mimic_iv_subjects_sample10_with_dx.jsonl"
OUTPUT_JSONL   = "/storage/mimic_data/output/mimic_iv_subjects_sample10_with_dx_px.jsonl"

# Filenames (csv or csv.gz OK)
PX_FILE      = os.path.join(HOSP_DIR, "procedures_icd.csv")
PX_FILE_GZ   = os.path.join(HOSP_DIR, "procedures_icd.csv.gz")
DICT_FILE    = os.path.join(HOSP_DIR, "d_icd_procedures.csv")
DICT_FILE_GZ = os.path.join(HOSP_DIR, "d_icd_procedures.csv.gz")

CHUNKSIZE    = 500_000    # adjust for RAM

# To keep tokens down, we use compact keys:
#   h=hadm_id, s=seq_num, d=chartdate, c=icd_code, v=icd_version, t=long_title
# (We can drop chartdate by setting KEEP_DATE=False if you want it even smaller.)
KEEP_DATE = True

def pick_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of these exist: " + ", ".join(paths))

def load_subject_ids_from_jsonl(path):
    """Load the set of subject_ids present in the input JSONL so we only collect what we need."""
    sids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            sid = obj.get("subject_id")
            if sid is not None:
                sids.add(int(sid))
    return sids

def build_icd_proc_dict(dict_path):
    """Return {(code, version): long_title}, keep strings to preserve leading zeros."""
    usecols = ["icd_code", "icd_version", "long_title"]
    df = pd.read_csv(dict_path, compression="infer", usecols=usecols, dtype=str, low_memory=False)
    df["icd_code"] = df["icd_code"].astype(str).str.strip()
    df["icd_version"] = df["icd_version"].astype(str).str.strip()
    df["long_title"] = df["long_title"].astype(str).str.strip()
    return {(c, v): t for c, v, t in zip(df["icd_code"], df["icd_version"], df["long_title"])}

def collect_px_per_subject(px_path, icd_map, wanted_subjects):
    """
    Stream procedures_icd; attach long_title; keep only subjects in wanted_subjects.
    Return dict: sid -> [ {h,s,d,c,v,t}, ... ] (compact keys)
    """
    out = {}
    usecols = ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"]
    for chunk in pd.read_csv(px_path, compression="infer", usecols=usecols, dtype=str,
                             chunksize=CHUNKSIZE, low_memory=False):
        # normalize strings and strip
        for col in usecols:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(str).str.strip()
        # restrict to subjects we care about (keeps memory low)
        chunk["subject_id_int"] = pd.to_numeric(chunk["subject_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["subject_id_int"])
        chunk["subject_id_int"] = chunk["subject_id_int"].astype("int64")
        chunk = chunk[chunk["subject_id_int"].isin(wanted_subjects)]
        if chunk.empty:
            continue
        # map long titles
        codes = chunk.get("icd_code", pd.Series([], dtype=str)).astype(str)
        vers  = chunk.get("icd_version", pd.Series([], dtype=str)).astype(str)
        chunk["long_title"] = [icd_map.get((c, v), None) for c, v in zip(codes, vers)]

        # build compact dicts
        for sid, group in chunk.groupby("subject_id_int"):
            rows = []
            for _, r in group.iterrows():
                item = {
                    "h": r.get("hadm_id"),
                    "s": r.get("seq_num"),
                    "c": r.get("icd_code"),
                    "v": r.get("icd_version"),
                    "t": r.get("long_title"),
                }
                if KEEP_DATE:
                    item["d"] = r.get("chartdate")
                rows.append(item)
            out.setdefault(int(sid), []).extend(rows)
    return out

def main():
    px_path   = pick_existing(PX_FILE, PX_FILE_GZ)
    dict_path = pick_existing(DICT_FILE, DICT_FILE_GZ)
    print(f"[INFO] procedures file: {px_path}")
    print(f"[INFO] proc dictionary: {dict_path}")

    # Which subjects do we need?
    print("[STEP] Loading subject_id universe from input JSONL ...")
    wanted_subjects = load_subject_ids_from_jsonl(INPUT_JSONL)
    print(f"[OK] subjects in JSONL: {len(wanted_subjects):,}")

    # Map (code,version) -> title
    print("[STEP] Building procedures ICD map ...")
    icd_map = build_icd_proc_dict(dict_path)
    print(f"[OK] ICD procedure entries: {len(icd_map):,}")

    # Collect procedures per subject (only the ones in JSONL)
    print("[STEP] Collecting procedures per subject (streaming) ...")
    per_subject_px = collect_px_per_subject(px_path, icd_map, wanted_subjects)
    print(f"[OK] subjects with procedures: {len(per_subject_px):,}")

    # Merge into JSONL with compact key 'procedures_icd_labeled'
    print("[STEP] Writing augmented JSONL ...")
    n_in = n_out = 0
    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            obj = json.loads(line)
            sid = int(obj.get("subject_id"))
            obj["procedures_icd_labeled"] = per_subject_px.get(sid, [])
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"[DONE] {n_in} -> {n_out} subjects -> {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
