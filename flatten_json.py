import json
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

INPUT_FILE = "/storage/mimic_data/output/mimic_iv_subjects_sample10_with_dx_px.jsonl"

rows = []
targets = []

top_dx_codes = {
    "4019", "E785", "I10", "2724", "Z87891", "K219", "53081", "25000",
    "F329", "I2510", "F419", "41401", "42731", "311", "N179", "4280",
    "Z20822", "V1582", "Z7901", "2449"
}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        sid = obj["subject_id"]

        # --- Features (super simple baseline: counts of each table) ---
        feats = {f"rows_{k}": len(v) for k, v in obj.items() if isinstance(v, list)}
        feats["age"] = obj.get("anchor_age", 0) if "anchor_age" in obj else 0
        feats["sex"] = obj.get("sex", "UNK")

        rows.append(feats)

        # --- Targets (binary vector for diagnoses) ---
        dxs = obj.get("diagnoses_icd_labeled", [])
        present = {d["icd_code"] for d in dxs}
        targets.append({dx: int(dx in present) for dx in top_dx_codes})

# Convert features to numeric matrix
vec = DictVectorizer(sparse=False)
X = vec.fit_transform(rows)
feature_names = vec.get_feature_names_out()

# Targets: multi-label matrix
Y = pd.DataFrame(targets)
