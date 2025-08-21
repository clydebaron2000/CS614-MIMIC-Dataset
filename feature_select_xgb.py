#!/usr/bin/env python3
import os, json, math, sys, csv, warnings
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score
)
import xgboost as xgb

# ---------------- CONFIG ----------------
INPUT_JSONL = "/storage/mimic_data/output/mimic_iv_subjects_sample10_with_dx_px.jsonl"
OUTPUT_DIR  = "/storage/mimic_data/output"
RANDOM_SEED = 42
TEST_SIZE   = 0.2
N_ESTIMATORS= 400
MAX_DEPTH   = 8
LEARNING_RATE = 0.06
SUBSAMPLE   = 0.8
COLSAMPLE   = 0.8
N_JOBS      = 8   # tune for your CPU

# The 20 target diagnosis codes you provided:
TARGET_CODES = [
    "4019","E785","I10","2724","Z87891","K219","53081","25000",
    "F329","I2510","F419","41401","42731","311","N179","4280",
    "Z20822","V1582","Z7901","2449"
]

# Optional: additional subject-level keys to pull if present (kept tiny & generic).
# If your subject JSON includes these, we'll use them; otherwise they are ignored.
OPTIONAL_SUBJECT_KEYS = ["age","sex","anchor_age","anchor_year"]


# ---------------- HELPERS ----------------
def safe_len(x):
    return len(x) if isinstance(x, list) else 0

def load_subjects(jsonl_path):
    """Load JSONL to Python objects (streaming)."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_row_features(obj):
    """
    Build a dict of numeric/categorical features from a subject JSON.
    Token-efficient, leakage-safe baseline:
      - counts per list table (prefix rows_)
      - include optional subject scalar keys if present
      - EXCLUDE diagnoses_icd_labeled (label leakage)
    """
    feats = {}
    for k, v in obj.items():
        if k == "diagnoses_icd_labeled":
            continue
        if isinstance(v, list):
            feats[f"rows_{k}"] = len(v)
    # add tiny set of optional scalar fields (if present)
    for k in OPTIONAL_SUBJECT_KEYS:
        if k in obj and isinstance(obj[k], (int, float, str)):
            feats[k] = obj[k]
    # normalize a couple of simple categoricals
    if "sex" in feats and isinstance(feats["sex"], str):
        feats["sex"] = feats["sex"].strip().upper() or "UNK"
    return feats

def build_label_row(obj, code_set):
    """Return dict {code: 0/1} for the TARGET_CODES based on diagnoses_icd_labeled."""
    dx = obj.get("diagnoses_icd_labeled", []) or []
    present = {str(d.get("icd_code","")).strip() for d in dx}
    return {code: int(code in present) for code in code_set}

def topk_from_series(series, k=25):
    """Return list of (feature, importance) sorted, top-k, skipping zeros."""
    s = series[series > 0].sort_values(ascending=False)
    return list(s.head(k).items())

def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------- MAIN ----------------
def main():
    ensure_output_dir(OUTPUT_DIR)

    # -------- Load & vectorize --------
    print(f"[LOAD] {INPUT_JSONL}")
    feat_dicts = []
    label_dicts = []
    sids = []

    code_set = set(TARGET_CODES)

    for obj in load_subjects(INPUT_JSONL):
        sids.append(obj.get("subject_id"))
        feat_dicts.append(build_row_features(obj))
        label_dicts.append(build_label_row(obj, code_set))

    print(f"[INFO] subjects: {len(sids)}")

    # DictVectorizer -> numeric matrix
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(feat_dicts)
    feature_names = vec.get_feature_names_out()

    Y = pd.DataFrame(label_dicts, columns=TARGET_CODES).astype(int)

    # Drop columns with zero variance (all 0 or all same value)
    var_mask = (X.std(axis=0) > 0)
    X = X[:, var_mask]
    feature_names = feature_names[var_mask]
    print(f"[INFO] features: {len(feature_names)} (after variance filter)")

    # -------- Split --------
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y.values, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # -------- Train one XGB per label --------
    per_label_importances = []   # list of pd.Series aligned to feature_names
    metrics_rows = []

    for j, code in enumerate(TARGET_CODES):
        y_tr = X_train, Y_train[:, j]
        y_te = X_test,  Y_test[:, j]

        ytr = Y_train[:, j]
        n_pos = int(ytr.sum())
        n_neg = int(len(ytr) - n_pos)
        # scale_pos_weight helps class imbalance; avoid div by zero
        spw = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0

        clf = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE,
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS,
            tree_method="hist",   # use "gpu_hist" if GPU available
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=spw
        )

        print(f"[TRAIN] {code}  (pos={n_pos}, neg={n_neg}, spw={spw:.2f})")
        clf.fit(X_train, ytr)

        # Importance (gain)
        booster = clf.get_booster()
        fmap = {f"f{idx}": name for idx, name in enumerate(feature_names)}
        raw_imp = booster.get_score(importance_type="gain")

        # Align to feature_names
        imp_vec = np.zeros(len(feature_names), dtype=float)
        for fkey, val in raw_imp.items():
            idx = int(fkey[1:])  # "f123" -> 123
            if 0 <= idx < len(imp_vec):
                imp_vec[idx] = float(val)
        imp_series = pd.Series(imp_vec, index=feature_names)
        per_label_importances.append(imp_series)

        # Metrics
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # robust metrics set (AUROC/AP if possible)
        try:
            auroc = roc_auc_score(Y_test[:, j], y_pred_proba)
        except Exception:
            auroc = float("nan")

        try:
            ap = average_precision_score(Y_test[:, j], y_pred_proba)
        except Exception:
            ap = float("nan")

        acc = accuracy_score(Y_test[:, j], y_pred)
        f1  = f1_score(Y_test[:, j], y_pred, zero_division=0)

        metrics_rows.append({
            "code": code, "pos_train": n_pos, "neg_train": n_neg,
            "AUROC": auroc, "AP": ap, "ACC": acc, "F1": f1
        })

        # Write per-label top-25 CSV
        top25 = topk_from_series(imp_series, k=25)
        out_csv = os.path.join(OUTPUT_DIR, f"top_features_xgb_{code}.csv")
        with open(out_csv, "w", encoding="utf-8", newline="") as fout:
            w = csv.writer(fout)
            w.writerow(["feature", "importance_gain"])
            for feat, score in top25:
                w.writerow([feat, f"{score:.6g}"])
        print(f"[WROTE] {out_csv}  (top 25 features)")

    # -------- Aggregate overall importance --------
    if per_label_importances:
        # Normalize each series to sum=1 to prevent labels with larger raw gain dominating
        normed = []
        for s in per_label_importances:
            s_sum = s.sum()
            if s_sum > 0:
                normed.append(s / s_sum)
            else:
                normed.append(s)
        overall = pd.concat(normed, axis=1).fillna(0).mean(axis=1)
        overall_top = topk_from_series(overall, k=25)

        out_csv = os.path.join(OUTPUT_DIR, "top_features_xgb_overall.csv")
        with open(out_csv, "w", encoding="utf-8", newline="") as fout:
            w = csv.writer(fout)
            w.writerow(["feature", "mean_normalized_importance"])
            for feat, score in overall_top:
                w.writerow([feat, f"{score:.6g}"])
        print(f"[WROTE] {out_csv}  (overall top 25)")

    # -------- Write metrics table --------
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(OUTPUT_DIR, "xgb_label_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[WROTE] {metrics_csv}")

    print("\n[DONE] Feature selection complete.")
    print("Check CSVs in:", OUTPUT_DIR)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

