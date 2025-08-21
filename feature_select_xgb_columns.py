#!/usr/bin/env python3
import os, json, sys, math, csv, warnings, re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import xgboost as xgb

# ---------------- CONFIG ----------------
INPUT_JSONL   = "/storage/mimic_data/output/mimic_iv_subjects_sample10_with_dx_px.jsonl"
OUTPUT_DIR    = "/storage/mimic_data/output"

RANDOM_SEED   = 42
TEST_SIZE     = 0.2

# XGB
N_ESTIMATORS  = 400
MAX_DEPTH     = 8
LEARNING_RATE = 0.06
SUBSAMPLE     = 0.8
COLSAMPLE     = 0.8
N_JOBS        = 8
USE_GPU       = True  # set False if GPU not available

# Targets — one-vs-rest
TARGET_CODES = [
    "4019","E785","I10","2724","Z87891","K219","53081","25000",
    "F329","I2510","F419","41401","42731","311","N179","4280",
    "Z20822","V1582","Z7901","2449"
]

# Feature engineering limits
TOPK_CATS_PER_COLUMN = 5
MAX_ROWS_PER_TABLE_SCAN = 2000

# Avoid leakage from diagnoses
SKIP_TABLE_PREFIXES = ("diagnoses",)
SKIP_EXACT_KEYS = {"diagnoses_icd_labeled"}

# Exclude specific (table.column) features everywhere (case-insensitive)
EXCLUDE_COLUMNS_EXACT = {
    "pharmacy.expiration_unit",
    "emar_detail.parent_field_ordina",  # we'll also prefix-match below
    "emar_detail.emar_seq",
}
EXCLUDE_COLUMNS_PREFIX = {
    "emar_detail.parent_field_ordina",  # full column often ends with 'ordinal'
}

# Optional scalar keys at subject root
OPTIONAL_SUBJECT_KEYS = ["age", "sex", "anchor_age", "anchor_year"]

# ---------------- helpers ----------------
def load_subjects(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def is_number(x):
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return True
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return False
        try:
            float(s)
            return True
        except Exception:
            return False
    return False

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def should_skip_table_key(k):
    if k in SKIP_EXACT_KEYS:
        return True
    lk = k.lower()
    return any(lk.startswith(p) for p in SKIP_TABLE_PREFIXES)

def is_excluded_column(table_key, col_name):
    """Return True if (table.column) should be excluded, case-insensitive, with prefix handling."""
    tk = str(table_key).lower().strip()
    ck = str(col_name).lower().strip()
    full = f"{tk}.{ck}"
    if full in EXCLUDE_COLUMNS_EXACT:
        return True
    # prefix rules
    for pref in EXCLUDE_COLUMNS_PREFIX:
        if full.startswith(pref):
            return True
    return False

# ---------------- 1) schema discovery ----------------
def discover_schema(jsonl_path):
    numeric_counts = defaultdict(lambda: defaultdict(int))
    value_counts   = defaultdict(lambda: defaultdict(Counter))
    total_counts   = defaultdict(lambda: defaultdict(int))
    rows_seen_per_table = Counter()

    for subj in load_subjects(jsonl_path):
        for tkey, rows in subj.items():
            if not isinstance(rows, list):
                continue
            if should_skip_table_key(tkey):
                continue

            remaining = MAX_ROWS_PER_TABLE_SCAN - rows_seen_per_table[tkey]
            if remaining <= 0:
                continue

            for row in rows[: max(0, remaining) ]:
                if not isinstance(row, dict):
                    continue
                rows_seen_per_table[tkey] += 1
                for col, val in row.items():
                    if is_excluded_column(tkey, col):
                        continue
                    if val is None:
                        continue
                    total_counts[tkey][col] += 1
                    if is_number(val):
                        numeric_counts[tkey][col] += 1
                    else:
                        sval = str(val).strip()
                        if len(sval) > 100:
                            sval = sval[:100]
                        value_counts[tkey][col][sval] += 1

    schema = defaultdict(dict)
    for tkey in total_counts:
        for col in total_counts[tkey]:
            if is_excluded_column(tkey, col):
                continue
            nn = total_counts[tkey][col]
            num_n = numeric_counts[tkey].get(col, 0)
            frac = (num_n / nn) if nn > 0 else 0.0
            if frac >= 0.60:
                schema[tkey][col] = {"type": "num", "top": []}
            else:
                top_vals = []
                if value_counts[tkey].get(col):
                    top_vals = [v for v, _ in value_counts[tkey][col].most_common(TOPK_CATS_PER_COLUMN)]
                schema[tkey][col] = {"type": "cat", "top": top_vals}
    return schema

# ---------------- 2) feature building ----------------
def build_features_for_subject(obj, schema):
    feats = {}
    # root-level hints
    for k in OPTIONAL_SUBJECT_KEYS:
        if k in obj and isinstance(obj[k], (int, float)):
            feats[f"root.{k}"] = float(obj[k])
        elif k in obj and isinstance(obj[k], str) and k.lower() == "sex":
            s = obj[k].strip().upper() or "UNK"
            feats[f"root.sex={s}"] = 1.0

    for tkey, rows in obj.items():
        if not isinstance(rows, list):
            continue
        if should_skip_table_key(tkey):
            continue

        feats[f"{tkey}.__rows"] = float(len(rows))  # keep a per-table count feature

        if tkey not in schema:
            continue
        colspec = schema[tkey]
        num_aggs = { col: [] for col, spec in colspec.items() if spec["type"] == "num" and not is_excluded_column(tkey, col) }
        cat_counters = { col: Counter() for col, spec in colspec.items() if spec["type"] == "cat" and spec["top"] and not is_excluded_column(tkey, col) }

        for row in rows:
            if not isinstance(row, dict):
                continue
            for col in list(num_aggs.keys()):
                if col in row and row[col] is not None and is_number(row[col]):
                    num_aggs[col].append(to_float(row[col]))
            for col, cnt in cat_counters.items():
                if col in row and row[col] is not None:
                    sval = str(row[col]).strip()
                    if len(sval) > 100: sval = sval[:100]
                    if sval in colspec[col]["top"]:
                        cnt[sval] += 1

        # finalize numeric
        for col, vals in num_aggs.items():
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            base = f"{tkey}.{col}"
            feats[f"{base}__count"] = float(arr.size)
            feats[f"{base}__mean"]  = float(np.mean(arr))
            feats[f"{base}__std"]   = float(np.std(arr, ddof=0))
            feats[f"{base}__min"]   = float(np.min(arr))
            feats[f"{base}__max"]   = float(np.max(arr))
        # finalize categorical
        for col, cnt in cat_counters.items():
            base = f"{tkey}.{col}"
            for val in schema[tkey][col]["top"]:
                feats[f"{base}={val}__count"] = float(cnt.get(val, 0))

    return feats

def build_labels_for_subject(obj, code_set):
    dx = obj.get("diagnoses_icd_labeled", []) or []
    present = {str(d.get("icd_code","")).strip() for d in dx}
    return {code: int(code in present) for code in code_set}

# Parse a feature name back to a *base column* for aggregation (column-level impact)
_feat_re_num = re.compile(r"^(?P<table>[^.]+)\.(?P<col>[^=]+?)__(count|mean|std|min|max)$")
_feat_re_cat = re.compile(r"^(?P<table>[^.]+)\.(?P<col>[^=]+?)=.*__count$")

def base_column_from_feature(feat_name):
    if feat_name.endswith(".__rows"):
        return feat_name  # treat table row count as its own "column"
    m = _feat_re_num.match(feat_name)
    if m:
        return f"{m.group('table')}.{m.group('col')}"
    m = _feat_re_cat.match(feat_name)
    if m:
        return f"{m.group('table')}.{m.group('col')}"
    return feat_name  # fallback

# ---------------- training ----------------
def train_xgb_per_label(X_train, X_test, Y_train, Y_test, feature_names, codes, params):
    per_label_importances = []
    metrics_rows = []

    for j, code in enumerate(codes):
        ytr = Y_train[:, j]
        n_pos = int(ytr.sum())
        n_neg = int(len(ytr) - n_pos)
        spw = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0

        clf = xgb.XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=RANDOM_SEED,
            n_jobs=params["n_jobs"],
            tree_method=params["tree_method"],
            predictor=params["predictor"],
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=spw,
            max_bin=params.get("max_bin", None)
        )

        print(f"[TRAIN] code={code}  (pos={n_pos}, neg={n_neg}, spw={spw:.2f})")
        clf.fit(X_train, ytr)

        booster = clf.get_booster()
        raw_imp = booster.get_score(importance_type="gain")
        imp_vec = np.zeros(len(feature_names), dtype=float)
        for fkey, val in raw_imp.items():   # "f123": gain
            try:
                idx = int(fkey[1:])
                if 0 <= idx < len(imp_vec):
                    imp_vec[idx] = float(val)
            except Exception:
                pass
        imp_series = pd.Series(imp_vec, index=feature_names)
        per_label_importances.append(imp_series)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        def safe(metric_fn, y_true, y_score):
            try:
                return metric_fn(y_true, y_score)
            except Exception:
                return float("nan")

        auroc = safe(roc_auc_score, Y_test[:, j], y_prob)
        ap    = safe(average_precision_score, Y_test[:, j], y_prob)
        acc   = accuracy_score(Y_test[:, j], y_pred)
        f1    = f1_score(Y_test[:, j], y_pred, zero_division=0)

        metrics_rows.append({"code": code, "pos_train": n_pos, "neg_train": n_neg,
                             "AUROC": auroc, "AP": ap, "ACC": acc, "F1": f1})

        # (optional) still write per-label top-25 *feature-level* for debugging
        top25 = imp_series[imp_series > 0].sort_values(ascending=False).head(25)
        out_csv = os.path.join(OUTPUT_DIR, f"top_features_xgb_columns_{code}.csv")
        top25.reset_index().rename(columns={"index":"feature", 0:"importance_gain"}).to_csv(out_csv, index=False)
        print(f"[WROTE] {out_csv}")

    return per_label_importances, pd.DataFrame(metrics_rows)

# ---------------- main ----------------
def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tree_method = "gpu_hist" if USE_GPU else "hist"
    predictor   = "gpu_predictor" if USE_GPU else "auto"

    print(f"[SCHEMA] discovering types/top-cats from {INPUT_JSONL}")
    schema = discover_schema(INPUT_JSONL)

    print("[BUILD] constructing matrices")
    feat_dicts, label_dicts, sids = [], [], []
    code_set = set(TARGET_CODES)

    for obj in load_subjects(INPUT_JSONL):
        sids.append(obj.get("subject_id"))
        feat_dicts.append(build_features_for_subject(obj, schema))
        label_dicts.append(build_labels_for_subject(obj, code_set))

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(feat_dicts)
    feature_names = vec.get_feature_names_out()
    Y = pd.DataFrame(label_dicts, columns=TARGET_CODES).astype(int).values

    # drop zero-variance features
    var_mask = (X.std(axis=0) > 0)
    X = X[:, var_mask]
    feature_names = feature_names[var_mask]
    print(f"[INFO] subjects={len(sids)}  features={len(feature_names)}")

    # split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=(Y.sum(axis=1) > 0)
    )

    params = dict(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        n_jobs=N_JOBS,
        tree_method=tree_method,
        predictor=predictor,
        max_bin=256 if USE_GPU else None
    )

    per_label_imps, metrics_df = train_xgb_per_label(
        X_train, X_test, Y_train, Y_test, feature_names, TARGET_CODES, params
    )

    # ----- Aggregate IMPORTANCE to COLUMN level -----
    # Build mapping feature -> base column
    base_map = {feat: base_column_from_feature(feat) for feat in feature_names}

    # Normalize per-label importances (sum=1) then aggregate by base column and average across labels
    if per_label_imps:
        col_frames = []
        for s in per_label_imps:
            s_sum = s.sum()
            s_norm = (s / s_sum) if s_sum > 0 else s
            # map to columns
            col_agg = defaultdict(float)
            for feat, val in s_norm.items():
                if val == 0: 
                    continue
                col_agg[base_map[feat]] += float(val)
            col_frames.append(pd.Series(col_agg))

        overall_cols = pd.concat(col_frames, axis=1).fillna(0).mean(axis=1)

        # Overall TOP 25 columns (most impactful)
        overall_top = overall_cols[overall_cols > 0].sort_values(ascending=False).head(25)
        out_csv_top = os.path.join(OUTPUT_DIR, "top_features_xgb_columns_overall.csv")
        overall_top.reset_index().rename(columns={"index":"column", 0:"mean_normalized_importance"}).to_csv(out_csv_top, index=False)
        print(f"[WROTE] {out_csv_top}")

        # Overall BOTTOM 50 columns (least impactful) — exclude exact zeros
        overall_least = overall_cols[overall_cols > 0].sort_values(ascending=True).head(50)
        out_csv_least = os.path.join(OUTPUT_DIR, "least_features_xgb_columns_overall.csv")
        overall_least.reset_index().rename(columns={"index":"column", 0:"mean_normalized_importance"}).to_csv(out_csv_least, index=False)
        print(f"[WROTE] {out_csv_least}")

    # metrics
    metrics_csv = os.path.join(OUTPUT_DIR, "xgb_label_metrics_columns.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[WROTE] {metrics_csv}")

    print("\n[DONE] Column-level feature selection (with exclusions) complete.")

if __name__ == "__main__":
    main()
