#!/usr/bin/env python3
import os, json

# -------- CONFIG --------
INPUT_JSONL  = "/storage/mimic_data/output/mimic_iv_subjects_sample10_with_dx_px.jsonl"
OUTPUT_JSONL = "/storage/mimic_data/output/mimic_iv_subjects_sample10_ft_min.jsonl"
GZIP_OUTPUT  = False  # set True to write .jsonl.gz

# Columns to remove, expressed as "table.column" (case-insensitive).
# Missing fields are ignored gracefully.
REMOVE = {
    "prescriptions.doses_per_24_hrs",
    "procedures_icd_labeled.__rows",
    "pharmacy.duration",
    "emar.pharmacy_id",
    "microbiologyevents.chartdate",
    "microbiologyevents.dilution_value",
    "prescriptions.stoptime",
    "emar.scheduletime",
    "procedureevents.storetime",
    "icustays.__rows",
    "drgcodes.drg_severity",
    "icustays.intime",
    "icustays.outtime",
    "pharmacy.doses_per_24_hrs",
    "prescriptions.poe_seq",
    "inputevents.subject_id",
    "pharmacy.subject_id",
    "emar.hadm_id",
    "datetimeevents.value",
    "admissions.deathtime",
    "microbiologyevents.storedate",
    "drgcodes.drg_mortality",
    "omr.chartdate",
    "icustays.los",
    "pharmacy.basal_rate",
    "services.subject_id",
    "emar_detail.new_iv_bag_hung",
    "emar_detail.restart_interval",
    "transfers.subject_id",
    "pharmacy.fill_quantity",
    "procedureevents.continueinnextdept",
    "datetimeevents.__rows",
    "outputevents.__rows",
    "services.__rows",
    "labevents.__rows",
    "pharmacy.__rows",
    "procedureevents.__rows",
    "microbiologyevents.org_itemid",
    "drgcodes.__rows",
    "pharmacy.one_hr_max",
    "ingredientevents.__rows",
    "transfers.__rows",
    "hcpcsevents.__rows",
    "omr.__rows",
    "datetimeevents.warning",
    "inputevents.totalamountuom",
    "pharmacy.lockout_interval",
    "procedureevents.caregiver_id",
    "prescriptions.ndc",
    "poe_detail.__rows",
}

# Extra slimming options
DROP_EMPTY_TABLES = True   # remove list-typed tables that become empty after stripping
STRIP_NULLS       = True   # drop keys with None/null from row dicts


def should_drop(table_key: str, col: str) -> bool:
    key = f"{table_key}.{col}".lower()
    return key in REMOVE


def clean_row(table_key: str, row: dict) -> dict:
    if not isinstance(row, dict):
        return row
    out = {}
    for k, v in row.items():
        # keep everything not explicitly removed
        if should_drop(table_key, k):
            continue
        if STRIP_NULLS and v is None:
            continue
        out[k] = v
    return out


def process_subject(obj: dict) -> dict:
    # Walk every list-typed "table" and remove fields from each row
    new_obj = {}
    for k, v in obj.items():
        if isinstance(v, list):
            # strip per-row columns
            cleaned_rows = [clean_row(k, r) for r in v if isinstance(r, dict)] + [r for r in v if not isinstance(r, dict)]
            # optionally drop empty tables
            if DROP_EMPTY_TABLES:
                cleaned_rows = [r for r in cleaned_rows if (not isinstance(r, dict)) or (len(r) > 0)]
                if not cleaned_rows:
                    continue
            new_obj[k] = cleaned_rows
        else:
            # top-level scalar stays as-is (we’re removing only table.column pairs)
            new_obj[k] = v
    return new_obj


def main():
    # choose writer (optionally gzip)
    if GZIP_OUTPUT:
        import gzip
        out_fh = gzip.open(OUTPUT_JSONL if OUTPUT_JSONL.endswith(".gz") else OUTPUT_JSONL + ".gz",
                           "wt", encoding="utf-8")
        out_path = out_fh.name
    else:
        out_fh = open(OUTPUT_JSONL, "w", encoding="utf-8")
        out_path = OUTPUT_JSONL

    n_in = n_out = 0
    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, out_fh as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            obj = json.loads(line)
            obj2 = process_subject(obj)
            fout.write(json.dumps(obj2, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[DONE] {n_in} -> {n_out} subjects written to {out_path}")


if __name__ == "__main__":
    main()
