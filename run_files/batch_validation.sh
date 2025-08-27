#!/usr/bin/env bash
set -euo pipefail

# ---------- Defaults (override via env or flags) ----------
PYBIN="${PYBIN:-/storage/cache/venvs/qlora/bin/python}"

OUT_ROOT="${OUT_ROOT:-/tmp/$USER}"
FILE="${FILE:-/storage/mimic_data/output/mimic_iv_subjects_sample10_with_dx_px.jsonl}"
SIDFILE="${SIDFILE:-/tmp/$USER/subject_ids.txt}"                   # pre-extracted subject IDs (one per line)
QFILE="${QFILE:-$HOME/run_files/questions_guidance.txt}"
SYS="${SYS:-$HOME/run_files/triage_system.txt}"
IDEAL="${IDEAL:-$HOME/run_files/ideal_guidance.jsonl}"
ADIR="${ADIR:-/home/fxp23/llama_ft/outputs/llama3-8b-qlora}"       # set ADIR="" for base-only
MODEL_ID="${MODEL_ID:-}"                                           # auto-detect if empty
N="${N:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
ECHO="${ECHO:-1}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options (all also configurable via env vars):
  --subjects-file PATH        JSONL with subjects (default: \$FILE)
  --subject-ids-file PATH     One subject_id per line (default: \$SIDFILE)
  --questions-file PATH       One question per line (default: \$QFILE)
  --system-prompt-file PATH   System prompt file (default: \$SYS)
  --ideal-file PATH           Rubric JSONL for ideal answers (default: \$IDEAL)
  --adapter-dir PATH          QLoRA adapter dir, "" for base-only (default: \$ADIR)
  --model-id PATH             Base model local path (default: auto-detect local snapshot)
  --python-bin PATH           Python to run (default: \$PYBIN)
  --n INT                     Number of trials (default: \$N)
  --out-root PATH             Output root dir (default: \$OUT_ROOT)
  --max-new-tokens INT        Max new tokens per answer (default: \$MAX_NEW_TOKENS)
  --echo INT                  0/1 print per-trial summaries (default: \$ECHO)
  -h, --help                  Show this help

Examples:
  N=10 ~/run_files/batch_validation.sh
  PYBIN=/storage/cache/venvs/qlora/bin/python N=25 ~/run_files/batch_validation.sh
  ADIR="" N=5 ~/run_files/batch_validation.sh                     # base-only benchmark
  SIDFILE=/tmp/$USER/subject_ids.txt N=20 ~/run_files/batch_validation.sh
EOF
}

# ---------- Parse flags ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subjects-file)       FILE="$2"; shift 2;;
    --subject-ids-file)    SIDFILE="$2"; shift 2;;
    --questions-file)      QFILE="$2"; shift 2;;
    --system-prompt-file)  SYS="$2"; shift 2;;
    --ideal-file)          IDEAL="$2"; shift 2;;
    --adapter-dir)         ADIR="$2"; shift 2;;
    --model-id)            MODEL_ID="$2"; shift 2;;
    --python-bin)          PYBIN="$2"; shift 2;;
    --n)                   N="$2"; shift 2;;
    --out-root)            OUT_ROOT="$2"; shift 2;;
    --max-new-tokens)      MAX_NEW_TOKENS="$2"; shift 2;;
    --echo)                ECHO="$2"; shift 2;;
    -h|--help)             usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# ---------- Auto-detect local snapshot if needed ----------
if [[ -z "$MODEL_ID" ]]; then
  MODEL_ID="$(ls -1d "$HOME"/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/* 2>/dev/null | tail -n1 || true)"
  [[ -n "$MODEL_ID" ]] || { echo "[ERR] Could not auto-detect local Llama 3.1 8B snapshot. Set MODEL_ID."; exit 1; }
fi

# ---------- Offline/cache env ----------
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# ---------- Sanity checks ----------
[[ -x "$PYBIN" || -f "$PYBIN" ]] || { echo "[ERR] python not found: $PYBIN"; exit 1; }
[[ -f "$FILE" ]] || { echo "[ERR] subjects file not found: $FILE"; exit 1; }
[[ -f "$SIDFILE" ]] || { echo "[ERR] subject IDs file not found: $SIDFILE"; exit 1; }
[[ -f "$QFILE" ]] || { echo "[ERR] questions file not found: $QFILE"; exit 1; }
[[ -f "$SYS" ]] || { echo "[ERR] system prompt file not found: $SYS"; exit 1; }
[[ -z "$ADIR" || -f "$ADIR/adapter_model.safetensors" ]] || { echo "[ERR] adapter_model.safetensors not found under: $ADIR"; exit 1; }

# ---------- Run ----------
echo "[INFO] Using:"
echo "  PYBIN      = $PYBIN"
echo "  MODEL_ID   = $MODEL_ID"
echo "  ADAPTER    = ${ADIR:-<base-only>}"
echo "  SUBJECTS   = $FILE"
echo "  SIDFILE    = $SIDFILE"
echo "  QFILE      = $QFILE"
echo "  SYS        = $SYS"
echo "  IDEAL      = $IDEAL"
echo "  N          = $N"
echo "  OUT_ROOT   = $OUT_ROOT"
echo

"$PYBIN" "$HOME/run_files/validation_batch.py" \
  --subjects-file "$FILE" \
  --subject-ids-file "$SIDFILE" \
  --questions-file "$QFILE" \
  --system-prompt-file "$SYS" \
  --ideal-file "$IDEAL" \
  --model-id "$MODEL_ID" \
  --adapter-dir "$ADIR" \
  --n "$N" \
  --out-root "$OUT_ROOT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --echo "$ECHO"
