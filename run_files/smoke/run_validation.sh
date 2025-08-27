#!/usr/bin/env bash
set -euo pipefail

# Positional args
SUBJECT_ID="${1:-11999520}"
QUESTION="${2:-I have a cough - what should I do?}"

# Paths (all under run_files now)
ADIR="${ADIR:-/home/fxp23/llama_ft/outputs/llama3-8b-qlora}"   # "" for base-only
FILE="${FILE:-/storage/mimic_data/output/mimic_iv_subjects_sample10_with_dx_px.jsonl}"
SYS="${SYS:-$HOME/run_files/triage_system.txt}"
IDEAL="${IDEAL:-$HOME/run_files/ideal_guidance.jsonl}"
OUT_ROOT="${OUT_ROOT:-/tmp/$USER}"

# Local model snapshot (offline)
if [[ -z "${MODEL_ID:-}" ]]; then
  SNAP="$(ls -1d "$HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/"* 2>/dev/null | tail -n1 || true)"
  [[ -n "$SNAP" ]] || { echo "[ERR] No local Llama snapshot found. Set MODEL_ID to a local path." >&2; exit 1; }
  export MODEL_ID="$SNAP"
fi

# Offline + UTF-8
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Call the validator (moved here)
python3 "$HOME/run_files/validation_script.py" \
  --subjects-file "$FILE" \
  --subject-id "$SUBJECT_ID" \
  --question "$QUESTION" \
  ${SYS:+--system-prompt-file "$SYS"} \
  --adapter-dir "$ADIR" \
  --out-root "$OUT_ROOT" \
  ${IDEAL:+--ideal-file "$IDEAL"}
