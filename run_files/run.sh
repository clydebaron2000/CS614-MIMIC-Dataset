#!/usr/bin/env bash
set -euo pipefail

# --- Config you almost always want ---
PYBIN="${PYBIN:-/storage/cache/venvs/qlora/bin/python}"
SIDFILE="${SIDFILE:-/tmp/$USER/subject_ids.txt}"   # your pre-extracted IDs
N="${1:-20}"                                       # trials to run (default 20)

# Optional: override these via environment if you want different files
: "${QFILE:=$HOME/run_files/questions_guidance.txt}"
: "${SYS:=$HOME/run_files/triage_system.txt}"
: "${IDEAL:=$HOME/run_files/ideal_guidance.jsonl}"
: "${ADIR:=/home/fxp23/llama_ft/outputs/llama3-8b-qlora}"  # set ADIR="" for base-only

# Auto-detect your cached Llama 3.1 8B snapshot if MODEL_ID not provided
if [[ -z "${MODEL_ID:-}" ]]; then
  MODEL_ID="$(ls -1d "$HOME"/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/* 2>/dev/null | tail -n1 || true)"
  [[ -n "$MODEL_ID" ]] || { echo "[ERR] Could not auto-detect local snapshot; set MODEL_ID."; exit 1; }
fi

# Offline/cache env (so we never hit the Hub during runs)
export PYBIN MODEL_ID SIDFILE QFILE SYS IDEAL ADIR N
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Sanity checks
[[ -x "$PYBIN" || -f "$PYBIN" ]] || { echo "[ERR] python not found: $PYBIN"; exit 1; }
[[ -f "$SIDFILE" ]] || { echo "[ERR] subject IDs file not found: $SIDFILE"; exit 1; }
[[ -f "$QFILE" ]] || { echo "[ERR] questions file not found: $QFILE"; exit 1; }
[[ -f "$SYS" ]] || { echo "[ERR] system prompt file not found: $SYS"; exit 1; }
[[ -z "$ADIR" || -f "$ADIR/adapter_model.safetensors" ]] || { echo "[ERR] adapter_model.safetensors missing under $ADIR"; exit 1; }

# Go!
echo "[INFO] Running $N trials…"
PYBIN="$PYBIN" SIDFILE="$SIDFILE" N="$N" \
  QFILE="$QFILE" SYS="$SYS" IDEAL="$IDEAL" ADIR="$ADIR" MODEL_ID="$MODEL_ID" \
  "$HOME/run_files/batch_validation.sh"
