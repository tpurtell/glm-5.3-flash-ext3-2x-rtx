#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1}"
MODEL_REVISION="${MODEL_REVISION:-319d66a8b53092b491f698440ecea781e4ddd4e4}"
DFLASH_MODEL_ID="${DFLASH_MODEL_ID:-incoai/GLM-5.3-Flash-DFlash2}"
DFLASH_MODEL_REVISION="${DFLASH_MODEL_REVISION:-dc77ff1c99eeb2df044ee3d4f0094eb033fee410}"
HF_CACHE_DIR="${HF_HOME:-${HOME}/.cache/huggingface}"

command -v hf >/dev/null 2>&1 || {
  echo "The Hugging Face 'hf' CLI is required." >&2
  exit 1
}

snapshot="$(hf download "${MODEL_ID}" --revision "${MODEL_REVISION}" --cache-dir "${HF_CACHE_DIR}/hub")"
hf cache verify "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --cache-dir "${HF_CACHE_DIR}/hub" \
  --fail-on-missing-files

printf 'Verified standard-cache snapshot: %s\n' "${snapshot}"

dflash_snapshot="$(hf download "${DFLASH_MODEL_ID}" \
  --revision "${DFLASH_MODEL_REVISION}" \
  --cache-dir "${HF_CACHE_DIR}/hub")"
hf cache verify "${DFLASH_MODEL_ID}" \
  --revision "${DFLASH_MODEL_REVISION}" \
  --cache-dir "${HF_CACHE_DIR}/hub" \
  --fail-on-missing-files

printf 'Verified DFlash2 snapshot: %s\n' "${dflash_snapshot}"
