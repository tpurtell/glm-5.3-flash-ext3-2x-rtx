#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1}"
MODEL_REVISION="${MODEL_REVISION:-8c02fb69a03ef86f0d1f9f0b607002c46102538c}"
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
