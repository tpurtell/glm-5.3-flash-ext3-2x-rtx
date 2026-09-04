#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=model-profiles.sh
source "${SCRIPT_DIR}/model-profiles.sh"
resolve_glm53_model_profile
DFLASH_MODEL_ID="${DFLASH_MODEL_ID:-incoai/GLM-5.3-Flash-DFlash2}"
DFLASH_MODEL_REVISION="${DFLASH_MODEL_REVISION:-bf582e4eacc1810f76656d1811693ff6c6737d2a}"
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
