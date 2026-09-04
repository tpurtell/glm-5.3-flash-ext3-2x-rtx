#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the self-contained K3.25 checkpoint}"
MODEL="${MODEL:-wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1}"
IMAGE="${IMAGE:-ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:latest}"
PORT="${PORT:-8001}"
BASE_URL="http://127.0.0.1:${PORT}"
OUTPUT_DIR="${OUTPUT_DIR:-${RECIPE}/.qualification/replayssm}"
REQUESTS_PER_PHASE=40
CONCURRENCY=4
POWER_LIMIT_WATTS="${POWER_LIMIT_WATTS:-400}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-glm53-replayssm-qualification}"
active_container=""
qualification_failed=0

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR is not a directory: ${MODEL_DIR}" >&2
  exit 2
fi
MODEL_DIR="$(realpath -e -- "${MODEL_DIR}")"
command -v docker >/dev/null
command -v nvidia-smi >/dev/null
busy_pids="$({
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true
} | awk '$1 ~ /^[0-9]+$/ {print $1}' | sort -u)"
if [[ -n "${busy_pids}" ]]; then
  echo "ReplaySSM qualification requires idle GPUs; active PIDs: $(tr '\n' ' ' <<<"${busy_pids}")" >&2
  exit 2
fi
gpu_caps="$(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits)"
if ! awk -F, -v expected="${POWER_LIMIT_WATTS}" '
  BEGIN { count = 0; bad = 0 }
  {
    count += 1
    power = $2
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", power)
    delta = power - expected
    if (delta < 0) delta = -delta
    if (delta > 0.05) bad = 1
  }
  END { if (count != 2 || bad) exit 1 }
' <<<"${gpu_caps}"; then
  echo "ReplaySSM qualification requires exactly two GPUs at ${POWER_LIMIT_WATTS} W" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "OUTPUT_DIR already exists; refusing to mix receipts: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(realpath -e -- "${OUTPUT_DIR}")"
docker image inspect --format '{{.Id}}' "${IMAGE}" >"${OUTPUT_DIR}/candidate-image-id.txt"

docker run --rm \
  --gpus all \
  --ipc=host \
  --entrypoint python3 \
  --volume "${SCRIPT_DIR}/test-replayssm-kda-materializer.py:/tmp/test-replayssm-kda-materializer.py:ro" \
  --volume "${OUTPUT_DIR}:/results" \
  "${IMAGE}" \
  /tmp/test-replayssm-kda-materializer.py \
  --output /results/kda-materializer.json

cleanup() {
  if [[ -n "${active_container}" ]] && docker inspect "${active_container}" >/dev/null 2>&1; then
    docker stop --time 120 "${active_container}" >/dev/null 2>&1 || true
    docker rm "${active_container}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

wait_ready() {
  local container="$1"
  local deadline=$((SECONDS + 3600))
  while (( SECONDS < deadline )); do
    local running health
    running="$(docker inspect --format '{{.State.Running}}' "${container}" 2>/dev/null || true)"
    health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "${container}" 2>/dev/null || true)"
    if [[ "${running}" != true ]]; then
      docker logs "${container}" >&2 || true
      echo "${container} exited before readiness" >&2
      return 1
    fi
    if [[ "${health}" == healthy ]]; then
      return 0
    fi
    if [[ "${health}" == unhealthy ]]; then
      docker logs "${container}" >&2 || true
      echo "${container} became unhealthy" >&2
      return 1
    fi
    sleep 10
  done
  docker logs "${container}" >&2 || true
  echo "timed out waiting for ${container}" >&2
  return 1
}

run_variant() {
  local variant="$1"
  local use_replayssm="$2"
  local container="${CONTAINER_PREFIX}-${variant}-$$"
  local root="${OUTPUT_DIR}/${variant}"
  mkdir -p "${root}"
  active_container="${container}"

  MODEL_PROFILE=custom \
  MODEL_ID="${MODEL}" \
  MODEL_REVISION=local-materialized \
  MODEL_DIR_OVERRIDE="${MODEL_DIR}" \
  SERVED_MODEL_NAME="${MODEL}" \
  IMAGE="${IMAGE}" \
  CONTAINER_NAME="${container}" \
  PORT="${PORT}" \
  SPECULATIVE_METHOD=mtp \
  MTP_TOKENS=5 \
  ADAPTIVE_MTP=1 \
  ADAPTIVE_MTP_MIN_DEPTH=1 \
  USE_REPLAYSSM="${use_replayssm}" \
  REPLAYSSM_BUFFER_LEN=10 \
  KV_CACHE_PROFILE=fp8 \
  MAX_MODEL_LEN=262144 \
  MAX_NUM_BATCHED_TOKENS=2048 \
  MAX_NUM_SEQS=16 \
  MAX_CUDAGRAPH_CAPTURE_SIZE=64 \
  GPU_MEMORY_UTILIZATION=0.950 \
  GLM53_STARTUP_WARMUP=1 \
  "${RECIPE}/start.sh"

  wait_ready "${container}"
  python3 "${SCRIPT_DIR}/capture-release-environment.py" \
    --container "${container}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --power-limit "${POWER_LIMIT_WATTS}" \
    --output "${root}/environment.json"
  if ! python3 "${SCRIPT_DIR}/test-replayssm-stress.py" \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --requests-per-phase "${REQUESTS_PER_PHASE}" \
    --concurrency "${CONCURRENCY}" \
    --target-tokens 32768 \
    --max-tokens 512 \
    --output "${root}/stress"; then
    echo "${variant} rolling stress failed; retaining receipts and running control" >&2
    qualification_failed=1
  fi
  if ! python3 "${SCRIPT_DIR}/audit-startup-jit.py" \
    --container "${container}" \
    --output "${root}/startup-jit-audit.json"; then
    echo "${variant} startup-JIT audit failed; retaining receipts" >&2
    qualification_failed=1
  fi
  docker logs --timestamps "${container}" >"${root}/server.log" 2>&1
  docker stop --time 120 "${container}" >/dev/null
  docker rm "${container}" >/dev/null
  active_container=""
}

run_variant replayssm 1
run_variant full-state-control 0

if ! python3 "${SCRIPT_DIR}/verify-replayssm-qualification.py" \
  --root "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --power-limit "${POWER_LIMIT_WATTS}" \
  --output "${OUTPUT_DIR}/verification.json"; then
  qualification_failed=1
fi

exit "${qualification_failed}"
